"""
Creates memmap files containing embedding vectors of a document collection and/or queries.
Document embeddings are necessary for training and evaluation. Query embeddings are only used when evaluating a trained
retrieval model for 1-stage retrieval (`retrieve.py`)
Requires:
    For document embeddings: doc_ID -> doc_token_IDs *memmap triad files*, produced by `create_memmaps.py`
    For query embeddings: query_ID -> query_token_IDs *JSON file*, produced by `convert_text_to_tokenized`

"""

import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from transformers import BertTokenizer, BertConfig, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

import utils
from dataset import CollectionDataset, pack_tensor_2D
from modeling import RepBERT, get_aggregation_function
from utils import readable_time

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def create_embed_memmap(ids, memmap_dir, dim, file_prefix=''):
    """
    Creates memmap files containing embedding vectors of a document collection and/or queries.

    :param ids: iterable of all IDs
    :param memmap_dir: directory where memmap files will be created
    :param dim: dimensionality of embedding vectors
    :param file_prefix: optional prefix prepended to memmap files

    :return embedding_memmap: (len(ids), dim) numpy memmap array of embedding vectors
    :return id_memmap: (len(ids),) numpy memmap array of IDs corresponding to rows of `embedding_memmap`
    """

    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)

    embedding_path = os.path.join(memmap_dir, file_prefix + "embedding.memmap")
    id_path = os.path.join(memmap_dir, file_prefix + "ids.memmap")
    embed_open_mode = "w+"  # "r+" if os.path.exists(embedding_path) else "w+"  # 'w+' will initialize with 0s
    id_open_mode = "w+"  #"r+" if os.path.exists(id_path) else "w+"
    logger.warning(f"Open Mode: embeddings: {embed_open_mode}, ids: {id_open_mode}")

    embedding_memmap = np.memmap(embedding_path, dtype='float32', mode=embed_open_mode, shape=(len(ids), dim))
    id_memmap = np.memmap(id_path, dtype='int32', mode=id_open_mode, shape=(len(ids),))
    id_memmap[:] = ids[:]
    # # not writable
    # id_memmap = np.memmap(id_path, dtype='int32', shape=(len(ids),))  # TODO: Why do this again?
    id_memmap.flush()

    return embedding_memmap, id_memmap


def load_queries(filepath):
    queries = dict()
    for line in tqdm(open(filepath), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


class MSMARCO_QueryDataset(Dataset):
    def __init__(self, tokenized_filepath, max_query_length, tokenizer):
        self.max_query_length = max_query_length
        self.queries = load_queries(tokenized_filepath)
        self.qids = list(self.queries.keys())
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.all_ids = self.qids

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        query_input_ids = self.queries[qid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        ret_val = {
            "input_ids": query_input_ids,
            "id": qid
        }
        return ret_val


class MSMARCO_DocDataset(Dataset):
    """The difference from CollectionDataset is that it adds special tokens.
    It can also further limit max sequence length"""

    def __init__(self, collection_memmap_dir, max_doc_length, tokenizer):
        self.max_doc_length = max_doc_length
        self.collection = CollectionDataset(collection_memmap_dir)
        self.pids = self.collection.pids
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.all_ids = self.collection.pids

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, item):
        pid = self.pids[item]
        doc_input_ids = self.collection[pid]
        doc_input_ids = doc_input_ids[:self.max_doc_length - 2]  # -2 to account for START/END tokens
        doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

        ret_val = {
            "input_ids": doc_input_ids,
            "id": pid
        }
        return ret_val


def get_collate_function(model_type='repbert', token_type_id=None):

    if model_type == 'repbert':
        mask_key_string = "valid_mask"
    else:
        mask_key_string = 'attention_mask'

    def collate_function(batch):
        input_ids_lst = [x["input_ids"] for x in batch]
        valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            mask_key_string: pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
        }
        if token_type_id is not None:
            data['token_type_ids'] = torch.full_like(data['input_ids'], token_type_id, dtype=torch.int64)
        id_lst = [x['id'] for x in batch]
        return data, id_lst

    return collate_function


def generate_embeddings(args, model, dataset):

    embedding_memmap, ids_memmap = create_embed_memmap(dataset.all_ids, args.memmap_dir, model.config.hidden_size)
    id2pos = {identity: i for i, identity in enumerate(ids_memmap)}

    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    token_type_id = None
    if args.model_type != 'repbert':  # this will be used for all HuggingFace models
        aggregation_func = get_aggregation_function(args.aggregation)
        if args.token_type_ids:
            token_type_id = 0 if isinstance(dataset, MSMARCO_QueryDataset) else 1
    collate_fn = get_collate_function(args.model_type, token_type_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # TODO: Why not more workers?

    # multi-gpu inference
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  # automatically splits the batch into chunks for each process/GPU
    model.eval()

    # Inference
    logger.info("Num examples: %d", len(dataset))
    logger.info("Batch size: %d", batch_size)

    start = timer()
    for batch, ids in tqdm(dataloader, desc="Computing embeddings"):

        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            output = model(**batch)
            if args.model_type != 'repbert':
                sequence_embeddings = aggregation_func(output[0], batch['attention_mask']).detach().cpu().numpy()
            else:
                # RepBERT already aggregates (averages) the embeddings, and only returns tensor
                sequence_embeddings = output.detach().cpu().numpy()

            positions = [id2pos[identity] for identity in ids]
            embedding_memmap[positions] = sequence_embeddings
    end = timer()
    logger.info("Done in {} hours, {} minutes, {} seconds\n".format(*readable_time(end - start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates memmap files containing embedding vectors of a document "
                                                 "collection and/or queries.")
    ## Required parameters
    parser.add_argument("--model_type", type=str, choices=['repbert', 'huggingface'], default='huggingface',
                        help="""Type of the entire (end-to-end) information retrieval model""")
    parser.add_argument("--encoder_from", type=str,
                        default="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
                        help="""Name of built-in Huggingface encoder, or for `model_type` 'huggingface' it can also be 
                        a directory. 'repbert' always uses 'bert-base-uncased'.""")
    parser.add_argument("--tokenizer_from", type=str, default=None,
                        help="""Optional: name of built-in Huggingface tokenizer, or for `model_type` 'huggingface' it can also be 
                        a directory. If not specified, it will be the same as `encoder_from`.""")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="A path of a pre-trained model directory, `model_type` OTHER THAN 'huggingface'.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory where embedding memmaps will be created.")
    parser.add_argument("--collection_memmap_dir", type=str, default=None,
                        help="""Contains memmap files of tokenized/numerized collection of documents. If specified, will 
                        generate corresponding document embeddings.""")
    parser.add_argument("--tokenized_queries", type=str, default=None,
                        help="""JSON file of tokenized/numerized queries. If specified, will generate corresponding
                         query embeddings""")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--aggregation", type=str, choices=['first', 'mean'], default='first',
                        help="How to aggregate individual token embeddings")
    parser.add_argument("--token_type_ids", action='store_true',
                        help="If set, it will include a tensor of 'token_type_ids' when 'model_type' is not 'repbert', "
                             "to be used as input in a huggingface encoder. 0s denote a query sequence, "
                             "and 1s a document sequence.")
    parser.add_argument("--per_gpu_batch_size", default=100, type=int)
    args = parser.parse_args()
    
    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Get tokenizer and model
    if args.model_type == 'repbert':  # for 'RepBERT' model class
        if not(os.path.exists(args.load_checkpoint)):
            raise IOError("For 'repbert', `load_model` should point to a directory with a checkpoint.")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Loaded tokenizer: {}".format(tokenizer))
        config = BertConfig.from_pretrained(args.load_checkpoint)
        model = None
    else:
        if args.tokenizer_from is None:
            args.tokenizer_from = args.encoder_from  # for example, "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_from)
        logger.info("Loaded tokenizer: {}".format(tokenizer))

        logger.info("Loading encoder from: '{}' ...".format(args.encoder_from))
        model = AutoModel.from_pretrained(args.encoder_from)
        model.to(args.device)

        if args.load_checkpoint is not None:  # parameter loading for 'CODER' checkpoint
            logger.info("Loading encoder weights from: '{}' ...".format(args.load_checkpoint))
            model = utils.load_encoder(model, args.load_checkpoint, device)
            
        model.eval()

    if args.tokenized_queries:
        if args.model_type == 'repbert':
            config.encode_type = "query"
            logger.info("Loading model from '{}' ...".format(args.load_checkpoint))
            model = RepBERT.from_pretrained(args.load_checkpoint, config=config)
            model.to(args.device)

        logger.info("Loading dataset ...")
        dataset = MSMARCO_QueryDataset(args.tokenized_queries, args.max_query_length, tokenizer)
        queries_dir = os.path.splitext(os.path.basename(args.tokenized_queries))[0] + "_embeddings_memmap"
        args.memmap_dir = os.path.join(args.output_dir, queries_dir)
        logger.info("Generating embeddings in {} ...".format(args.memmap_dir))
        generate_embeddings(args, model, dataset)

    if args.collection_memmap_dir:
        if args.model_type == 'repbert':
            config.encode_type = "doc"
            logger.info("Loading model from '{}' ...".format(args.load_checkpoint))
            model = RepBERT.from_pretrained(args.load_checkpoint, config=config)
            model.to(args.device)

        logger.info("Loading dataset ...")
        dataset = MSMARCO_DocDataset(args.collection_memmap_dir, args.max_doc_length, tokenizer)
        args.memmap_dir = os.path.join(args.output_dir, "doc_embeddings_memmap")
        logger.info("Generating embeddings in {} ...".format(args.memmap_dir))
        generate_embeddings(args, model, dataset)
