import os
import math
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from collections import namedtuple, defaultdict
from transformers import BertTokenizer, BertConfig, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from dataset import CollectionDataset, pack_tensor_2D
from modeling import RepBERT

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def create_embed_memmap(ids, memmap_dir, dim, file_prefix=''):

    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)

    embedding_path = os.path.join(memmap_dir, file_prefix + "embedding.memmap")
    id_path = os.path.join(memmap_dir, file_prefix + "ids.memmap")
    embed_open_mode = "r+" if os.path.exists(embedding_path) else "w+"  # 'w+' will initialize with 0s
    id_open_mode = "r+" if os.path.exists(id_path) else "w+"
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
    def __init__(self, tokenized_filepath, max_query_length):
        self.max_query_length = max_query_length
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

    def __init__(self, collection_memmap_dir, max_doc_length):
        self.max_doc_length = max_doc_length
        self.collection = CollectionDataset(collection_memmap_dir)
        self.pids = self.collection.pids
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.all_ids = self.collection.pids

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, item):
        pid = self.pids[item]
        doc_input_ids = self.collection[pid]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

        ret_val = {
            "input_ids": doc_input_ids,
            "id": pid
        }
        return ret_val


def get_collate_function():
    def collate_function(batch):
        input_ids_lst = [x["input_ids"] for x in batch]
        valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
        }
        id_lst = [x['id'] for x in batch]
        return data, id_lst

    return collate_function


def generate_embeddings(args, model, dataset):

    embedding_memmap, ids_memmap = create_embed_memmap(dataset.all_ids, args.memmap_dir, model.config.hidden_size)
    id2pos = {identity: i for i, identity in enumerate(ids_memmap)}

    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly  # TODO: So what?
    collate_fn = get_collate_function()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # TODO: Why not more workers?

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  # automatically splits the batch into chunks for each process/GPU
    # Eval!
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    start = timer()
    for batch, ids in tqdm(dataloader, desc="Computing embeddings"):
        model.eval()
        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            output = model(**batch)
            sequence_embeddings = output.detach().cpu().numpy()
            positions = [id2pos[identity] for identity in ids]
            embedding_memmap[positions] = sequence_embeddings
    end = timer()
    print("time:", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates memmap files containing embedding vectors of a document "
                                                 "collection and/or queries.")
    ## Required parameters
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--collection_memmap_dir", type=str, default=None,
                        help="""Contains memmap files of tokenized/numerized collection of documents. If specified, will 
                        generate corresponding document embeddings.""")
    parser.add_argument("--tokenized_queries", type=str, default=None,
                        help="""JSON file of tokenized/numerized queries. If specified, will generate corresponding
                         query embeddings""")
    # parser.add_argument("--tokenizer_type", type=str, choices=['bert', 'roberta'], default='bert',
    #                     help="""Type of tokenizer for the model component used for encoding queries (and passages)""")
    # parser.add_argument("--tokenizer_from", type=str, default=None,
    #                     help="""A path of a directory containing a saved custom tokenizer (vocabulary and added tokens).
    #                     It is optional and used together with `tokenizer_type`.""")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--per_gpu_batch_size", default=100, type=int)
    args = parser.parse_args()

    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = BertConfig.from_pretrained(args.load_model_path)

    if args.tokenized_queries:
        config.encode_type = "doc"
        logger.info("Loading model ...")
        model = RepBERT.from_pretrained(args.load_model_path, config=config)
        model.to(args.device)

        logger.info("Loading dataset ...")
        dataset = MSMARCO_QueryDataset(args.tokenized_queries, args.max_query_length)
        queries_dir = os.path.splitext(os.path.basename(args.tokenized_queries))[0] + "_embeddings_memmap"
        args.memmap_dir = os.path.join(args.output_dir, queries_dir)
        logger.info("Generating embeddings in {} ...".format(args.memmap_dir))
        generate_embeddings(args, model, dataset)
    if args.collection_memmap_dir:
        config.encode_type = "query"
        model = RepBERT.from_pretrained(args.load_model_path, config=config)
        model.to(args.device)

        logger.info("Loading dataset ...")
        dataset = MSMARCO_DocDataset(args.collection_memmap_dir, args.max_doc_length)
        args.memmap_dir = os.path.join(args.output_dir, "doc_embeddings_memmap")
        logger.info("Generating embeddings in {} ...".format(args.memmap_dir))
        generate_embeddings(args, model, dataset)
