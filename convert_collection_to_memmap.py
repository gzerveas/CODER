import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def cvt_collection_to_memmap(args):
    collection_size = sum(1 for _ in open(args.tokenized_collection))
    token_ids = np.memmap(f"{args.output_dir}/token_ids.memmap", dtype='int32',
                          mode='w+', shape=(collection_size, args.max_seq_length))
    pids = np.memmap(f"{args.output_dir}/pids.memmap", dtype='int32',
                     mode='w+', shape=(collection_size,))
    lengths = np.memmap(f"{args.output_dir}/lengths.memmap", dtype='int32',
                        mode='w+', shape=(collection_size,))

    for idx, line in enumerate(tqdm(open(args.tokenized_collection),
                                    desc="collection", total=collection_size)):
        data = json.loads(line)
        assert int(data['id']) == idx
        pids[idx] = idx
        lengths[idx] = len(data['ids'])
        ids = data['ids'][:args.max_seq_length]
        token_ids[idx, :lengths[idx]] = ids
    return


def create_candidates_memmap(candidates_filepath, memmap_dir, max_docs, file_prefix=''):
    """
    Load candidate (retrieved) documents/passages for each query from a file, and store them as a memmaped files.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).

    :param candidates_filepath: path to file of candidate (retrieved) documents/passages per query
    :param memmap_dir:
    :param max_docs:
    :param file_prefix:
    :return:
        qid_memmap: (num_queries,) numpy memmap array containing query IDs
        candidate_memmap: (num_queries, max_docs) numpy memmap array containing `max_docs` retrieved passage IDs (in
            order of decreasing relevance for query); the order of rows corresponds to qid_memmap
    """

    candidates_memmap_path = os.path.join(memmap_dir, file_prefix + "candidates.memmap")
    qid_memmap_path = os.path.join(memmap_dir, file_prefix + "qids.memmap")
    candidates_open_mode = "r+" if os.path.exists(candidates_memmap_path) else "w+"  # 'w+' will initialize with 0s
    qid_open_mode = "r+" if os.path.exists(qid_memmap_path) else "w+"  # 'w+' will initialize with 0s
    logger.warning(f"Open Mode: candidate pIDs: {candidates_open_mode}, qIDs: {qid_open_mode}")

    candidates_df = pd.read_csv(candidates_filepath, delimiter='\t', index_col=0, memory_map=True, dtype=np.int32)
    qids = candidates_df.index.unique()

    candidates_memmap = np.memmap(candidates_memmap_path, dtype='int32', mode=candidates_open_mode,
                                  shape=(len(qids), max_docs))
    qid_memmap = np.memmap(qid_memmap_path, dtype='int32', mode=qid_open_mode, shape=(len(qids),))
    qid_memmap[:] = qids[:]
    qid_memmap.flush()  # make sure to write to disk (probably redundant)

    for i, qid in enumerate(qids):
        pids = candidates_df.loc[qid]  # Series containing PIDs corresponding to given qID
        candidates_memmap[i, :len(pids)] = pids
    candidates_memmap.flush()  # make sure to write to disk (probably redundant)

    return qid_memmap, candidates_memmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_collection", type=str,
                        default="./data/tokenize/collection.tokenize.json")
    parser.add_argument("--output_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="The maximum length of each document in tokens. Affects memory footprint of memmap. "
                             "Is typically reduced later when defining max. model input length.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    cvt_collection_to_memmap(args)
