"""
Creates memmap arrays (a triad of files) of:
    1) integer document IDs and respective integer *token IDs*
    2) integer query IDs and respective (retrieved) *document IDs*
The collection memmap triad is processed by `precompute.py` to obtain document vector embeddings/representations, which
are necessary for training and evaluation of the retrieval model (and are stored in another memmap).
The query_ID -> candidate_doc_IDs memmap triad is necessary for training the retrieval model (and can be used for reranking).
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import logging
import time

import utils

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def convert_collection_to_memmap(tokenized_file, memmap_dir, max_length, file_prefix=''):
    """
    Converts a line-JSON file containing a collection of passage/doc IDs (key: "id") and a respective list of tokens (key: "ids")
    into a triad of related memmap files inside the same directory:
    (collection_size, max_seq_length) array of integer token IDs
    (collection_size,) array of respective passage IDs
    (collection_size,) array of num. of tokens per passage ID
    """
    collection_size = sum(1 for _ in open(tokenized_file))

    tokenids_memmap_path = os.path.join(memmap_dir, file_prefix + "token_ids.memmap")
    pids_memmap_path = os.path.join(memmap_dir, file_prefix + "pids.memmap")
    lengths_memmap_path = os.path.join(memmap_dir, file_prefix + "lengths.memmap")

    # NOTE: int32 should be enough for token_ids, pids and lengths (max 2'147'483'647)
    token_ids = np.memmap(tokenids_memmap_path, dtype='int32', mode='w+', shape=(collection_size, max_length))
    pids = np.memmap(pids_memmap_path, dtype='int32', mode='w+', shape=(collection_size,))
    lengths = np.memmap(lengths_memmap_path, dtype='int32', mode='w+', shape=(collection_size,))

    for i, line in enumerate(tqdm(open(tokenized_file), desc="Documents", total=collection_size)):
        data = json.loads(line)
        # assert int(data['id']) == idx  # This holds for MSMARCO, but not generally
        pids[i] = int(data['id'])
        ids = data['ids'][:max_length]
        lengths[i] = len(data['ids'])
        token_ids[i, :lengths[i]] = ids
    return


def parse_line(line):

    fields = line.strip().split('\t')
    qid = int(fields[0])
    pid = int(fields[1])

    return qid, pid


# TODO: REALLY SLOW because of slow access of pandas multi-row indexing
# import pandas as pd
def pandas_create_candidates_memmap(candidates_filepath, memmap_dir, max_docs, file_prefix=''):
    """
    Load candidate (retrieved) documents/passages for each query from a file, and store them as a memmaped files.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).

    :param candidates_filepath: path to file of candidate (retrieved) documents/passages per query
    :param memmap_dir: directory where to create memmap files
    :param max_docs: maximun number of candidate documents per query.
    :param file_prefix: will pe prepended to memmap file names
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

    candidates_df = pd.read_csv(candidates_filepath, delimiter='\t', header=None, index_col=0, memory_map=True, dtype=np.int32)
    candidates_df = candidates_df.iloc[:, 0]  # select only 1st column (pIDs), ignoring ranking and scores
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


def create_candidates_memmap(candidates_filepath, memmap_dir, max_docs, num_qids=None, file_prefix=''):
    """
    Load candidate (retrieved) documents/passages for each query from a file, and store them as a memmaped files.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).

    Since the number of queries is not known, to pre-allocate memory for the memmap, this code first counts the lines
    of the candidate files.
    Candidates files most typically contain a fixed number of candidates (`max_docs`) per query, so in principle the
    number of rows (queries) should be num_lines/max_docs. However, this code also handles variable numbers per query
    (systematically up to ~50% less than `max_docs`, occasionally also many less), by pre-allocating excessive space
    and then shrinking the number of rows to the counted number of unique queries.

    :param candidates_filepath: path to file of candidate (retrieved) documents/passages per query
    :param memmap_dir: directory where to create memmap files
    :param max_docs: maximum expected number of documents per query.
        0s will be used for padding if less candidates are found for a query.
    :param num_qids: maximum expected number of queries. If not given, a bigger array will be allocated and then resized
    :param file_prefix: string to prepend to file names
    :return: None, but creates the following on disk:
        qid_memmap: (num_queries,) numpy memmap array containing query IDs
        lengths_memmap: (num_queries,) numpy memmap array containing number of candidates per query
        candidate_memmap: (num_queries, max_docs) numpy memmap array containing `max_docs` retrieved passage IDs (in
            order of decreasing relevance for query); the order of rows corresponds to qid_memmap
    """

    total_candidates = sum(1 for _ in open(candidates_filepath))
    if num_qids is None:
        num_qids = int(1.5*(total_candidates / max_docs))  # estimate (prob. overestimate) number of queries

    candidates_memmap_path = os.path.join(memmap_dir, file_prefix + "candidates.memmap")
    qid_memmap_path = os.path.join(memmap_dir, file_prefix + "qids.memmap")
    lengths_memmap_path = os.path.join(memmap_dir, file_prefix + "lengths.memmap")

    candidates_memmap = np.memmap(candidates_memmap_path, dtype='int32', mode='w+', shape=(num_qids, max_docs))
    qid_memmap = np.memmap(qid_memmap_path, dtype='int32', mode='w+', shape=(num_qids,))
    lengths_memmap = np.memmap(lengths_memmap_path, dtype='int32', mode='w+', shape=(num_qids,))

    with open(candidates_filepath, 'r') as f:

        qid, pid = parse_line(next(f))

        current_qid = qid
        query_ind = 0
        candidate_ind = 0
        candidates_memmap[query_ind, candidate_ind] = pid
        for line in tqdm(f, desc="Candidates", total=total_candidates-1):

            qid, pid = parse_line(line)
            candidate_ind += 1

            if qid != current_qid:
                qid_memmap[query_ind] = current_qid
                lengths_memmap[query_ind] = candidate_ind  # trick works because indices start from 0, but candidate_ind was just increased
                current_qid = qid
                query_ind += 1
                candidate_ind = 0

            candidates_memmap[query_ind, candidate_ind] = pid
        qid_memmap[query_ind] = current_qid
        lengths_memmap[query_ind] = candidate_ind + 1

    count_queries = query_ind + 1
    if num_qids != count_queries:  # shrink extra pre-allocated memory
        logger.warning("Will resize memmaps from {} rows (expected num. queries) to "
                       "{} (counted queries)".format(num_qids, count_queries))
        # candidates_memmap = np.require(candidates_memmap, requirements=['O'])  # to "own" the data
        candidates_memmap.base.resize(4 * count_queries * max_docs)  # x4 to convert to number of bytes
        qid_memmap.base.resize(4 * count_queries)  # x4 to convert to number of bytes
        lengths_memmap.base.resize(4 * count_queries)  # x4 to convert to number of bytes

    candidates_memmap.flush()  # make sure to write to disk
    qid_memmap.flush()  # make sure to write to disk
    lengths_memmap.flush()  # make sure to write to disk

    # NOTE: The numpy object is not aware of the correct shape at this point!
    # a new np.memmap(filename, dtype='int32', more='r') should be used to "refresh" it

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates binary memmap files out of a tokenized/numerized document collection"
                                                 "JSON file and/or a candidate (retrieved) documents text file. "
                                                 "Collection memmap (integer token ids per document) must be further "
                                                 "processed by `precompute.py` to obtain embedding vectors per document.")
    parser.add_argument("--tokenized_collection", type=str, default=None,
                        help="JSON file containing {docID : list of document token IDs} pairs, "
                             "produced by convert_text_to_tokenized.py")
    parser.add_argument("--candidates", type=str, default=None,
                        help="text file of candidate (retrieved) documents/passages per query, in the format "
                             "qID1 \t pID1\n qID1 \t pID2\n .... Only 1st column (pIDs) is used, rest are ignored. "
                             "This can be produced by e.g. Anserini  (This is output usually submitted to TREC)")
    parser.add_argument("--output_collection_dir", dest='collection_memmap_dir', type=str, default="collection_memmap",
                        help="Directory where memmap files will be created")
    parser.add_argument("--output_candidates_dir", dest='candidates_memmap_dir', type=str, default="candidates_memmap",
                        help="Directory where memmap files will be created")
    parser.add_argument("--auto_name_dir", action='store_true',
                        help="If set, will use `candidates` path to create the output directory name. "
                             "In this case, `output_candidates_dir` will specify the root.")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="The maximum length of each document in tokens. Affects memory footprint of memmap. "
                             "It is typically reduced later in the dataloader.")
    parser.add_argument("--max_candidates", type=int, default=1000,
                        help="The maximum number of candidate passages for each query. Affects memory footprint of memmap. "
                             "It is typically reduced later in the dataloader.")
    args = parser.parse_args()

    if args.tokenized_collection is not None:
        if not os.path.exists(args.collection_memmap_dir):
            os.makedirs(args.collection_memmap_dir)
        logger.info("Creating memmaps for collection in: {} ...".format(args.collection_memmap_dir))
        start_time = time.time()
        convert_collection_to_memmap(args.tokenized_collection, args.collection_memmap_dir, args.max_seq_length)
        runtime = time.time() - start_time
        logger.info("Runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(runtime)))

    if args.candidates is not None:
        if args.auto_name_dir:
            prefix = os.path.splitext(os.path.basename(args.candidates))[0]
            candidates_memmap_dir = os.path.join(args.candidates_memmap_dir, prefix + "_memmap")
        else:
            candidates_memmap_dir = args.candidates_memmap_dir
        if not os.path.exists(candidates_memmap_dir):
            os.makedirs(candidates_memmap_dir)
        logger.info("Creating memmaps for candidates in: {} ...".format(candidates_memmap_dir))
        start_time = time.time()
        create_candidates_memmap(args.candidates, candidates_memmap_dir, args.max_candidates)
        runtime = time.time() - start_time
        logger.info("Runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(runtime)))

    logger.info("All done!")
