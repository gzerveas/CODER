"""
Used only for debugging
"""

import random
from collections import defaultdict
import sys
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


def randomize_queries(input_path, output_path):
    """
    Reads a file with scored candidate documents for each query, and writes a new file
    which picks a random query ID (from the set of existing query IDs) for each shuffled ist of candidates
    """
    score_dict = defaultdict(list)
    logger.info("Reading real ranks and scores from {} ...".format(input_path))
    for line in tqdm(open(input_path), desc="Line: "):
        query_id, doc_id = line.split("\t")[0:2]
        score_dict[int(query_id)].append(int(doc_id))
    logger.info("Writing false ranks and scores to {} ...".format(output_path))
    qids = list(score_dict.keys())
    random.shuffle(qids)
    with open(output_path, "w") as outFile:
        for i, candidates_lst in tqdm(enumerate(score_dict.values()), desc="Queries: "):
            query_id = qids[i]
            random.shuffle(candidates_lst)
            for rank_idx, doc_id in enumerate(candidates_lst):
                outFile.write("{}\t{}\t{}\t{}\n".format(query_id, doc_id, rank_idx + 1, 1/(rank_idx + 1)))


if __name__ == "__main__":

    randomize_queries(sys.argv[1], sys.argv[2])
    logger.info("Done!")