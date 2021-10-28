import random
from collections import defaultdict
import sys
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


def convert_scores_to_ranks(input_path, output_path):
    """
    Reads a file with scored candidate documents for each query, and writes a new file
    which sorts candidates per query by score and adds a rank field
    """
    score_dict = defaultdict(list)
    logger.info("Reading scores from {} ...".format(input_path))
    for line in tqdm(open(input_path), desc="Line: "):
        query_id, doc_id, score = line.split("\t")
        score_dict[int(query_id)].append((float(score), int(doc_id)))
    logger.info("Writing ranks and scores to {} ...".format(output_path))
    with open(output_path, "w") as outFile:
        for query_id, candidates_lst in tqdm(score_dict.items(), desc="Query: "):
            random.shuffle(candidates_lst)
            candidates_lst = sorted(candidates_lst, key=lambda x: x[0], reverse=True)
            for rank_idx, (score, doc_id) in enumerate(candidates_lst):
                outFile.write("{}\t{}\t{}\t{}\n".format(query_id, doc_id, rank_idx + 1, score))


if __name__ == "__main__":

    convert_scores_to_ranks(sys.argv[1], sys.argv[2])
