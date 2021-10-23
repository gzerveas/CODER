"""
Converts documents in a collection and/or text queries into JSON files containing in each line
an ID and the corresponding sequence (list) of token IDs: {"id": seq_id, "ids": ids}"
"""

import os
import json
import argparse
import glob
import re
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


def get_tokenizer(args):
    """Initialize and return tokenizer object based on args"""

    if os.path.exists(args.tokenizer_from):
        logger.info("Will load tokenizer from: {}".format(args.tokenizer_from))
    else:
        logger.info("Will use standard HuggingFace '{}' as a tokenizer".format(args.tokenizer_from))

    return AutoTokenizer.from_pretrained(args.tokenizer_from)


def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))  # simply to get number of lines
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size, desc=f"Tokenize: {os.path.basename(input_file)}"):
            seq_id, text = line.split("\t")
            tokens = tokenizer.tokenize(text)  # does NOT add special "BOS"/"EOS" tokens
            ids = tokenizer.convert_tokens_to_ids(tokens)
            outFile.write(json.dumps({"id": seq_id, "ids": ids}))
            outFile.write("\n")
    

def tokenize_queries(args, tokenizer):

    if os.path.isdir(args.queries):
        data_paths = glob.glob(os.path.join(args.queries, '*'))  # list of all paths
        selected_paths = list(filter(lambda x: re.search(r"queries\..*\.tsv", x), data_paths))
    else:
        selected_paths = [args.queries]

    for input_path in selected_paths:
        output_path = os.path.join(args.output_dir, os.path.basename(input_path)[:-4] + '.json')
        tokenize_file(tokenizer, input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts documents in a collection and/or text queries into "
                                                 "JSON files containing sequences (lists) of token IDs")
    parser.add_argument("--output_dir", type=str, default="tokenized",
                        help='Path of directory where to write tokenized collection and/or queries in JSON format.')
    parser.add_argument("--queries", type=str, default=None,
                        help="Path of text queries. If a directory, will tokenize all 'queries.*.tsv' found within, "
                             "otherwise the specified file path. If flag is not used, no queries will be tokenized.")
    parser.add_argument("--collection", type=str, default=None,
                        help="Path of the text file containing the document collection. "
                             "If flag is not used, no collection will be tokenized.")
    parser.add_argument("--tokenizer_type", type=str, choices=['bert', 'roberta'], default=None,
                        help="""Type of tokenizer for the model component used for encoding queries (and passages).
                        Optional, mainly to be used when loading a custom tokenizer configuration from a directory 
                        using `tokenizer_from`.
                        If not set, then `tokenizer_from` should be used to initialize one of the built-in HuggingFace tokenizers.""")
    parser.add_argument("--tokenizer_from", type=str, default=None,
                        help="""When used together with `tokenizer_type`, it is an optional path of a directory 
                        containing a saved custom tokenizer (vocabulary and added tokens). When `tokenizer_type` is not
                        set, it is a string used to initialize one of the built-in HuggingFace tokenizers.""")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = get_tokenizer(args)
    logger.info("Loaded tokenizer: {}".format(tokenizer))

    if args.queries is not None:
        tokenize_queries(args, tokenizer)  
    if args.collection is not None:
        tokenize_file(tokenizer, args.collection, os.path.join(args.output_dir, 'collection.tokenized.json'))

    tokenizer_config_path = os.path.join(args.output_dir, "tokenizer_config")
    if tokenizer_config_path != args.tokenizer_from:
        tokenizer.save_pretrained(tokenizer_config_path)
        logger.info("Saved tokenizer config in: {}".format(tokenizer_config_path))
    logger.info("Done!")
