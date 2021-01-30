import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer


def get_tokenizer(args):
    """Initialize and return tokenizer object based on args"""

    if args.tokenizer_type == 'bert':
        if args.tokenizer_from is None:  # if not a directory path
            args.tokenizer_from = 'bert-base-uncased'
        return BertTokenizer.from_pretrained(args.tokenizer_from)
    elif args.tokenizer_type == 'roberta':
        if args.tokenizer_from is None:  # if not a directory path
            args.tokenizer_from = 'roberta-base'
        return RobertaTokenizer.from_pretrained(args.tokenizer_from)


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
    for mode in ["train", "dev", "eval", "eval.small"]:
        output_path = f"{args.output_dir}/queries.{mode}.json"
        tokenize_file(tokenizer, f"{args.msmarco_dir}/queries.{mode}.tsv", output_path)


def tokenize_collection(args, tokenizer):
    output_path = f"{args.output_dir}/collection.tokenize.json"
    tokenize_file(tokenizer, f"{args.msmarco_dir}/collection.tsv", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized")
    parser.add_argument("--tokenize_queries", action="store_true")
    parser.add_argument("--tokenize_collection", action="store_true")
    parser.add_argument("--tokenizer_type", type=str, choices=['bert', 'roberta'], default='bert',
                        help="""Type of tokenizer for the model component used for encoding queries (and passages)""")
    parser.add_argument("--tokenizer_from", type=str, default=None,
                        help="""A path of a directory containing a saved custom tokenizer (vocabulary and added tokens).
                        It is optional and used together with `tokenizer_type`.""")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = get_tokenizer(args)

    if args.tokenize_queries:
        tokenize_queries(args, tokenizer)  
    if args.tokenize_collection:
        tokenize_collection(args, tokenizer) 

    tokenizer.save_pretrained(args.output_dir)