import argparse
import os
import sys
from tqdm import tqdm
import pdb

from document_neutrality import DocumentNeutrality

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection-path', action='store', dest='collection_path',
                    default="/share/cp/datasets/ir/msmarco/passage/processed/collection.clean.tsv",
                    help='path the the collection file in tsv format (docid [tab] doctext)')
parser.add_argument('--representative-words-path', action='store', dest='representative_words_path',
                    default="resources/wordlist_protectedattribute_gender.txt",
                    help='path to the list of representative words which define the protected attribute')
parser.add_argument('--threshold', action='store', type=int, default=1,
                    help='threshold on the number of sensitive words')
parser.add_argument('--out-file', action='store', dest='out_file', 
                    default="/share/cp/datasets/ir/msmarco/passage/fair-retrieval-results/collection_with_neutrality.tsv",
                    help='output file containing docids and document neutrality scores')

args = parser.parse_args()




doc_neutrality = DocumentNeutrality(representative_words_path=args.representative_words_path,
                                  threshold=args.threshold,
                                  groups_portion={'f':0.5, 'm':0.5})

with open(args.out_file, "w", encoding="utf8") as fw:
    with open(args.collection_path, "r", encoding="utf8") as fr:
        for line in tqdm(fr):
            vals = line.strip().split('\t')
            if len(vals) != 2:
                print("Failed parsing the line (skipped):\n %s " % line.strip())
                continue
                
            docid = vals[0]
            doctext = vals[1]
            
            doctokens = doctext.lower().split(' ') # it is expected that the input document is already cleaned and pre-tokenized
            
            _neutrality = doc_neutrality.get_neutrality(doctokens)
            
            fw.write("%s\t%f\n" % (docid, _neutrality))



        



            