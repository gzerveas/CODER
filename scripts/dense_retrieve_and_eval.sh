#!/bin/bash
# Perform dense retrieval, store and evaluate output

ROOT="/users/gzerveas/data/TripClick/"
DOC_EMB_DIR=$ROOT"bertbase_reps_dot/representations_with_types/doc_embeddings_memmap"
QUERY_EMB_DIR=$ROOT"bertbase_reps_dot/representations_with_types/topics"
PRED_PATH=$ROOT"bertbase_reps_dot/predictions/BertDot.top1000"
QRELS_PATH=$ROOT"collection/qrels_tsv/qrels.raw"

for part in "head" "torso" "tail"; do
  for split in "val" "test"; do
    # Retrieve
    python ~/Code/multidocscoring_transformer/retrieve.py --doc_embedding_dir $DOC_EMB_DIR --query_embedding_dir $QUERY_EMB_DIR"$part"."$split"_embeddings_memmap/ --output_path "$PRED_PATH"."$part"."$split".tsv --per_gpu_doc_num 4000000;
    # Reformat
    awk '{print $1 " Q0 " $2 " " $3 " " $4 " nonsense"}' "$PRED_PATH"."$part"."$split".tsv > "$PRED_PATH"."$part"."$split".trec
    # Evaluate
    ~/data/MS_MARCO/trec2019/trec_eval-9.0.7/trec_eval -m all_trec "$QRELS_PATH"."$part"."$split".tsv "$PRED_PATH"."$part"."$split".trec > "$PRED_PATH"."$part"."$split".eval
    echo "$PRED_PATH"."$part"."$split".eval
    grep -E "recip|ndcg_cut_10|recall_10" "$PRED_PATH"."$part"."$split".eval
  done
done