#!/bin/bash
# Perform dense retrieval, store and evaluate output

set -e  # exit if any command fails

## SETTINGS
# Encoder model for query embedding
#MODEL_PATH="/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/SmoothL_temp-learn_boost1_top10_RNN-r67_scoresTASB_CODER_TASB_IZc_2023-03-18_21-30-48_2np/checkpoints/model_best_36000.pth"
MODEL_PATH="/users/gzerveas/data/gzerveas/MultidocScoringTr/NewExperiments/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_Axm_2021-11-12_16-42-08_IZc/checkpoints/model_best.pth"

ROOT="/users/gzerveas/data/TripClick/"
QUERIES_DIR=$ROOT"bertbase_reps_dot/processed/"  #"repbert/preprocessed/" # tokenization is shared between many models

DOC_EMB_DIR=$ROOT"tasb/representations/zeroshot/doc_embeddings_memmap/"
MODEL_ALIAS="tasb_zeroshot"
QUERY_EMB_DIR=$ROOT"tasb/representations/"$MODEL_ALIAS/
PRED_ROOT=$ROOT"tasb/predictions/"
PRED_PATH=$PRED_ROOT$MODEL_ALIAS".top1000"

# Evaluation
QRELS_PATH=$ROOT"collection/qrels_tsv/qrels.raw"
RELEVANCE_LEVEL=1

## SCRIPT

for part in "head" "torso" "tail"; do
  for split in "val" "test"; do
    # Embed queries
    python ~/Code/multidocscoring_transformer/precompute.py --model_type huggingface --encoder_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" --tokenized_queries "$QUERIES_DIR"topics."$part"."$split".json  --output_dir $QUERY_EMB_DIR --aggregation first --per_gpu_batch_size 512
    # Retrieve
    python ~/Code/multidocscoring_transformer/retrieve.py --doc_embedding_dir $DOC_EMB_DIR --query_embedding_dir $QUERY_EMB_DIR"topics.""$part"."$split"_embeddings_memmap/ --output_path "$PRED_PATH"."$part"."$split".tsv --per_gpu_doc_num 4000000;
    # Evaluate (pytrec_eval)
    python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$part"."$split".tsv --qrels_path "$QRELS_PATH"."$part"."$split".tsv --relevance_level $RELEVANCE_LEVEL --write_to_json --records_file $PRED_ROOT"records_""$part"."$split".xls
    # Evaluate with official trec_eval executable
    # awk '{print $1 " Q0 " $2 " " $3 " " $4 " nonsense"}' "$PRED_PATH"."$part"."$split".tsv > "$PRED_PATH"."$part"."$split".trec  # convert to trec format
    # EVAL_FILE="$PRED_PATH"."$part"."$split".eval
    # ~/data/MS_MARCO/trec2019/trec_eval-9.0.7/trec_eval -m all_trec "$QRELS_PATH"."$part"."$split".tsv "$PRED_PATH"."$part"."$split".trec > $EVAL_FILE
    # echo $EVAL_FILE
    # grep -E "recip|ndcg_cut_10|recall_10" $EVAL_FILE | tee -a $EVAL_FILE
  done
done

echo "All done!"