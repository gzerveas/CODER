#!/bin/bash
# Perform dense retrieval, store and evaluate output

set -e  # exit if any command fails

## SETTINGS
# Encoder model for query embedding
MODEL_PATH="/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_GeomOnly_rboost1.863_norm-std_top32_CODER-TASB-IZc_2023-04-26_00-37-53_ov6/checkpoints/model_best_34000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_GeomOnly_rboost1.308_norm-std_top27_CODER-TASB-IZc_2023-04-21_15-00-29_ZPi/checkpoints/model_best_44000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost8.843_norm-maxminmax_top69_CODER-TASB-IZc_2023-04-23_18-32-59_H0n/checkpoints/model_best_34000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost6.626_norm-maxminmax_top93_CODER-TASB-IZc_2023-04-23_04-46-20_Fqx/checkpoints/model_best_30000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost2.348_norm-std_top64_CODER-TASB-IZc_2023-04-23_11-06-49_8ln/checkpoints/model_best_30000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost7.582_norm-maxminmax_top67_CODER-TASB-IZc_2023-04-24_13-42-56_myD/checkpoints/model_best_34000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost6.373_norm-maxmin_top47_CODER-TASB-IZc_2023-04-21_21-04-09_4PF/checkpoints/model_best_34000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost5.061_norm-maxminmax_top94_CODER-TASB-IZc_2023-04-22_16-32-16_WJa/checkpoints/model_best_34000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/SmoothJ_JaccOnly_rboost5.061_norm-maxminmax_top94_keyLoss_CODER-TASB-IZc_2023-04-25_21-33-54_GVX/checkpoints/model_best_62000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_CODER-TASB-IZc_rboost3.191_norm-None_top4_2023-04-19_23-38-53_pF7/checkpoints/model_best_76000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNNr67_CODER-TASB-IZc_rboost2.4_norm-maxmin_top5_2023-04-19_09-06-41_vSC/checkpoints/model_best_44000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_CODER-TASB-IZc_rboost4.559_norm-maxminmax_top22_2023-04-20_08-53-48_bs7/checkpoints/model_best_76000.pth"
# MODEL_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNNr67_CODER-TASB-IZc_rboost1.222_norm-maxmin_top4_2023-04-19_00-18-16_35G/checkpoints/model_best_48000.pth"
# MODEL_PATH="/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/SmoothL_temp-learn_boost1_top10_RNN-r67_scoresTASB_CODER_TASB_IZc_2023-03-18_21-30-48_2np/checkpoints/model_best_36000.pth"

#MODEL_PATH="/users/gzerveas/data/gzerveas/MultidocScoringTr/NewExperiments/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_Axm_2021-11-12_16-42-08_IZc/checkpoints/model_best.pth"

MODEL_ALIAS="SmoothL_GeomOnly_rboost1.863_norm-std_top32_CODER-TASB-IZc_2023-04-26_00-37-53_ov6"
# MODEL_ALIAS="SmoothL_GeomOnly_rboost1.308_norm-std_top27_CODER-TASB-IZc_2023-04-21_15-00-29_ZPi"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost8.843_norm-maxminmax_top69_CODER-TASB-IZc_2023-04-23_18-32-59_H0n"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost6.626_norm-maxminmax_top93_CODER-TASB-IZc_2023-04-23_04-46-20_Fqx"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost2.348_norm-std_top64_CODER-TASB-IZc_2023-04-23_11-06-49_8ln"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost7.582_norm-maxminmax_top67_CODER-TASB-IZc_2023-04-24_13-42-56_myD"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost6.373_norm-maxmin_top47_CODER-TASB-IZc_2023-04-21_21-04-09_4PF"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost5.061_norm-maxminmax_top94_CODER-TASB-IZc_2023-04-22_16-32-16_WJa"
# MODEL_ALIAS="SmoothJ_JaccOnly_rboost5.061_norm-maxminmax_top94_keyLoss_CODER-TASB-IZc_2023-04-25_21-33-54_GVX"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost3.191_norm-None_top4_CODER-TASB-IZc_2023-04-19_23-38-53_pF7"
# MODEL_ALIAS="SmoothL_RNNr67_rboost2.4_norm-maxmin_top5_CODER-TASB-IZc_2023-04-19_09-06-41_vSC"
# MODEL_ALIAS="SmoothL_JaccOnly_rboost4.559_norm-maxminmax_top22_CODER-TASB-IZc_2023-04-20_08-53-48_bs7"
# MODEL_ALIAS="SmoothL_RNN-r67_rboost1.222_norm-maxmin_top4_CODER-TASB-IZc_2023-04-19_00-18-16_35G"
# MODEL_ALIAS="SmoothL_temp-learn_boost1_top10_RNN-r67_scoresTASB_CODER_TASB_IZc_2023-03-18_21-30-48_2np"

ROOT="/users/gzerveas/data/MS_MARCO/"
QUERIES_DIR=$ROOT"TAS-B/preprocessed/"  #"repbert/preprocessed/" # tokenization is shared between many models

DOC_EMB_DIR=$ROOT"TAS-B/representations/doc_embeddings_memmap"
QUERY_EMB_DIR=$ROOT"coder/representations/label_smoothing/"$MODEL_ALIAS/
PRED_ROOT=$ROOT"coder/predictions/"
PRED_PATH=$PRED_ROOT$MODEL_ALIAS".top1000"
RECORDS_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TREC/"

# Evaluation
# QRELS_PATH=$ROOT"/users/gzerveas/data/MS_MARCO/*2019/qrels."
# RELEVANCE_LEVEL=1

## SCRIPT

for dataset in "2019" "2020"; do
# Embed queries
python ~/Code/multidocscoring_transformer/precompute.py --encoder_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" --load_checkpoint $MODEL_PATH --tokenized_queries "$QUERIES_DIR"queries."$dataset"test.json  --output_dir $QUERY_EMB_DIR --aggregation first --per_gpu_batch_size 512
# Retrieve
python ~/Code/multidocscoring_transformer/retrieve.py --doc_embedding_dir $DOC_EMB_DIR --query_embedding_dir $QUERY_EMB_DIR"queries"."$dataset"test_embeddings_memmap/ --output_path "$PRED_PATH"."$dataset"test.tsv --per_gpu_doc_num 2000000
# Evaluate (pytrec_eval)
python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$dataset"test.tsv --qrels_path "$ROOT"trec$dataset/qrels."$dataset"test.tsv --relevance_level 1 --write_to_json --records_file $RECORDS_PATH"$dataset"test_records.xls
python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$dataset"test.tsv --qrels_path "$ROOT"trec$dataset/qrels."$dataset"test.tsv --relevance_level 2 --write_to_json --records_file $RECORDS_PATH"$dataset"test_str_records.xls
# Evaluate with official trec_eval executable
# awk '{print $1 " Q0 " $2 " " $3 " " $4 " nonsense"}' "$PRED_PATH"."$part"."$dataset".tsv > "$PRED_PATH"."$part"."$dataset".trec  # convert to trec format
# EVAL_FILE="$PRED_PATH"."$part"."$dataset".eval
# ~/data/MS_MARCO/trec2019/trec_eval-9.0.7/trec_eval -m all_trec "$QRELS_PATH"."$part"."$dataset".tsv "$PRED_PATH"."$part"."$dataset".trec > $EVAL_FILE
# echo $EVAL_FILE
# grep -E "recip|ndcg_cut_10|recall_10" $EVAL_FILE | tee -a $EVAL_FILE
done

echo "All done!"


# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2020test_records.xls  --relevance_level 1 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2020/qrels.2020test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2020test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2020test_str_records.xls  --relevance_level 2 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2020/qrels.2020test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2020test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2019test_records.xls  --relevance_level 1 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2019/qrels.2019test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2019test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2019test_str_records.xls  --relevance_level 2 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2019/qrels.2019test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2019test.tsv