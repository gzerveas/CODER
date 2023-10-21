#!/bin/bash
# Perform dense retrieval, store and evaluate output

set -e  # exit if any command fails

## SETTINGS

MAX_DOCS_GPU=3000000  # 2000000 for "large" GPUs, 1000000 (half) for "gpu-debug"

# Encoder model for query embedding
paths=(
/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/Uniform_SmoothL_among-all_CODER-TASB-IZc_weights_35G_2023-06-21_03-04-34_y0P/checkpoints/model_best_44000.pth
/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/Uniform_SmoothL_among-rand-max-inj4_CODER-TASB-IZc_weights_35G_2023-06-21_03-24-23_9xD/checkpoints/model_best_192000.pth

# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.766_norm-std_top37_lr1.44e-06_warmup12000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_03-51-06_4u0/checkpoints/model_best_48000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.043_norm-None_top3_lr2.94e-06_warmup15000_finlrr0.10_CODER-CoCo-IZc_2023-06-04_22-36-44_cbB/checkpoints/model_best_18000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost4.148_norm-maxmin_top5_lr4.87e-06_warmup8000_finlrr0.10_CODER-CoCo-IZc_2023-06-04_19-23-37_1hs/checkpoints/model_best_18000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.614_norm-std_top63_lr1.68e-06_warmup13000_finlrr0.10_CODER-CoCo-IZc_2023-06-07_17-15-28_hog/checkpoints/model_best_48000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.598_norm-std_top62_lr3.44e-06_warmup1000_finlrr0.10_CODER-CoCo-IZc_2023-06-07_18-31-16_vtl/checkpoints/model_best_30000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.525_norm-std_top34_lr1.37e-06_warmup12000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_14-47-18_UCV/checkpoints/model_best_48000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNN-Kb3_rboost1.385_norm-std_top55_lr1.85e-06_warmup1000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_06-34-43_uQ2/checkpoints/model_best_38000.pth"
# "/users/gzerveas/data/gzerveas/MultidocScoringTr/NewExperiments/NoDecoder_Qenc_cocodenser_augsep_cocodenser1000_Rneg0_ListnetLoss_Axm_IZc_config_2022-04-03_23-04-54_BHh/checkpoints/model_best_30000.pth"

# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_GeomOnly_rboost1.863_norm-std_top32_CODER-TASB-IZc_2023-04-26_00-37-53_ov6/checkpoints/model_best_34000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_GeomOnly_rboost1.308_norm-std_top27_CODER-TASB-IZc_2023-04-21_15-00-29_ZPi/checkpoints/model_best_44000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost8.843_norm-maxminmax_top69_CODER-TASB-IZc_2023-04-23_18-32-59_H0n/checkpoints/model_best_34000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost6.626_norm-maxminmax_top93_CODER-TASB-IZc_2023-04-23_04-46-20_Fqx/checkpoints/model_best_30000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost2.348_norm-std_top64_CODER-TASB-IZc_2023-04-23_11-06-49_8ln/checkpoints/model_best_30000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost7.582_norm-maxminmax_top67_CODER-TASB-IZc_2023-04-24_13-42-56_myD/checkpoints/model_best_34000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost6.373_norm-maxmin_top47_CODER-TASB-IZc_2023-04-21_21-04-09_4PF/checkpoints/model_best_34000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_rboost5.061_norm-maxminmax_top94_CODER-TASB-IZc_2023-04-22_16-32-16_WJa/checkpoints/model_best_34000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/SmoothJ_JaccOnly_rboost5.061_norm-maxminmax_top94_keyLoss_CODER-TASB-IZc_2023-04-25_21-33-54_GVX/checkpoints/model_best_62000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_CODER-TASB-IZc_rboost3.191_norm-None_top4_2023-04-19_23-38-53_pF7/checkpoints/model_best_76000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNNr67_CODER-TASB-IZc_rboost2.4_norm-maxmin_top5_2023-04-19_09-06-41_vSC/checkpoints/model_best_44000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_JaccOnly_CODER-TASB-IZc_rboost4.559_norm-maxminmax_top22_2023-04-20_08-53-48_bs7/checkpoints/model_best_76000.pth"
# "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/TrainingExperiments/AUTO_RNNr67_CODER-TASB-IZc_rboost1.222_norm-maxmin_top4_2023-04-19_00-18-16_35G/checkpoints/model_best_48000.pth"
# "/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/TrainingExperiments/SmoothL_temp-learn_boost1_top10_RNN-r67_scoresTASB_CODER_TASB_IZc_2023-03-18_21-30-48_2np/checkpoints/model_best_36000.pth"

#"/users/gzerveas/data/gzerveas/MultidocScoringTr/NewExperiments/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_Axm_2021-11-12_16-42-08_IZc/checkpoints/model_best.pth"
)

aliases=(

Uniform_SmoothL_among-all_CODER-TASB-IZc_weights_35G_2023-06-21_03-04-34_y0P
Uniform_SmoothL_among-rand-max-inj4_CODER-TASB-IZc_weights_35G_2023-06-21_03-24-23_9xD

# SmoothL_RNN-Kb3_rboost1.766_norm-std_top37_lr1.44e-06_warmup12000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_03-51-06_4u0
# SmoothL_RNN-Kb3_rboost1.043_norm-None_top3_lr2.94e-06_warmup15000_finlrr0.10_CODER-CoCo-IZc_2023-06-04_22-36-44_cbB
# SmoothL_RNN-Kb3_rboost4.148_norm-maxmin_top5_lr4.87e-06_warmup8000_finlrr0.10_CODER-CoCo-IZc_2023-06-04_19-23-37_1hs
# "SmoothL_RNN-Kb3_rboost1.614_norm-std_top63_lr1.68e-06_warmup13000_finlrr0.10_CODER-CoCo-IZc_2023-06-07_17-15-28_hog"
# "SmoothL_RNN-Kb3_rboost1.598_norm-std_top62_lr3.44e-06_warmup1000_finlrr0.10_CODER-CoCo-IZc_2023-06-07_18-31-16_vtl"
# "SmoothL_RNN-Kb3_rboost1.525_norm-std_top34_lr1.37e-06_warmup12000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_14-47-18_UCV"
# "SmoothL_RNN-Kb3_rboost1.385_norm-std_top55_lr1.85e-06_warmup1000_finlrr0.10_CODER-CoCo-IZc_2023-06-06_06-34-43_uQ2"
# "coder_cocondenser_IZc"  # CODER(CoCondenser) using IZc config 
# "cocodenser_retriever.augsep"  # augsep is the best cocondenser for trec2019, but without augsep the best for trec2020!


# "SmoothL_GeomOnly_rboost1.863_norm-std_top32_CODER-TASB-IZc_2023-04-26_00-37-53_ov6"
# "SmoothL_GeomOnly_rboost1.308_norm-std_top27_CODER-TASB-IZc_2023-04-21_15-00-29_ZPi"
# "SmoothL_JaccOnly_rboost8.843_norm-maxminmax_top69_CODER-TASB-IZc_2023-04-23_18-32-59_H0n"
# "SmoothL_JaccOnly_rboost6.626_norm-maxminmax_top93_CODER-TASB-IZc_2023-04-23_04-46-20_Fqx"
# "SmoothL_JaccOnly_rboost2.348_norm-std_top64_CODER-TASB-IZc_2023-04-23_11-06-49_8ln"
# "SmoothL_JaccOnly_rboost7.582_norm-maxminmax_top67_CODER-TASB-IZc_2023-04-24_13-42-56_myD"
# "SmoothL_JaccOnly_rboost6.373_norm-maxmin_top47_CODER-TASB-IZc_2023-04-21_21-04-09_4PF"
# "SmoothL_JaccOnly_rboost5.061_norm-maxminmax_top94_CODER-TASB-IZc_2023-04-22_16-32-16_WJa"
# "SmoothJ_JaccOnly_rboost5.061_norm-maxminmax_top94_keyLoss_CODER-TASB-IZc_2023-04-25_21-33-54_GVX"
# "SmoothL_JaccOnly_rboost3.191_norm-None_top4_CODER-TASB-IZc_2023-04-19_23-38-53_pF7"
# "SmoothL_RNNr67_rboost2.4_norm-maxmin_top5_CODER-TASB-IZc_2023-04-19_09-06-41_vSC"
# "SmoothL_JaccOnly_rboost4.559_norm-maxminmax_top22_CODER-TASB-IZc_2023-04-20_08-53-48_bs7"
# "SmoothL_RNN-r67_rboost1.222_norm-maxmin_top4_CODER-TASB-IZc_2023-04-19_00-18-16_35G"
# "SmoothL_temp-learn_boost1_top10_RNN-r67_scoresTASB_CODER_TASB_IZc_2023-03-18_21-30-48_2np"
)

ROOT="/users/gzerveas/data/MS_MARCO/"
QUERIES_DIR=$ROOT"repbert/preprocessed/"  #"TAS-B/preprocessed/"  # tokenization is shared between many models

DOC_EMB_DIR=$ROOT"TAS-B/representations/doc_embeddings_memmap/" #"cocodenser/representations_retriever/collection_augsep_embeddings_memmap/" # "TAS-B/representations/doc_embeddings_memmap/"
QUERY_EMB_DIR=$ROOT"coder/representations/label_smoothing/" #"cocodenser/representations_retriever/"  #"coder/representations/"  #"coder/representations/label_smoothing/
PRED_ROOT=$ROOT"coder/predictions/Dense_Retrieval/label_smoothing/" # "cocodenser/predictions/"  #"coder/predictions/"
RECORDS_PATH="/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/evaluation_records/MS_MARCO/"

# Evaluation
# QRELS_PATH=$ROOT"/users/gzerveas/data/MS_MARCO/*2019/qrels."
# RELEVANCE_LEVEL=1

## SCRIPT

for i in "${!paths[@]}"; do

  MODEL_PATH=${paths[$i]}
  MODEL_ALIAS=${aliases[$i]}
  QUERY_EMB_DIR=$QUERY_EMB_DIR"$MODEL_ALIAS/"
  PRED_PATH=$PRED_ROOT$MODEL_ALIAS".top1000"

  echo "Evaluating: $MODEL_ALIAS"

  for dataset in "2019test" "2020test" "dev.small"; do #"dev"; do
    # Embed queries
    orig_huggingface_model="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    # orig_huggingface_model="Luyu/co-condenser-marco"
    python ~/Code/multidocscoring_transformer/precompute.py --encoder_from $orig_huggingface_model --load_checkpoint $MODEL_PATH --tokenized_queries "$QUERIES_DIR"queries."$dataset".json  --output_dir $QUERY_EMB_DIR --aggregation first --per_gpu_batch_size 512
    # Retrieve
    python ~/Code/multidocscoring_transformer/retrieve.py --doc_embedding_dir $DOC_EMB_DIR --query_embedding_dir $QUERY_EMB_DIR"queries"."$dataset"_embeddings_memmap/ --output_path "$PRED_PATH"."$dataset".tsv --per_gpu_doc_num $MAX_DOCS_GPU
    # Evaluate (pytrec_eval)
    if [[ "$dataset" == *test ]]; then
      python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$dataset".tsv --qrels_path "$ROOT"trec${dataset:0:4}/qrels."$dataset".tsv --relevance_level 1 --write_to_json --records_file $RECORDS_PATH"$dataset"_records.xls
      python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$dataset".tsv --qrels_path "$ROOT"trec${dataset:0:4}/qrels."$dataset".tsv --relevance_level 2 --write_to_json --records_file $RECORDS_PATH"$dataset"_str_records.xls
    else
      python ~/Code/multidocscoring_transformer/trec_eval_all.py --pred_path "$PRED_PATH"."$dataset".tsv --qrels_path "$ROOT"qrels."$dataset".tsv --relevance_level 1 --write_to_json --records_file $RECORDS_PATH"$dataset"_records.xls
    fi
    # Evaluate with official trec_eval executable
    # awk '{print $1 " Q0 " $2 " " $3 " " $4 " nonsense"}' "$PRED_PATH"."$part"."$dataset".tsv > "$PRED_PATH"."$part"."$dataset".trec  # convert to trec format
    # EVAL_FILE="$PRED_PATH"."$part"."$dataset".eval
    # ~/data/MS_MARCO/trec2019/trec_eval-9.0.7/trec_eval -m all_trec "$QRELS_PATH"."$part"."$dataset".tsv "$PRED_PATH"."$part"."$dataset".trec > $EVAL_FILE
    # echo $EVAL_FILE
    # grep -E "recip|ndcg_cut_10|recall_10" $EVAL_FILE | tee -a $EVAL_FILE
  done
  echo "Evaluation of $MODEL_ALIAS done"
done
echo "All done!"


# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2020test_records.xls  --relevance_level 1 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2020/qrels.2020test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2020test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2020test_str_records.xls  --relevance_level 2 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2020/qrels.2020test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2020test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2019test_records.xls  --relevance_level 1 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2019/qrels.2019test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2019test.tsv
# python ~/Code/multidocscoring_transformer/trec_eval_all.py --records_file /gpfs/data/ceickhof/gzerveas/RecipNN/2019test_str_records.xls  --relevance_level 2 --qrels_path /gpfs/data/ceickhof/MS_MARCO/trec2019/qrels.2019test.tsv --pred_path ~/data/MS_MARCO/coder/predictions/Dense_Retrieval/NoDecoder_Qenc_tasb_aggFirst_tasb1000_Rneg0_ListnetLoss_ROPloss_valTASB_IZc_top1000.2019test.tsv