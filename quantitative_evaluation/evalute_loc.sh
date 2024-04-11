# evalute_loc.sh

export TASK_NAME="loc"
export DATASET="siq2"
export SPLIT="val" # never train because why? # train, val, test
export MODEL_NAME="LLaVA-Lightning-7B-v1-1" # TODO
export PROJECT_ROOT="${HOME}/vtom"
export EXPERIMENT_NAME="${DATASET}_${TASK_NAME}_${SPLIT}_${MODEL_NAME}"
export EXPERIMENT_DIRPATH="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}"
export OUTPUT_PER_VIDEO_DIRPATH="${EXPERIMENT_DIRPATH}/output_per_video"
export OUTPUT_DIRPATH="${EXPERIMENT_DIRPATH}/output"
export VIDEO_DIR="${PROJECT_ROOT}/data/${DATASET}/video"
export LOC_DIRPATH="${PROJECT_ROOT}/data/siq2/loc"
export PRED_FPATH="preds_${EXPERIMENT_NAME}"
export RESULTS_FPATH="results_${EXPERIMENT_NAME}.json"
export GT_WITH_TS_FPATH="${LOC_DIRPATH}/loc_${SPLIT}_with_ts.json"
export INSTRUCTION_FINETUNING_FPATH="${LOC_DIRPATH}/loc_${SPLIT}_instruction_with_ts.json"


mkdir -p "${OUTPUT_PER_VIDEO_DIRPATH}"
mkdir -p "${OUTPUT_DIRPATH}"


python ${PROJECT_ROOT}/scripts/convert_instruction_json_to_training_format_siq2_loc.py \
        --input_json_file "${LOC_DIRPATH}/loc_${SPLIT}.json" \
        --output_json_file "${INSTRUCTION_FINETUNING_FPATH}" \
        --gt_ts_file "${GT_WITH_TS_FPATH}"


# Generate video features and predictions
export NPROC_PER_NODE=2
export OMP_NUM_THREADS=$(($(nproc) / ${NPROC_PER_NODE}))
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_loc.py \
    --model-name "${PROJECT_ROOT}/${MODEL_NAME}" \
    --video_dir "${VIDEO_DIR}" \
    --gt_file_qa "${INSTRUCTION_FINETUNING_FPATH}" \
    --output_dir "${OUTPUT_PER_VIDEO_DIRPATH}" \
    --output_name "${PRED_FPATH}"


# PYTHONPATH="./:$PYTHONPATH" python quantitative_evaluation/evaluate_loc.py \
#     --pred_path "${OUTPUT_DIRPATH}/${PRED_FPATH}.json" \
#     --output_dir "${OUTPUT_PER_VIDEO_DIRPATH}" \
#     --output_json "${OUTPUT_DIRPATH}/${RESULTS_FPATH}" \
#     --num_tasks 1
