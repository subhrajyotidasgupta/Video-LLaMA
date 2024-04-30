#!/bin/bash

DIR="/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/results"
# MODEL_DIR='/home/mila/s/subhrajyoti.dasgupta/scratch/hf_models/Video-LLaMA-2-7B-Pretrained/VL_LLaMA_2_7B_Pretrained.pth'

ANNO_DIR='/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/gt/iavqd/music_avqa/'
# VIDEO_DIR='/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/avqa/videos'
VIDEO_DIR='/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/ivqd/music_avqa/videos'
# VIDEO_DIR='/home/mila/s/subhrajyoti.dasgupta/scratch/macaw/data/music_avqa/MUSIC-AVQA-videos-Real'
TASK='IAVQD'
DATASET='music_avqa'
SPLIT='test'
GT_FILE="${ANNO_DIR}/instruct_${SPLIT}_${TASK}.json"
# GT_FILE='/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/gt/iaqd/avqa/val_output.json'

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}

NUM_SAMPLES=20

PRED_FILE=${OUTPUT_DIR}/${TASK}_${DATASET}_${SPLIT}_f${NUM_FRAME}_result_${NUM_SAMPLES}.json

# PRED_FILE=/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/results/IAVQD/IAVQD_test_f96_result_1000_nota.json
# GT_FILE=/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/gt/iavqd/audioset/instruct_test_IAVQD_44K.json

python evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} --gt_file ${GT_FILE} \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --num_samples ${NUM_SAMPLES} \
--batch_size 1 \
# # # --video_llama_model_path ${MODEL_DIR} \

python metrics.py --gt_file ${GT_FILE} \
--pred_file ${PRED_FILE} \
--dataset_name ${DATASET} \
--task_name ${TASK} \
--output_file /home/mila/s/subhrajyoti.dasgupta/scratch/videollama/scores/iavqd/metrics_${DATASET}_${TASK}_result.json