#!/bin/sh

SAVED_MODEL_FOLDER=../best_model
PREPROCESSING_METHOD=linearized_penman_with_tree_level
SUMMARY_AMR_FILE=process_xlsum_indonesian/generated_summary_amr.txt

echo saved model folder ${SAVED_MODEL_FOLDER}
echo preprocessing method ${PREPROCESSING_METHOD}
echo summary amr folder ${SUMMARY_AMR_FILE}

####### preprocess summary amr ###########
cd ..
mkdir process_xlsum_indonesian/preprocessed_summary

python preprocess/preprocess.py \
--source_file_path ${SUMMARY_AMR_FILE} \
--result_amr_path process_xlsum_indonesian/preprocessed_summary/summary.amr.txt \
--result_sent_path process_xlsum_indonesian/preprocessed_summary/summary.gold.txt \
--mode ${PREPROCESSING_METHOD}