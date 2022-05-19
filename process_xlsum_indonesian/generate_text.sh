cd ../tree_level_embeddings

SAVED_MODEL_FOLDER=../best_model

python evaluate_indoT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../process_xlsum_indonesian/preprocessed_summary --result_folder ../process_xlsum_indonesian/result
