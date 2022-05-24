mkdir result
cd ../../train

python train_indoT5.py \
--model_type indo-t5 \
--n_epochs 2 --lr 0.0001 \
--data_folder ../process_xlsum_indonesian/finetuning_text_generation/data_train \
--result_folder ../process_xlsum_indonesian/finetuning_text_generation/result \
--resume_from_checkpoint True \
--saved_model_folder_path ../best_model
