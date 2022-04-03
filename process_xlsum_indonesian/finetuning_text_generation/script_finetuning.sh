mkdir result

python train_indoT5.py \
--model_type indo-t5 \
--n_epochs 2 --lr 0.0001 \
--data_folder data \
--result_folder result \
--resume_from_checkpoint True \
--saved_model_folder_path ../../best_model