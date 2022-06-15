PREPROCESSING_METHOD=$1
SILVER_DATA_FOLDER=$2

echo preprocessing method ${PREPROCESSING_METHOD}
echo silver data folder ${SILVER_DATA_FOLDER}

####### preprocess silver data #########
cd ..
mkdir data/preprocessed_silver_data

python preprocess/preprocess.py \
--source_folder_path ${SILVER_DATA_FOLDER} \
--result_amr_path data/preprocessed_silver_data/train.amr.txt \
--result_sent_path data/preprocessed_silver_data/train.sent.txt \
--mode ${PREPROCESSING_METHOD}

## use dev and test data from amr_simple
cp data/preprocessed_data/${PREPROCESSING_METHOD}/dev* data/preprocessed_silver_data
cp data/preprocessed_data/${PREPROCESSING_METHOD}/test* data/preprocessed_silver_data


######### train on silver data ###########
cd tree_level_embeddings
mkdir result/result_supervised_task_adaptation

python train_mT5.py \
--model_type mT5 \
--n_epochs 1 \
--lr 0.0001 \
--max_seq_len_amr 256 \
--max_seq_len_sent 192 \
--data_folder ../data/preprocessed_silver_data  \
--result_folder result/result_supervised_task_adaptation
