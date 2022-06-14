PREPROCESSING_METHOD=$1
SILVER_DATA_FOLDER=$2
EPOCH=$3

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
cd train
mkdir result/result_supervised_task_adaptation

python train_mT5.py \
--model_type mT5 \
--n_epochs ${EPOCH} \
--lr 1e-4 \
--data_folder ../data/preprocessed_silver_data  \
--result_folder result/result_supervised_task_adaptation

### if you want to continue finetuning directyly
### python train_mT5.py --model_type mT5 --n_epochs 3 --data_folder ../data/preprocessed_data/linearized_penman  --result_folder result/result_linearized_penman \
### --resume_from_checkpoint True --saved_model_folder_path result/result_supervised_task_adaptation