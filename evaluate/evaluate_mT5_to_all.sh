#!/bin/sh

SAVED_MODEL_FOLDER=$1
PREPROCESSING_METHOD=$2

echo saved model folder ${SAVED_MODEL_FOLDER}
echo preprocessing method ${PREPROCESSING_METHOD}

####### preprocess all data test #########
cd ..
mkdir data/test/preprocessed_data

# amr_simple_test
echo preprocess amr_simple_test
mkdir data/test/preprocessed_data/amr_simple_test

python preprocess/preprocess.py \
--source_file_path data/test/amr_simple_test.txt \
--result_amr_path data/test/preprocessed_data/amr_simple_test/test.amr.txt \
--result_sent_path data/test/preprocessed_data/amr_simple_test/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

# b-salah-darat
echo preprocess b-salah-darat
mkdir data/test/preprocessed_data/b-salah-darat

python preprocess/preprocess.py \
--source_file_path data/test/b-salah-darat.txt \
--result_amr_path data/test/preprocessed_data/b-salah-darat/test.amr.txt \
--result_sent_path data/test/preprocessed_data/b-salah-darat/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

# c-gedung-roboh
echo preprocess c-gedung-roboh
mkdir data/test/preprocessed_data/c-gedung-roboh

python preprocess/preprocess.py \
--source_file_path data/test/c-gedung-roboh.txt \
--result_amr_path data/test/preprocessed_data/c-gedung-roboh/test.amr.txt \
--result_sent_path data/test/preprocessed_data/c-gedung-roboh/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

# d-indo-fuji
echo preprocess d-indo-fuji
mkdir data/test/preprocessed_data/d-indo-fuji

python preprocess/preprocess.py \
--source_file_path data/test/d-indo-fuji.txt \
--result_amr_path data/test/preprocessed_data/d-indo-fuji/test.amr.txt \
--result_sent_path data/test/preprocessed_data/d-indo-fuji/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

# f-bunuh-diri
echo preprocess f-bunuh-diri
mkdir data/test/preprocessed_data/f-bunuh-diri

python preprocess/preprocess.py \
--source_file_path data/test/f-bunuh-diri.txt \
--result_amr_path data/test/preprocessed_data/f-bunuh-diri/test.amr.txt \
--result_sent_path data/test/preprocessed_data/f-bunuh-diri/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

# g-gempa-dieng
echo preprocess g-gempa-dieng
mkdir data/test/preprocessed_data/g-gempa-dieng

python preprocess/preprocess.py \
--source_file_path data/test/g-gempa-dieng.txt \
--result_amr_path data/test/preprocessed_data/g-gempa-dieng/test.amr.txt \
--result_sent_path data/test/preprocessed_data/g-gempa-dieng/test.sent.txt \
--mode ${PREPROCESSING_METHOD}

####### predict all and evaluate #########
cd evaluate
mkdir result

# amr_simple_test
echo evaluate on data amr_simple_test
mkdir result/amr_simple_test

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/amr_simple_test --result_folder result/amr_simple_test

# b-salah-darat
echo evaluate on data b-salah-darat
mkdir result/b-salah-darat

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/b-salah-darat --result_folder result/b-salah-darat

# c-gedung-roboh
echo evaluate on data c-gedung-roboh
mkdir result/c-gedung-roboh

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/c-gedung-roboh --result_folder result/c-gedung-roboh

# d-indo-fuji
echo evaluate on data d-indo-fuji
mkdir result/d-indo-fuji

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/d-indo-fuji --result_folder result/d-indo-fuji

# f-bunuh-diri
echo evaluate on data f-bunuh-diri
mkdir result/f-bunuh-diri

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/f-bunuh-diri --result_folder result/f-bunuh-diri

# g-gempa-dieng
echo evaluate on data g-gempa-dieng
mkdir result/g-gempa-dieng

python evaluate_mT5.py --saved_model_folder_path ${SAVED_MODEL_FOLDER} \
--data_folder ../data/test/preprocessed_data/g-gempa-dieng --result_folder result/g-gempa-dieng

##### move result ######
mkdir result/${PREPROCESSING_METHOD}
mv result/* result/${PREPROCESSING_METHOD}