# linearized_penman
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/linearized_penman/train.amr.txt --result_sent_path data/preprocessed_data/linearized_penman/train.sent.txt --mode linearized_penman
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/linearized_penman/dev.amr.txt --result_sent_path data/preprocessed_data/linearized_penman/dev.sent.txt --mode linearized_penman
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/linearized_penman/test.amr.txt --result_sent_path data/preprocessed_data/linearized_penman/test.sent.txt --mode linearized_penman

# dfs
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/dfs/train.amr.txt --result_sent_path data/preprocessed_data/dfs/train.sent.txt --mode dfs
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/dfs/dev.amr.txt --result_sent_path data/preprocessed_data/dfs/dev.sent.txt --mode dfs
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/dfs/test.amr.txt --result_sent_path data/preprocessed_data/dfs/test.sent.txt --mode dfs

# nodes_only
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/nodes_only/train.amr.txt --result_sent_path data/preprocessed_data/nodes_only/train.sent.txt --mode nodes_only
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/nodes_only/dev.amr.txt --result_sent_path data/preprocessed_data/nodes_only/dev.sent.txt --mode nodes_only
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/nodes_only/test.amr.txt --result_sent_path data/preprocessed_data/nodes_only/test.sent.txt --mode nodes_only

# rule_based_traversal
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/rule_based_traversal/train.amr.txt --result_sent_path data/preprocessed_data/rule_based_traversal/train.sent.txt --mode rule_based_traversal
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/rule_based_traversal/dev.amr.txt --result_sent_path data/preprocessed_data/rule_based_traversal/dev.sent.txt --mode rule_based_traversal
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/rule_based_traversal/test.amr.txt --result_sent_path data/preprocessed_data/rule_based_traversal/test.sent.txt --mode rule_based_traversal

# dfs_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/dfs_with_tree_level/train.amr.txt --result_sent_path data/preprocessed_data/dfs_with_tree_level/train.sent.txt --mode dfs_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/dfs_with_tree_level/dev.amr.txt --result_sent_path data/preprocessed_data/dfs_with_tree_level/dev.sent.txt --mode dfs_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/dfs_with_tree_level/test.amr.txt --result_sent_path data/preprocessed_data/dfs_with_tree_level/test.sent.txt --mode dfs_with_tree_level

# linearized_penman_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_akhyar/amr_simple_amany_v2.txt --result_amr_path data/preprocessed_data/linearized_penman_with_tree_level/train.amr.txt --result_sent_path data/preprocessed_data/linearized_penman_with_tree_level/train.sent.txt --mode linearized_penman_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_dev.txt --result_amr_path data/preprocessed_data/linearized_penman_with_tree_level/dev.amr.txt --result_sent_path data/preprocessed_data/linearized_penman_with_tree_level/dev.sent.txt --mode linearized_penman_with_tree_level
python preprocess/preprocess.py --source_file_path data/raw_data_ilmy/amr_simple_test.txt --result_amr_path data/preprocessed_data/linearized_penman_with_tree_level/test.amr.txt --result_sent_path data/preprocessed_data/linearized_penman_with_tree_level/test.sent.txt --mode linearized_penman_with_tree_level
