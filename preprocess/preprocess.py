from os import listdir, path
import argparse
from tqdm import tqdm
from graph_utils import convert_linearized_penman_to_rule_based_traversal, convert_linearized_penman_to_traversal_with_tree_level
import re

class PreprocessAMR:   
    def preprocess(self, source_file_path, result_amr_path, result_sent_path, source_folder_path=None, mode="linearized_penman"):
        if (source_folder_path==None and source_file_path==None):
            raise Exception("please specify source file path or source folder path")
        if (source_folder_path and source_file_path):
            raise Exception("please specify either only source file path or source folder path")
        
        if ("with_tree_level" not in mode):
            list_pair_sent_amr = []
            if (source_file_path):
                if (mode=="linearized_penman"):
                    list_pair_sent_amr = self.linearize_penman(source_file_path)
                elif (mode=="dfs"):
                    list_pair_sent_amr = self.dfs_nodes_and_edges_only(source_file_path)
                elif (mode=="nodes_only"):
                    list_pair_sent_amr = self.nodes_only(source_file_path)
                elif (mode=='rule_based_traversal'):
                    list_pair_sent_amr = self.rule_based_traversal(source_file_path)
                else:
                    raise Exception("specified linearization mode not valid")
            else:
                list_file = [f for f in listdir(source_folder_path)]
                for file_name in list_file:
                    if (mode=="linearized_penman"):
                        list_pair_sent_amr_from_file = self.linearize_penman(path.join(source_folder_path, file_name))
                    elif (mode=="dfs"):
                        list_pair_sent_amr_from_file = self.dfs_nodes_and_edges_only(path.join(source_folder_path, file_name))
                    elif (mode=="nodes_only"):
                        list_pair_sent_amr_from_file = self.nodes_only(path.join(source_folder_path, file_name))
                    elif (mode=='rule_based_traversal'):
                        list_pair_sent_amr_from_file = self.rule_based_traversal(path.join(source_folder_path, file_name))
                    else:
                        raise Exception("specified linearization mode not valid")
                    list_pair_sent_amr += list_pair_sent_amr_from_file
                    
                
            print("total:", len(list_pair_sent_amr), "pair sent-amr")
            
            f = open(result_amr_path, "w")
            for (sent,amr) in list_pair_sent_amr:
                f.write(amr.strip())
                f.write("\n")
            f.close()

            f = open(result_sent_path, "w")
            for (sent,amr) in list_pair_sent_amr:
                f.write(sent.strip())
                f.write("\n")
            f.close()

        else: # mode == <dfs/linearized_penman>_with_tree_level
            list_tuple_sent_amr_level = []
            if (source_file_path):
                list_tuple_sent_amr_level = self.traversal_with_tree_level(source_file_path, mode[:-16])
            else:
                list_file = [f for f in listdir(source_folder_path)]
                for file_name in list_file:
                    list_tuple_sent_amr_level_from_file = self.traversal_with_tree_level(path.join(source_folder_path, file_name), mode[:-16])
                list_tuple_sent_amr_level += list_tuple_sent_amr_level_from_file
            
            print("total:", len(list_tuple_sent_amr_level), " tuple_sent_amr_level")
            
            f = open(result_amr_path, "w")
            for (sent,amr,str_list_level) in list_tuple_sent_amr_level:
                f.write(amr.strip())
                f.write("\n")
            f.close()

            f = open(result_sent_path, "w")
            for (sent,amr,str_list_level) in list_tuple_sent_amr_level:
                f.write(sent.strip())
                f.write("\n")
            f.close()

            f = open(result_amr_path+'.tree_level', "w")
            for (sent,amr,str_list_level) in list_tuple_sent_amr_level:
                f.write(str_list_level.strip())
                f.write("\n")
            f.close()

    def _tidy_up_linearized_penman(self, linearized_penman):
        list_item = linearized_penman.split()

        # map variable to node
        map_variable_node = {}
        for i in range(len(list_item)):
            if (list_item[i] == '/'):
                map_variable_node[list_item[i-1]] = list_item[i+1]

        # remove variable and slash
        pattern_to_be_removed = '\( \w+ /'
        subtitute = '('
        final_str = re.sub(pattern_to_be_removed, subtitute, linearized_penman)

        pattern_to_be_removed = '/'
        subtitute = ''
        final_str = re.sub(pattern_to_be_removed, subtitute, final_str)

        # convert remaining variable (usually for coreference cases) to node
        list_item = final_str.split()
        for i in range(len(list_item)):
            if (list_item[i] in map_variable_node):
                list_item[i] = map_variable_node[list_item[i]]

        final_str = " ".join(list_item)

        return final_str

    def linearize_penman(self, file_path):
        with open(file_path) as f:
            data = f.readlines()

        # transform original AMR to linearized penman notation based on format Ribeiro dkk (2020)
        list_pair_sent_amr = []   # list[(<sent>, <amr>), ...]

        sent_now = ""
        amr_now = ""
        is_reading_amr = False

        for idx_line in tqdm(range(len(data))):
            line = data[idx_line]
            line = line.strip()

            # print(line)
            if (line==''):
                if (is_reading_amr):
                    is_reading_amr = False      # reading AMR finish then append it to list
                    amr_now = " ".join(amr_now.split())
                    amr_now = self._tidy_up_linearized_penman(amr_now)
                    list_pair_sent_amr.append((sent_now, amr_now))
                    amr_now = ""
                    sent_now = ""
                continue

            line += "."

            if (line[0]=='#'):  # read label (sentence)
                if ('# ::snt ' in line):
                    sent_now = line[8:-1]
                    temp_sent_now = sent_now.split()
                    sent_now = " ".join(temp_sent_now)
                else: # ignore other than sentence/amr
                    continue
            else:  #reading amr
                is_reading_amr = True
                found_slash = False # /
                found_colon = False # :
                for c in line:
                    if (c=='.'):
                        amr_now += " "
                    elif (c==')' or c=='(' or c=='/'):
                        amr_now += " " + c + " "
                    else:
                        amr_now += c

        amr_now = " ".join(amr_now.split())
        if (amr_now != ""):
            amr_now = self._tidy_up_linearized_penman(amr_now)
            list_pair_sent_amr.append((sent_now, amr_now))
        
        return list_pair_sent_amr
    
    def dfs_nodes_and_edges_only(self, file_path):
        list_pair_sent_amr = self.linearize_penman(file_path)
    
        for i in range(len(list_pair_sent_amr)):
            (sent,amr) = list_pair_sent_amr[i]            
            curr_amr = ""
            for c in amr:
                if (c=='(' or c==')'):
                    continue

                curr_amr += c
            list_pair_sent_amr[i] = (sent, " ".join(curr_amr.split()))

        return list_pair_sent_amr

    def nodes_only(self, file_path):
        list_pair_sent_amr = self.linearize_penman(file_path)
    
        for i in range(len(list_pair_sent_amr)):
            (sent,amr) = list_pair_sent_amr[i]
            curr_amr = ""
            for kata in amr.split():
                if (kata=='(' or kata==')' or (':' in kata)):
                    continue

                curr_amr += kata + " "
            list_pair_sent_amr[i] = (sent, " ".join(curr_amr.split()))

        return list_pair_sent_amr

    def rule_based_traversal(self, file_path):
        list_pair_sent_amr = self.linearize_penman(file_path)

        for i in range(len(list_pair_sent_amr)):
            (sent,amr) = list_pair_sent_amr[i]
            result_path_traversal = convert_linearized_penman_to_rule_based_traversal(amr)
            list_pair_sent_amr[i] = (sent, result_path_traversal)

        return list_pair_sent_amr

    def traversal_with_tree_level(self, file_path, mode):
        list_pair_sent_amr = self.linearize_penman(file_path)
        list_tuple_sent_amr_level = []

        for i in range(len(list_pair_sent_amr)):
            (sent,amr) = list_pair_sent_amr[i]
            result_path_traversal, result_list_level = convert_linearized_penman_to_traversal_with_tree_level(amr, mode)

            list_tuple_sent_amr_level.append((sent, result_path_traversal, result_list_level))
        return list_tuple_sent_amr_level

if __name__=="__main__":
    PREPROCESS_AMR = PreprocessAMR()

    ## Example in code
    # TRAIN_FILE_PATH = '../data/raw_data_ilmy/amr_simple_train.txt'
    # PREPROCESS_AMR.preprocess(source_file_path=None,result_amr_path="amr_train_nodes_only.txt", result_sent_path="sent_train.txt", source_folder_path='../data/raw_data_ilmy', mode="nodes_only")
    # PREPROCESS_AMR.preprocess(source_file_path=TRAIN_FILE_PATH,result_amr_path="amr_train_nodes_only.txt", result_sent_path="sent_train.txt", source_folder_path=None, mode="nodes_only")

    ## Example running from terminal
    # python preprocess.py --source_file_path ../data/raw_data_ilmy/amr_simple_train.txt --result_amr_path amr_train_nodes_only.txt --result_sent_path sent_train.txt --mode nodes_only

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="linearized_penman", help="linearized_penman/ dfs/ nodes_only")
    parser.add_argument("--source_file_path", default=None, help="pair sentence amr raw data file")
    parser.add_argument("--source_folder_path", default=None, help="pair sentence amr raw data file")
    parser.add_argument("--result_sent_path", help="result file for sentences")
    parser.add_argument("--result_amr_path", help="result file for preprocessed amr")
    args = parser.parse_args()
    print(args)

    PREPROCESS_AMR.preprocess(
        source_file_path=args.source_file_path,
        result_amr_path=args.result_amr_path, 
        result_sent_path=args.result_sent_path, 
        source_folder_path=args.source_folder_path, 
        mode=args.mode
    )

    




