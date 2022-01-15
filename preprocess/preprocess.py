from os import listdir, path

class PreprocessAMR:   
    def preprocess(self, split, file_path, result_folder_path):
        list_amr, list_sent = self.linearize_penman(file_path)
        print(split)
        print(len(list_amr))
        print(len(list_sent))
        
        f = open(path.join(result_folder_path, f'{split}.amr.txt'), "w")
        for amr in list_amr:
            f.write(amr.strip())
            f.write("\n")
        f.close()

        f = open(path.join(result_folder_path, f'{split}.sent.txt'), "w")
        for sent in list_sent:
            f.write(sent.strip())
            f.write("\n")
        f.close()

    def linearize_penman(self, file_path):
        with open(file_path) as f:
            data = f.readlines()

        # transform original AMR to linearized penman notation based on format Ribeiro dkk (2020)
        list_amr = []
        list_sent = []

        sent_now = ""
        amr_now = ""
        is_reading_amr = False

        for line in data:
            line = line.strip()

            # print(line)
            if (line==''):
                if (is_reading_amr):
                    is_reading_amr = False      # reading AMR finish then append it to list
                    amr_now = " ".join(amr_now.split())
                    list_amr.append(amr_now)
                    amr_now = ""
                continue

            line += "."

            if (line[0]=='#'):  # read label (sentence)
                if ('# ::snt' in line):
                    sent_now = line[8:-1]
                    temp_sent_now = sent_now.split()
                    sent_now = " ".join(temp_sent_now)
                    list_sent.append(sent_now)
                else: # ignore other than sentence/amr
                    continue
            else:  #reading amr
                is_reading_amr = True
                found_slash = False # /
                found_colon = False # :
                for c in line:
                    if (c=='.'):
                        amr_now += " "
                        found_colon = False
                        found_slash = False
                    elif (c=='('):
                        amr_now += " " + c + " "
                    elif (c==')' and not found_colon and not found_slash):
                        amr_now += " " + c + " "
                    elif (c=='/'):
                        found_slash = True
                    elif (c==':'):
                        found_colon = True
                        amr_now += c
                    else:
                        if (found_colon):
                            amr_now += c
                            if (c==' '):
                                found_colon = False
                        elif (found_slash):
                            if (c==')'):   
                                amr_now += " "
                                found_slash = False
                            amr_now += c

        amr_now = " ".join(amr_now.split())
        if (amr_now != ""):
            list_amr.append(amr_now)
        
        return list_amr, list_sent
    

if __name__=="__main__":
    PREPROCESS_AMR = PreprocessAMR()
    RESULT_FOLDER_PATH = 'data/preprocessed_data'

    TRAIN_FILE_PATH = 'data/raw_data_ilmy/amr_simple_train.txt'
    DEV_FILE_PATH = 'data/raw_data_ilmy/amr_simple_dev.txt'
    TEST_FILE_PATH = 'data/raw_data_ilmy/amr_simple_test.txt'

    PREPROCESS_AMR.preprocess(split='train',file_path=TRAIN_FILE_PATH,result_folder_path=RESULT_FOLDER_PATH)
    PREPROCESS_AMR.preprocess(split='dev',file_path=DEV_FILE_PATH,result_folder_path=RESULT_FOLDER_PATH)
    PREPROCESS_AMR.preprocess(split='test',file_path=TEST_FILE_PATH,result_folder_path=RESULT_FOLDER_PATH)




