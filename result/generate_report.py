import sys
import os

sys.path.append('..')
from utils.scoring import calc_corpus_bleu_score

def generate_report_from_result_specific_data(path_folder, dataset_name):
    complete_path = os.path.join(path_folder, dataset_name)
    with open(os.path.join(complete_path, 'test_generations.txt')) as f:
        test_generations = f.readlines()
    with open(os.path.join(complete_path, 'test_label.txt')) as f:
        test_label = f.readlines()
    
    list_tuple_bleu_hyp_label_source = []  # (bleu, hyp, label,source)
    for i in range(len(test_generations)):
        hyp = test_generations[i].strip()
        label = test_label[i].strip()
        score = calc_corpus_bleu_score([hyp], [label])
        list_tuple_bleu_hyp_label_source.append((score, hyp, label, dataset_name))      

    return list_tuple_bleu_hyp_label_source

def get_hyps_and_labels_from_folder(folder_result):
    with open(os.path.join(folder_result, 'test_generations.txt')) as f:
        test_generations = f.readlines()
    with open(os.path.join(folder_result, 'test_label.txt')) as f:
        test_label = f.readlines()    

    list_hyp = []
    list_label = []
    for i in range(len(test_generations)):
        hyp = test_generations[i].strip()
        label = test_label[i].strip()[:-1]
        list_hyp.append(hyp)
        list_label.append(label)

    return list_hyp, list_label                

def aggregate_bleu_score(list_path_folder_result):
    list_all_hyp = []
    list_all_label = []
    for path_folder_result in list_path_folder_result:
        list_hyp, list_label = get_hyps_and_labels_from_folder(path_folder_result)
        list_all_hyp += list_hyp
        list_all_label += list_label

    bleu = calc_corpus_bleu_score(list_all_hyp, list_all_label)
    return bleu

LIST_NEWS_DATASET = ['b-salah-darat', 'c-gedung-roboh', 'd-indo-fuji', 'f-bunuh-diri', 'g-gempa-dieng']

if __name__ == '__main__':
    path_folder =  sys.argv[1]

    # aggregate bleu score of all news dataset
    list_path_folder_news_dataset = [os.path.join(path_folder, data_name) for data_name in LIST_NEWS_DATASET]
    news_bleu_score = aggregate_bleu_score(list_path_folder_news_dataset)
    print('news bleu score: ', str(news_bleu_score))
    with open(os.path.join(path_folder,'aggregate_bleu_news_dataset.txt'), 'w') as f:
        f.write(str(news_bleu_score))
    
    # generate list_tuple_bleu_hyp_label_source from amr_simple_test
    list_tuple_bleu_hyp_label_source = generate_report_from_result_specific_data(path_folder, 'amr_simple_test')
    list_tuple_bleu_hyp_label_source.sort(reverse=True)
    with open(os.path.join(path_folder, 'report_amr_simple_test.csv'), 'w') as f:
        f.write('bleu,hyp,label,source\n')
        for (bleu, hyp, label,source) in list_tuple_bleu_hyp_label_source:
            f.write(f'{str(bleu)},"{str(hyp)}","{str(label)}",{str(source)}\n')

    # generate list_tuple_bleu_hyp_label_source from all news dataset
    list_tuple_bleu_hyp_label_source = []
    for dataset_name in LIST_NEWS_DATASET:
        list_tuple_bleu_hyp_label_source += generate_report_from_result_specific_data(path_folder, dataset_name)

    list_tuple_bleu_hyp_label_source.sort(reverse=True)
    with open(os.path.join(path_folder, 'report_news_dataset_test.csv'), 'w') as f:
        f.write('bleu,hyp,label,source\n')
        for (bleu, hyp, label,source) in list_tuple_bleu_hyp_label_source:
            f.write(f'{str(bleu)},"{str(hyp)}","{str(label)}",{str(source)}\n')


