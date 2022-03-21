import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd

import string
import collections
import json

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer

def get_text_cos_sim(teks_ringkasan_pred, teks_artikel):
    punc = ',.-""''?!\/'

    #do cosine similarity
    rata_rata = {}
        
    #untuk setiap kalimat di artikel
    for idx,sent in enumerate(teks_artikel):
        sim,rt = kemiripan_antara_2_kalimat(teks_artikel[idx].split(), teks_ringkasan_pred.split())
        rata_rata[idx] = rt #matriks similarity

    sorted_score_sent = dict(sorted(rata_rata.items(), key=lambda item: item[1], reverse=True))
    rank_sent = list(sorted_score_sent.keys())
    
#         print('sorted_score_sent',sorted_score_sent)
#         print('rank_sent',rank_sent)

    # print('kumpulan kata:\n', teks_ringkasan_pred)
    # print()
    # rank = 1
    # for i in sorted_score_sent:
    #     print('rank',rank,'==> kalimat ke-',i+1,'(score=',sorted_score_sent[i],') ==>',teks_artikel[i])
    #     rank+=1
    # print()
    
    ringkasan_prediksi = ''
    temp_kalimat_artikel = {}
    for i in range(0,2):
        try:
#                 print(sorted_score_sent[rank_sent[i]],rank_sent[i], teks_artikel[rank_sent[i]])
#                 print()
            kalimat_dari_artikel = ''
            if rank_sent[i] == 0:
                kalimat_dari_artikel = hapus_header_berita(teks_artikel[rank_sent[i]])
            else:
                kalimat_dari_artikel = teks_artikel[rank_sent[i]]
            temp_kalimat_artikel[rank_sent[i]] = kalimat_dari_artikel
#                 ringkasan_prediksi += kalimat_dari_artikel
        except:
            continue
        
    otka = collections.OrderedDict(sorted(temp_kalimat_artikel.items()))
    for i in otka:
        ringkasan_prediksi += otka[i]

    
    return ringkasan_prediksi

def kemiripan_antara_2_kalimat(kalimat1, kalimat2):
    kalimat_berita = kalimat1
    kumpulan_kata = kalimat2
    
    similarity = []
    rata_rata = 0
    rata_rata2 = 0
    total = 0
    
#     print('kumpulan_kata',kumpulan_kata,len(kumpulan_kata))
#     print('kalimat_berita',kalimat_berita,len(kalimat_berita))
    
    k_k = np.empty(shape=(len(kumpulan_kata),)) #vocab
    k_k.fill(1)
    
    k_b = np.empty(shape=(len(kumpulan_kata),))
    k_b.fill(-1)
        

#     untuk setiap kata di kalimat yang panjang
    for j, y in enumerate(kalimat_berita):
        baris = []
        kalimat_berita[j] = kalimat_berita[j].translate(str.maketrans('', '', string.punctuation))
        kalimat_berita[j] = kalimat_berita[j].lower()
#         untuk setiap kata di kalimat yang pendek
        for j2,y2 in enumerate(kumpulan_kata):
            if (kalimat_berita[j] == kumpulan_kata[j2]):
                if k_b[j2,] == -1:
                    k_b[j2,] = 1
                else:
                    k_b[j2,]+=1

    cossim = dot(k_b, k_k)/(norm(k_b)*norm(k_k))
    cos_sim = round(cossim,2)
    return (similarity, cossim)

def hapus_header_berita(kalimat_pertama):
#     print('before hapus header:')
#     print(kalimat_pertama,'\n')
    
    kalimat_pertama = kalimat_pertama.split()
    
    #hapus dash yang nempel di kata pertama
    kapital = kalimat_pertama[1].capitalize() == kalimat_pertama[1]
    all_kapital = kalimat_pertama[1].upper() == kalimat_pertama[1]
    if ('-' in kalimat_pertama[0] or '–' in kalimat_pertama[0]) and (kapital or all_kapital):
        kalimat_pertama.remove(kalimat_pertama[0])
    
    #hapus dash yang tidak nempel di kata pertama
    if '-' in kalimat_pertama:
        awal_dash = kalimat_pertama.index('-')
        header = kalimat_pertama[:awal_dash+1]
        
        kapital = kalimat_pertama[awal_dash+1].capitalize() == kalimat_pertama[awal_dash+1]
        all_kapital = kalimat_pertama[awal_dash+1].upper() == kalimat_pertama[awal_dash+1]
        if awal_dash<5 and (kapital or all_kapital):
            kalimat_pertama = kalimat_pertama[awal_dash+1:]
            
    if '–' in kalimat_pertama:
        awal_dash = kalimat_pertama.index('–')
        header = kalimat_pertama[:awal_dash+1]
        
        kapital = kalimat_pertama[awal_dash+1].capitalize() == kalimat_pertama[awal_dash+1]
        all_kapital = kalimat_pertama[awal_dash+1].upper() == kalimat_pertama[awal_dash+1]
        if awal_dash<5 and (kapital or all_kapital):
            kalimat_pertama = kalimat_pertama[awal_dash+1:]
    
    #hapus dash kedua (sekarang jadi di awal karena dash pertama sudah dihapus di atas)
    kapital = kalimat_pertama[1].capitalize() == kalimat_pertama[1]
    all_kapital = kalimat_pertama[1].upper() == kalimat_pertama[1]
    if ('-' in kalimat_pertama[0] or '–' in kalimat_pertama[0]) and (kapital or all_kapital):
        kalimat_pertama.remove(kalimat_pertama[0])
        
    kalimat_pertama = ' '.join(kalimat_pertama)
    kalimat_pertama += ' '
        
    return kalimat_pertama

if __name__ == "__main__":
    # print(get_text_cos_sim('test 1', ['test 1', 'test 2', 'test 3', 'test 4']))

    # get raw articles from test data
    json_file_test_path = '../../../peringkasan_abstraktif_amr/xlsum_indonesian/indonesian_test.jsonl'
    data = []
    with open(json_file_test_path,  encoding='utf-8') as f:
        while True:
            line = f.readline()
            if (not line):
                break
            data.append(json.loads(line))
    print(sent_tokenize(data[0]['text']))
    # print(len(data))

    # get simple nlg result
    df = pd.read_csv('../result_simple_nlg.tsv', sep='\t')
    list_simple_nlg = df['simple_nlg'].values

    result = []
    for i in tqdm(range(len(data))):
        result.append(get_text_cos_sim(list_simple_nlg[i], sent_tokenize(data[i]['text'])))

    # calculate score
    list_label = pd.read_csv('result_with_score.csv')['label'].values

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'])
    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    list_score_rouge1 = []
    list_score_rouge2 = []
    list_score_rougeL = []
    for i in range(len(list_label)):
        scores = scorer.score(result[i].strip().lower(), list_label[i].strip().lower())
    #     print(scores)
        list_score_rouge1.append(scores['rouge1'].fmeasure)
        list_score_rouge2.append(scores['rouge2'].fmeasure)
        list_score_rougeL.append(scores['rougeL'].fmeasure)
    print(sum(list_score_rouge1) / len(list_score_rouge1))
    print(sum(list_score_rouge2) / len(list_score_rouge2))
    print(sum(list_score_rougeL) / len(list_score_rougeL))


    df_to_save = pd.DataFrame({'simple_nlg':list_simple_nlg, 'postprocess': result,
                                    'label':list_label, 'rouge1': list_score_rouge1, 'rouge2': list_score_rouge2, 'rougeL': list_score_rougeL})
    df_to_save.to_csv('baseline_postprocess.csv', index=False)

