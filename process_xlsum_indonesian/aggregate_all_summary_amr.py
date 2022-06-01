import os
from tqdm import tqdm
import json

SUMMARY_AMR_FOLDER="D:/Taufiq kuliah/tugas/TUGAS AKHIR/code and data/peringkasan_berbasis_amr/semantic_summ/subgraf_ringkasan_xlsum_indo/summ_ramp_5_passes_len_edges_exp_0_xlsum_indo"
XLSUM_INDONESIAN_FOLDER_PATH = "../../peringkasan_berbasis_amr/xlsum_indonesian/"

if __name__ == "__main__":
    json_file_test_path = os.path.join(XLSUM_INDONESIAN_FOLDER_PATH, 'indonesian_test.jsonl')
    raw_data = []
    with open(json_file_test_path,  encoding='utf-8') as f:
        while True:
            line = f.readline()
            if (not line):
                break
            raw_data.append(json.loads(line))
    print(raw_data[0])
    print(len(raw_data))

    result_file_path = "generated_summary_amr.txt"
    outfile = open(result_file_path, 'w+', encoding='utf8')

    list_result_simple_nlg = []  # list (id, simple_nlg, gold summary)

    for item in tqdm(raw_data):
        id = item['id']
        sent = item['summary']

        summary_amr_file_path =  os.path.join(SUMMARY_AMR_FOLDER, f'{id}_system')
        with open(summary_amr_file_path, encoding='utf8') as f:
            outfile.write(f'# ::id {id}\n')
            outfile.write(f'# ::snt {sent}\n')
            for line in f:
                outfile.write(line)
                if ('# ::simple_nlg ' in line):
                    list_result_simple_nlg.append((id, line[len('# ::simple_nlg ' ):], sent))
            outfile.write('\n\n')
    outfile.close()

    list_simple_nlg = []
    list_label = []
    ## output simple nlg
    with open('result_simple_nlg.tsv', 'w', encoding='utf8') as f:
        f.write('id\tsimple_nlg\tgold_summary\n')
        for (id, simple_nlg, gold_summary) in list_result_simple_nlg:
            list_simple_nlg.append(simple_nlg)
            list_label.append(gold_summary)
            f.write(id)
            f.write('\t')
            f.write(simple_nlg.strip())
            f.write('\t')
            f.write(gold_summary.strip())
            f.write('\n')


    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'])

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    list_score_rouge1 = []
    list_score_rouge2 = []
    list_score_rougeL = []
    for i in range(len(list_label)):
        scores = scorer.score(list_simple_nlg[i].strip().lower(), list_label[i].strip().lower())
        #     print(scores)
        list_score_rouge1.append(scores['rouge1'].fmeasure)
        list_score_rouge2.append(scores['rouge2'].fmeasure)
        list_score_rougeL.append(scores['rougeL'].fmeasure)
    print(sum(list_score_rouge1) / len(list_score_rouge1))
    print(sum(list_score_rouge2) / len(list_score_rouge2))
    print(sum(list_score_rougeL) / len(list_score_rougeL))