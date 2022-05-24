# amr-to-text-indonesia
Author: Taufiq Husada Daryanto

## Description
Pembangkitan teks Bahasa Indonesia dari graf AMR dengan metode finetuning model bahasa pralatih (IndoT5/IndoBART) dengan tambahan supervised task adaptation dan tree-level embeddings

## How to run trained model
https://colab.research.google.com/drive/1F5ubJ03fwvW4fXcOrsxVxJXihrZ30-uQ?usp=sharing
<!-- with tree level embeddings: https://colab.research.google.com/drive/1q4CNhhlR8O-Mm465udoi8AdrbOamE1-g?usp=sharing -->

## How to train (reproduce the result)
see folder `examples`

## Result text generation
see folder `result`

## trained model (open for public)
1. Finetuning indoT5 + supervised task adaptation: https://drive.google.com/file/d/1hrE9R4I5lBpNyEngY7Tdz84L3sRp_015/view?usp=sharing
2. Finetuning indoT5 + supervised task adaptation + additional finetuning for xlsum indo: https://drive.google.com/file/d/1Tacnafczh24Fqet9Upmn4_amPSBO1hKs/view?usp=sharing

## Dataset used
https://github.com/taufiqhusada/amr-indo-dataset

# Peringkasan berbasis AMR dengan penambahan pembangkitan teks dari AMR
Using XLsum-Indonesian dataset
## Result
1. Summary AMR output: `process_xlsum_indonesian/generated_summary_amr.txt`
2. Output pembangkitan teks dengan trained model nomor 3: `process_xlsum_indonesian/generated_summary_amr.txt/finetuning_text_generation/result_after_additional_finetuning.csv`
3. Output pembangkitan teks dengan trained model nomor 2: `process_xlsum_indonesian/result_with_score.csv`

## Code
1. peringkasan: https://drive.google.com/drive/folders/1E3ufHOQ8r-DjfjhzfWUxl4lpvvc21cgF?usp=sharing (open it using std.stei email)
2. pembangkitan teks: this repo (see how to run)
