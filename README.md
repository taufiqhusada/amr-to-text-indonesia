# amr-to-text-indonesia
Author: Taufiq Husada Daryanto

## Description
AMR-to-text generation for Indonesian language using fine-tuning pretrained language model (IndoT5/IndoBART/mT5) with additional supervised task adaptation and tree-level embedding
- best model: finetuning indot5 (using linearized PENMAN input) + supervised task adaptation 

## How to run trained model
[https://colab.research.google.com/drive/1F5ubJ03fwvW4fXcOrsxVxJXihrZ30-uQ?usp=sharing](https://colab.research.google.com/drive/1q4CNhhlR8O-Mm465udoi8AdrbOamE1-g?usp=sharing)

## How to train (reproduce the result)
see folder `examples`

## Result text generation
see folder `result`

## trained model (open for public)
1. Finetuning indoT5 + supervised task adaptation: https://drive.google.com/file/d/1hrE9R4I5lBpNyEngY7Tdz84L3sRp_015/view?usp=sharing
2. Finetuning indoT5 + supervised task adaptation + additional finetuning for xlsum indo: https://drive.google.com/file/d/1Tacnafczh24Fqet9Upmn4_amPSBO1hKs/view?usp=sharing

## Dataset used
https://github.com/taufiqhusada/amr-indo-dataset

# Indonesian AMR-based summarization by incorporating AMR-to-text generation model
Using XLsum-Indonesian dataset
## Result
1. Summary AMR output: `process_xlsum_indonesian/generated_summary_amr.txt`
2. Output pembangkitan teks dengan trained model nomor 2 (best for xlsum indo): `process_xlsum_indonesian/generated_summary_amr.txt/finetuning_text_generation/result_after_additional_finetuning.csv`
3. Output pembangkitan teks dengan trained model nomor 1: `process_xlsum_indonesian/result_with_score.csv`

## Code
1. AMR-based summarization (from input article to summary graph): https://drive.google.com/drive/folders/1E3ufHOQ8r-DjfjhzfWUxl4lpvvc21cgF?usp=sharing (open it using std.stei email)
2. Generate text from summary graph: this repo (see how to run)
