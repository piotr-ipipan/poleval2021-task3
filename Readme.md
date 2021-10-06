# PolEval 2021 Task 3: Post-correction of OCR results

## Results

| Submission                   | test-A | test-B     |
|------------------------------|--------|------------|
| ED 3 (mt5-base)              | 4.986  | 5.001      |
| ED 3 pl (plt5-large)         | 4.281  | **4.302**  |
| plt5-base                    | 5.355  | 5.327      |
|                              |        |            |
| Mateusz Piotrowski (mt5-xxl) | 3.725  | **3.744**  |

Model plT5-large achieves 4.302 WER score. The best solution is 3.744 WER score using xxl model.

## Instructions

```shell
git clone https://github.com/poleval/2021-ocr-correction.git
mkdir data
```

You can use model `enelpol/poleval2021-task3` or train yourself.

### Training

1. Prepare aligned texts using space-tokenizer.
```shell
python3 align.py 2021-ocr-correction/dev-0/in.tsv 2021-ocr-correction/dev-0/expected.tsv data/dev.jsonl data/dev-baseline.tsv
python3 align.py 2021-ocr-correction/train/in.tsv 2021-ocr-correction/train/expected.tsv data/train.jsonl data/train-baseline.tsv
python3 align.py 2021-ocr-correction/test-A/in.tsv 2021-ocr-correction/test-A/expected.tsv data/test-A.jsonl data/test-A-baseline.tsv
```

2. Prepare data for training.
```shell
python3 chunk_generator_context.py data/dev.jsonl data/dev.train.tsv --chunk 50 --stride 0
Input and output size in subtokens:
Input mean: 100.20682910235149, max: 157, min: 1
Output mean: 99.27256878003146, max: 165, min: 1

python3 chunk_generator_context.py data/train.jsonl data/train.train.tsv --chunk 50 --stride 0
Input and output size in subtokens:
Input mean: 100.26346787291853, max: 197, min: 1
Output mean: 99.32334532501505, max: 241, min: 1

python3 chunk_generator_context.py data/test-A.jsonl data/test-A.train.tsv --chunk 50 --stride 0
Input and output size in subtokens:
Input mean: 100.11499283996649, max: 179, min: 1
Output mean: 99.20710059171597, max: 179, min: 1
```
```shell
shuf data/train.train.tsv > data/train.train.tsv.shuf
```

3. Train mt5.
```shell
python3 train_mt5.py data/train.train.tsv.shuf data/dev.train.tsv --batch_size 12 --num_beams 2 --warmup_steps 1 --epochs 1 --evaluate_during_training_steps 500 --scheduler linear_schedule_with_warmup
python3 train_mt5.py data/train.train.tsv.shuf data/dev.train.tsv --batch_size 12 --num_beams 2 --warmup_steps 1 --epochs 1 --evaluate_during_training_steps 500 --scheduler linear_schedule_with_warmup --model_name allegro/plt5-base --model_type t5
python3 train_mt5.py data/train.train.tsv.shuf data/dev.train.tsv --batch_size 12 --num_beams 2 --warmup_steps 1 --epochs 1 --evaluate_during_training_steps 500 --scheduler linear_schedule_with_warmup --model_name allegro/plt5-large --model_type t5
```

### Prediciton

4. Predict.
```shell
MODEL="enelpol/poleval2021-task3" # "model_bin_*/best_model/" #edit
MODEL_TYPE=t5
python3 predict_mt5.py 2021-ocr-correction/test-A/in.tsv 2021-ocr-correction/test-A/output.tsv --model_name $MODEL --model_type $MODEL_TYPE --chunk 50 --stride 0 --num_beams 8 --batch_size 8
python3 predict_mt5.py 2021-ocr-correction/test-B/in.tsv 2021-ocr-correction/test-B/output.tsv --model_name $MODEL --model_type $MODEL_TYPE --chunk 50 --stride 0 --num_beams 8 --batch_size 8
```