PATH=/data/quyet/comet-atomic-2020
# pop_cskb??

# train
CUDA_VISIBLE_DEVICES=7 TRAIN_BATCH_SIZE=64 DO_TRAIN=True DO_PRED=False \
TRAIN_DATA_PATH=kg/atomic2020_data-feb2021/train_withInversedSample.tsv DEV_DATA_PATH=kg/atomic2020_data-feb2021/dev.tsv TEST_DATA_PATH=kg/atomic2020_data-feb2021/test.tsv \
OUT_DIR=models/comet_gpt2b_inversedPopAtomic2020 TOKENIZER=gpt2 GPT2_MODEL=gpt2 \
python models/comet_atomic2020_gpt2/comet_gpt2.py    

# generation
CUDA_VISIBLE_DEVICES=7 TRAIN_BATCH_SIZE=64 DO_TRAIN=False DO_PRED=True \
TRAIN_DATA_PATH=kg/atomic2020_data-feb2021/train_withInversedSample.tsv DEV_DATA_PATH=kg/atomic2020_data-feb2021/dev.tsv TEST_DATA_PATH=kg/atomic2020_data-feb2021/test.tsv PRED_FILE=eval/test_for_gpt2.tsv \
OUT_DIR=models/comet_gpt2b_inversedPopAtomic2020 TOKENIZER=gpt2 GPT2_MODEL=models/comet_gpt2b_inversedPopAtomic2020/final_model_gpt2.pt \
python models/comet_atomic2020_gpt2/comet_gpt2.py 

# evaluate
python eval/eval.py