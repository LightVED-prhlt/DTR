datapath="/home/Data/datasets/imagenet1k/"

r05="logs/dvit_deit-s-224_r0.5/checkpoint-best.pth"
r05_dtr="logs/dvit_deit-s_224_r0.5_rkt/checkpoint-best.pth"

r06="logs/dvit_deit-s-224_r0.6/checkpoint-best.pth"
r06_dtr="logs/dvit_deit-s-224_r0.6_rkt/checkpoint-best.pth"

r07="checkpoints/dvit-deit-s-384_r0.7.pth"
r07_dtr="logs/dvit-deit-s-224_r0.7_rkt/checkpoint-best.pth"

python infer.py \
    --data_path $datapath \
    --model deit-s \
    --model_path $r07_dtr \
    --base_rate 0.7 \
    --input_size 224 \
    --with_dtr
