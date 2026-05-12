datapath="/home/Data/datasets/imagenet1k/"

p05="finetune_log/p0.5_sin_revival/checkpoint_best.pth"
p05_rkt="finetune_log/p0.5_con_revival/checkpoint_best.pth"

p06="finetune_log/p0.6_sin_revival/checkpoint_best.pth"
p06_rkt="finetune_log/p0.6_con_revival/checkpoint_best.pth"

p07="finetune_log/p0.7_sin_revival/checkpoint_best.pth"
p07_rkt="finetune_log/p0.7_con_revival/checkpoint_best.pth"

python3 main.py \
    --model deit_small_patch16_shrink_base \
    --base_keep_rate 0.7 \
    --resume $p07_rkt \
    --data-path $datapath \
    --eval \
    --fuse_token
