now=$(date +"%Y%m%d_%H%M%S")
datapath="/home/Data/datasets/imagenet1k/"
base_rate=0.7

logdir="logs/dvit_deit-s-224_r{$base_rate}_{$now}"
# logdir="logs/dvit_deit-s-224_r{$base_rate}_dtr_{$now}"

python main.py \
    --output_dir $logdir \
    --model deit-s \
    --input_size 224 \
    --batch_size 128 \
    --update_freq 8 \
    --data_path $datapath \
    --epochs 30 \
    --base_rate $base_rate \
    --lr 1e-3 \
    --warmup_epochs 5 \
    # --with_dtr