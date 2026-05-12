now=$(date +"%Y%m%d_%H%M%S")

logdir="./finetune_log/p0.7_con_revival"
# logdir="./finetune_log/exp_$now"

datapath="/home/Data/datasets/imagenet1k/"

ckpt="./checkpoints/deit_small_patch16_224-cd65a155.pth"

echo "output dir: $logdir"

python3 main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--sched cosine \
	--lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size 128 \
	--update-freq 8 \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--finetune $ckpt \
	--data-path $datapath \
	--output_dir $logdir \

echo "output dir for the last exp: $logdir"
