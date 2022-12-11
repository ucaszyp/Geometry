CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=27916 main.py single \
--seed 42 \
--phase train \
--batch-size 2 \
--epochs 200 \
--lr 0.014 \
--momentum 0.9 \
--lr-mode poly \
--workers 12 \
--task depth \
--use_uncertain 1 \
--vis 1