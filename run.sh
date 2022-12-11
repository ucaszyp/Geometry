CUDA_VISIBLE_DEVICES=3 \
python main.py single \
--seed 42 \
--phase train \
--batch-size 2 \
--epochs 200 \
--lr 0.014 \
--momentum 0.9 \
--lr-mode poly \
--workers 12 \
--task normal \
--use_uncertain 1
