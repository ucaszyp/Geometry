CUDA_VISIBLE_DEVICES=1 \
python main.py single \
--seed 42 \
--phase train \
--batch-size 6 \
--epochs 200 \
--lr 0.01 \
--momentum 0.9 \
--lr-mode poly \
--workers 12 \
--task depth \
--use_uncertain 1 \
--uncertain_all 1
