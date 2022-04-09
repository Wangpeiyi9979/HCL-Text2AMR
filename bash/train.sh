
# bash bash/train/layer-amr-cur-train.sh 0 bart-large
CUDA_VISIBLE_DEVICES=$1 python bin/train.py \
--config configs/HCL.yaml \
--direction amr \
--IC_steps 500 \
--SC_steps 1000 \
