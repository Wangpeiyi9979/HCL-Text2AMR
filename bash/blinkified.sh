export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=$1 python bin/blinkify.py \
  --datasets data/tmp/test$2.0-pred.txt \
  --out  data/tmp/test$2.0-pred.blinkified.txt \
  --device cuda \
  --blink-models-dir BLINK/models/