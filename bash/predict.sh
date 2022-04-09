CUDA_VISIBLE_DEVICES=$1 python bin/predict_amrs.py \
  --datasets ./data/amr_annotation_$2.0/data/amrs/split/test/*.txt   \
  --gold-path data/tmp/test$2.0-gold.txt     \
  --pred-path data/tmp/test$2.0-pred.txt   \
  --checkpoint $3   \
  --beam-size 5     \
  --batch-size 5000     \
  --device cuda     \
  --penman-linearization \
  --use-pointer-tokens
