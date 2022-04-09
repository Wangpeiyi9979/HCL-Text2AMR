

# bash bash/blinkified-fine-eval.sh 0 3
echo '=>smatch scores without postprocessing'
smatch.py -f data/tmp/test$2.0-pred.txt ./data/tmp/test$2.0-gold.txt
echo '=> Postprocessing...'
export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=$1 python bin/blinkify.py \
  --datasets data/tmp/test$2.0-pred.txt \
  --out  data/tmp/test$2.0-pred.blinkified.txt \
  --device cuda \
  --blink-models-dir BLINK/models/

cd amr-evaluation/
export PYTHONPATH=`pwd`
echo 'Evaluation..'
./evaluation.sh ../data/tmp/test$2.0-pred.blinkified.txt ../data/tmp/test$2.0-gold.txt


  # --gold-path data/tmp/test$2.0-gold.txt     \
  # --pred-path data/tmp/test$2.0-pred.txt    \