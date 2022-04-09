cd amr-evaluation/
export PYTHONPATH=`pwd`
echo 'Evaluation..'
./evaluation.sh ../data/tmp/test$1.0-pred.blinkified.txt ../data/tmp/test$1.0-gold.txt

