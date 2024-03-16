export_path=$1
model=$2
python micace_evaluator.py $export_path $model attention
python micace_evaluator.py $export_path $model gradient
python micace_evaluator.py $export_path $model attattr