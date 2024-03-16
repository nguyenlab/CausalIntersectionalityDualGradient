export_path=$1
model=$2
python miate_evaluator.py $export_path $model attention
python miate_evaluator.py $export_path $model gradient
python miate_evaluator.py $export_path $model attattr
python miate_evaluator.py $export_path $model attention --pick_correct_labels
python miate_evaluator.py $export_path $model gradient --pick_correct_labels
python miate_evaluator.py $export_path $model attattr --pick_correct_labels