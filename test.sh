output_file=/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/results/
model_name=gpt-4o-2024-08-06
# robust_fill
# data_file="/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/data/inductive/RobustFill.json"
# task_name=robust_fill

# miniarc
data_file="/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/data/inductive/miniarc.jsonl"
task_name=arc

history_file=/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/results/history
n_examples=3
cache_file=/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/results/cache1.jsonl

export OPENAI_API_KEY=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
# 获取当前时间，格式为YYYYMMDD_HHMMSS
current_time=$(date +"%Y%m%d_%H%M%S")

# 生成新的output_file路径
output_file="${output_dir}/${current_time}_${model_name}_${task_name}"

conda run -n codereasoning2 --no-capture-output python /Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/run_task.py \
  --data_file $data_file \
  --output_file $output_file \
  --model_name $model_name \
  --task_name $task_name \
  --n_examples $n_examples \
  --cache_file $cache_file \

# conda run -n codereasoning --no-capture-output python -m pdb /Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/run_task.py \
#   --data_file $data_file \
#   --output_file $output_file \
#   --model_name $model_name \
#   --task_name $task_name \
#   --n_examples $n_examples \
#   --cache_file $cache_file