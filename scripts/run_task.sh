model_size=70B
task_name=arc
data_file=./data/inductive/miniarc.jsonl
method=rule
rule_type=nl
model_name=Meta-Llama-3-${model_size}-Instruct
temperature=0.7
output_file=results/llama3_${model_size}_${task_name}_${method}_${rule_type}_${temperature}

/data/yzzhao/miniconda3/bin/conda run -n llm --no-capture-output python /data2/yzzhao/pythonCode/lm-inductive-reasoning/run_task.py \
  --task_name $task_name \
  --data_file $data_file \
  --mode generate \
  --model_name $model_name \
  --output_file $output_file \
  --method $method \
  --rule_type $rule_type \
  --temperature $temperature