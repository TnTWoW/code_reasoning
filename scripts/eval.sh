data_file= "./data/inductive/list_function.jsonl"
input_file=Llama8B_LF_iid_cache.jsonl
output_file=results/llama8B_LF_iid_output
model_name=Meta-Llama-8B-Instruct
task_name=list_function
n_examples=100
cache_file=Llama8B_LF_iid_cache.jsonl
/data/yzzhao/miniconda3/bin/conda run -n llm --no-capture-output python /data2/yzzhao/pythonCode/lm-inductive-reasoning/eval_task.py \
  --data_file $data_file \
  --input_file $input_file \
  --output_file $output_file \
  --model_name $model_name \
  --task_name $task_name \
  --n_examples $n_examples \
  --cache_file $cache_file