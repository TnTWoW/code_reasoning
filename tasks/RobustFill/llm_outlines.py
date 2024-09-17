import outlines.text.generate.sample as sample
import outlines.text.generate as generate
import outlines.models as models
import torch
from transformers import BitsAndBytesConfig

# base_model_name = "reciprocate/tiny-llama"
# base_model_name = "mistralai/Mistral-7B-v0.1"
# base_model_name = "meta-llama/Llama-2-7b-hf"
# base_model_name = "meta-llama/Llama-2-13b-hf"
# base_model_name = "meta-llama/Llama-2-70b-hf"
base_model_name = "Phind/Phind-CodeLlama-34B-v2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = models.transformers(
    base_model_name,
    device="auto",
    model_kwargs={
        "quantization_config": bnb_config,
        #"device_map":"auto",
        "trust_remote_code": True,
        "use_auth_token": True,
    },
)

def gen(prompt, choices):
    return gen_greedy(prompt, choices)

def gen_greedy(prompt, choices):
    return generate.choice(model, choices, sampler=sample.greedy)(prompt)

if __name__ == '__main__':
    print(gen("Do you like apples or oranges?", ["apples", "oranges"]))
