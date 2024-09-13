import asyncio
import json
import logging
import os
import pickle
from math import ceil
from random import random

from typing import List

import transformers
import torch

HISTORY_FILE = "history.jsonl"
CACHE_FILE = "generate_cache.pkl"
EXP_CAP = 4

logger = logging.getLogger(__name__)

GPT_MODELS = {"gpt-3.5-turbo-0613", "gpt-4-0613"}
CLAUDE_MODELS = {"claude-2"}
LLAMA_MODELS = {"Llama-3-70B-Instruct, Llama-3-8B-Instruct, Llama-3-8B, Llama-3-70B"}
SYSTEM_MSG = [{"role": "system", "content": "You are a professional Python programmer. Please write a Python program according to the instructions."}]

def generate_llama(
    prompts,
    pipeline,
    model_name,
    system_msg,
    historys,
    max_tokens=256,
    temperature=0.6,
    n=1,
    history_file=HISTORY_FILE,
):
    # load model
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    responses = []
    for prompt, his in zip(prompts, historys):
        prompt = [{"role": "user", "content": prompt}]
        if his is not None:
            prompt = his + prompt
        prompt = SYSTEM_MSG + prompt
        prompt = pipeline.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        response = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=n,
        )

        response = [sample['generated_text'][len(prompt):] for sample in response]
        if n == 1:
            response = response[0]
        responses.append(response)
    return responses


def generate_batch_wrapper(
    fn, prompts, pipeline, model_name, system_msg, histories, *args, **kwargs
):
    def _generate(prompts, histories):
        return fn(prompts, pipeline, model_name, system_msg, histories, *args, **kwargs)
    return _generate(prompts, histories)
def generate_batch(
    prompts,
    pipeline,
    model_name,
    system_msg=None,
    histories=None,
    max_tokens=256,
    temperature=0.6,
    retry=100,
    num_beams=1,
    skip_cache=False,
    n=1,
    cache_file=CACHE_FILE,
    history_file=HISTORY_FILE,
    **openai_kwargs,
):
    cache = {}
    if not skip_cache and os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, "rb"))

    prompt2key = lambda p, h: (
        p,
        model_name,
        system_msg,
        tuple([tuple(e.items()) for e in h]) if h is not None else None,
        max_tokens,
        temperature,
        num_beams,
        n,
    )

    unseen_prompt_pairs = set()
    if histories is None:
        histories = [None] * len(prompts)
    for prompt, his in zip(prompts, histories):
        key = prompt2key(prompt, his)
        if (
                (key not in cache)
                or (key in cache and n == 1 and cache[key] is None)  # previous call failed
                or (key in cache and n > 1 and None in cache[key])
        ):
            history_cache_key = tuple([tuple(e.items()) for e in his]) if his else None
            prompt_key = prompt if isinstance(prompt, str) else tuple(prompt)
            unseen_prompt_pairs.add((prompt_key, history_cache_key))
    unseen_prompts = []
    unseen_histories = []
    for prompt, his in unseen_prompt_pairs:
        unseen_prompts.append(prompt)
        if his is not None:
            his = [dict(e) for e in his]
        unseen_histories.append(his)

    if len(unseen_prompts) > 0:
        logger.info("History:")
        logger.info(unseen_histories[0])
        logger.info("Prompt:")
        logger.info(unseen_prompts[0])
        logger.info(f"Calling {model_name} for {len(unseen_prompts)} prompts")
        responses = generate_batch_wrapper(
            generate_llama,
            unseen_prompts,
            pipeline,
            model_name,
            system_msg,
            unseen_histories,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            history_file=history_file,
            **openai_kwargs,
        )
        cache = {}
        if not skip_cache:
            if os.path.exists(cache_file):
                cache = pickle.load(open(cache_file, "rb"))
        for prompt, his, response in zip(
                unseen_prompts, unseen_histories, responses
        ):
            key = prompt2key(prompt, his)
            cache[key] = response
        if not skip_cache:
            pickle.dump(cache, open(cache_file, "wb"))

    return [cache[prompt2key(prompt, his)] for prompt, his in zip(prompts, histories)]


if __name__ == "__main__":
    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    generate_batch(prompts=prompts, model_name="Llama-3-8B-Instruct")