import asyncio
import json
import logging
import os
import pickle
from math import ceil
from random import random
from pydantic import BaseModel

import anthropic
import openai
from openai import AsyncOpenAI

import tiktoken
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

ANTHROPIC_CLIENT = anthropic.AsyncAnthropic()

HISTORY_FILE = "/Users/fengwenjun/Desktop/CODE/Code_Resoning/code_reasoning/results/history.jsonl"
CACHE_FILE = "query_cache.pkl"
EXP_CAP = 4

logger = logging.getLogger(__name__)

MODEL2COST = {
    "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.0000025, "output": 0.0000075},
    "gpt-4o-2024-08-06": {"input": 0.00000125, "output": 0.000005},
}

MODEL2BATCH_SIZE = {
    "gpt-3.5-turbo-0613": 500,
    "gpt-4-0613": 500,
    "claude-2": 100,
    "llama3": 500,
    "gpt-4o": 500,
    "gpt-4o-2024-08-06": 500,
}

GPT_MODELS = {"gpt-3.5-turbo-0613", "gpt-4-0613", "gpt-4o", "gpt-4o-2024-08-06"}
CLAUDE_MODELS = {"claude-2"}
LLAMA_MODELS = {"Llama-3-70B-Instruct, Llama-3-8B-Instruct, Llama-3-8B, Llama-3-70B"}
class Step(BaseModel):
    Subrule: str
    input: str
    output: str

class CodeReasoning(BaseModel): #重要的理解地方
    steps: list[Step]
    rule: str


async def query_openai_struct(
    prompt,
    model_name,
    system_msg,
    history,
    max_tokens=None,
    temperature=0,
    retry=100,
    n=1,
    history_file=HISTORY_FILE,
    **kwargs,
):
    # reference: https://github.com/ekinakyurek/mylmapis/blob/b0adb192135898fba9e9dc88f09a18dc64c1f1a9/src/network_manager.py
    messages = []
    if system_msg is not None:
        messages += [{"role": "system", "content": system_msg}]
    if history is not None:
        messages += history
    messages += [{"role": "user", "content": prompt}]
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    kwargs["n"] = n
    aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # response = await aclient.chat.completions.create(model=model_name, messages=messages, **kwargs)
    for i in range(retry + 1):
        wait_time = (1 << min(i, EXP_CAP)) + random() / 10
        try:
            response = await aclient.beta.chat.completions.parse(model=model_name, messages=messages, response_format=CodeReasoning, **kwargs)
            print("history_file : ", history_file)
            with open(history_file, "a") as f:
                f.write(json.dumps((model_name, messages, kwargs, response.to_json())) + "\n")
            if any(choice.finish_reason != "stop" for choice in response.choices):
                print("Truncated response!")
                print(response)
            contents = [choice.message.content for choice in response.choices]
            if n == 1:
                return contents[0]
            else:
                return contents
        except (
            openai.AuthenticationError,
            openai.APIStatusError,
            openai.APIError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            if i == retry:
                raise e
            else:
                await asyncio.sleep(wait_time)
        except openai.BadRequestError as e:
            logger.error(e)
            if n == 1:
                return ""
            else:
                return [""] * n

async def query_openai(
    prompt,
    model_name,
    system_msg,
    history,
    max_tokens=None,
    temperature=0,
    retry=100,
    n=1,
    history_file=HISTORY_FILE,
    **kwargs,
):
    # reference: https://github.com/ekinakyurek/mylmapis/blob/b0adb192135898fba9e9dc88f09a18dc64c1f1a9/src/network_manager.py
    messages = []
    if system_msg is not None:
        messages += [{"role": "system", "content": system_msg}]
    if history is not None:
        messages += history
    messages += [{"role": "user", "content": prompt}]
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    kwargs["n"] = n
    aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # response = await aclient.chat.completions.create(model=model_name, messages=messages, **kwargs)
    for i in range(retry + 1):
        wait_time = (1 << min(i, EXP_CAP)) + random() / 10
        try:
            response = await aclient.chat.completions.create(model=model_name, messages=messages, **kwargs)
            with open(history_file, "a") as f:
                f.write(json.dumps((model_name, messages, kwargs, response.to_json())) + "\n")
            if any(choice.finish_reason != "stop" for choice in response.choices):
                print("Truncated response!")
                print(response)
            contents = [choice.message.content for choice in response.choices]
            if n == 1:
                return contents[0]
            else:
                return contents
        except (
            openai.AuthenticationError,
            openai.APIStatusError,
            openai.APIError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.RateLimitError,
            openai.ConflictError,
            openai.RateLimitError,
        ) as e:
            if i == retry:
                raise e
            else:
                await asyncio.sleep(wait_time)
        except openai.BadRequestError as e:
            logger.error(e)
            if n == 1:
                return ""
            else:
                return [""] * n


async def query_anthropic(
    prompt,
    model_name,
    system_msg,
    history,
    max_tokens=9000,
    temperature=0,
    retry=100,
    history_file=HISTORY_FILE,
    **kwargs,
):
    assert system_msg is None
    prompt_history = []
    prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
    messages = []
    if history is not None:
        prompt_history = ""
        for his in history:
            if his["role"] == "user":
                prompt_history += f"{anthropic.HUMAN_PROMPT} {his['content']} "
            elif his["role"] == "assistant":
                prompt_history += f"{anthropic.AI_PROMPT} {his['content']} "
        prompt = prompt_history + prompt
    messages += [{"role": "user", "content": prompt}]
    if max_tokens is None:
        max_tokens = 9000
    kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature

    for i in range(retry + 1):
        wait_time = (1 << min(i, EXP_CAP)) + random() / 10
        try:
            response = await ANTHROPIC_CLIENT.completions.create(
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=model_name,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            with open(history_file, "a") as f:
                f.write(
                    json.dumps((model_name, messages, kwargs, response.completion))
                    + "\n"
                )
            if response.stop_reason != "stop_sequence":
                print("Truncated response!")
                print(response)
            return response.completion.lstrip()
        except anthropic.BadRequestError as e:
            logger.error(e)
            return ""
        except anthropic.APIError as e:
            if i == retry:
                raise e
            else:
                await asyncio.sleep(wait_time)


def query_batch_wrapper(
    fn, prompts, model_name, system_msg, histories, *args, **kwargs
):
    async def _query(prompts, histories):
        async_responses = [
            fn(prompt, model_name, system_msg, his, *args, **kwargs)
            for prompt, his in zip(prompts, histories)
        ]
        return await tqdm_asyncio.gather(*async_responses)

    all_results = asyncio.run(_query(prompts, histories))
    return all_results


def query_batch(
    prompts,
    model_name,
    system_msg=None,
    histories=None,
    max_tokens=None,
    temperature=0,
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

        num_calls_per_n = 1 if model_name in GPT_MODELS else n
        batch_size = ceil(MODEL2BATCH_SIZE[model_name] / num_calls_per_n)
        total_batches = ceil(len(unseen_prompts) / batch_size)
        for start in tqdm(
            range(0, len(unseen_prompts), batch_size),
            desc="Querying batch",
            total=total_batches,
        ):
            unseen_prompts_batch = unseen_prompts[start : start + batch_size]
            unseen_histories_batch = unseen_histories[start : start + batch_size]
            if model_name in GPT_MODELS:
                responses = query_batch_wrapper(
                    query_openai,
                    unseen_prompts_batch,
                    model_name,
                    system_msg,
                    unseen_histories_batch,
                    max_tokens,
                    temperature,
                    retry,
                    n,
                    history_file,
                    **openai_kwargs,
                )
            elif model_name in CLAUDE_MODELS:
                assert system_msg is None
                if n > 1:
                    num_prompts = len(unseen_prompts_batch)
                    orig_unseen_prompts_batch = unseen_prompts_batch
                    unseen_prompts_batch = [
                        prompt for prompt in unseen_prompts_batch for _ in range(n)
                    ]
                    orig_unseen_histories_batch = unseen_histories_batch
                    unseen_histories_batch = [
                        his for his in unseen_histories_batch for _ in range(n)
                    ]
                responses = query_batch_wrapper(
                    query_anthropic,
                    unseen_prompts_batch,
                    model_name,
                    system_msg,
                    unseen_histories_batch,
                    max_tokens,
                    temperature,
                    retry,
                    **openai_kwargs,
                )
                if n > 1:
                    responses = [
                        responses[i : i + n] for i in range(0, len(responses), n)
                    ]
                    assert (
                        len(responses) * n
                        == num_prompts * n
                        == sum(len(r) for r in responses)
                        == len(unseen_prompts_batch)
                        == len(unseen_histories_batch)
                    )
                    unseen_prompts_batch = orig_unseen_prompts_batch
                    unseen_histories_batch = orig_unseen_histories_batch
            elif model_name in LLAMA_MODELS:
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Reload cache for better concurrency. Otherwise multiple query processes can overwrite
            # each other
            cache = {}
            if not skip_cache:
                if os.path.exists(cache_file):
                    cache = pickle.load(open(cache_file, "rb"))
            for prompt, his, response in zip(
                unseen_prompts_batch, unseen_histories_batch, responses
            ):
                key = prompt2key(prompt, his)
                cache[key] = response
            if not skip_cache:
                pickle.dump(cache, open(cache_file, "wb"))

    return [cache[prompt2key(prompt, his)] for prompt, his in zip(prompts, histories)]

def query_batch_struct(
    prompts,
    model_name,
    system_msg=None,
    histories=None,
    max_tokens=None,
    temperature=0,
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

        num_calls_per_n = 1 if model_name in GPT_MODELS else n
        batch_size = ceil(MODEL2BATCH_SIZE[model_name] / num_calls_per_n)
        total_batches = ceil(len(unseen_prompts) / batch_size)
        for start in tqdm(
            range(0, len(unseen_prompts), batch_size),
            desc="Querying batch",
            total=total_batches,
        ):
            unseen_prompts_batch = unseen_prompts[start : start + batch_size]
            unseen_histories_batch = unseen_histories[start : start + batch_size]
            if model_name in GPT_MODELS:
                responses = query_batch_wrapper( #重要的地方，开始调用api的地方
                    query_openai_struct,
                    unseen_prompts_batch,
                    model_name,
                    system_msg,
                    unseen_histories_batch,
                    max_tokens,
                    temperature,
                    retry,
                    n,
                    history_file,
                    **openai_kwargs,
                )
            elif model_name in CLAUDE_MODELS:
                assert system_msg is None
                if n > 1:
                    num_prompts = len(unseen_prompts_batch)
                    orig_unseen_prompts_batch = unseen_prompts_batch
                    unseen_prompts_batch = [
                        prompt for prompt in unseen_prompts_batch for _ in range(n)
                    ]
                    orig_unseen_histories_batch = unseen_histories_batch
                    unseen_histories_batch = [
                        his for his in unseen_histories_batch for _ in range(n)
                    ]
                responses = query_batch_wrapper(
                    query_anthropic,
                    unseen_prompts_batch,
                    model_name,
                    system_msg,
                    unseen_histories_batch,
                    max_tokens,
                    temperature,
                    retry,
                    **openai_kwargs,
                )
                if n > 1:
                    responses = [
                        responses[i : i + n] for i in range(0, len(responses), n)
                    ]
                    assert (
                        len(responses) * n
                        == num_prompts * n
                        == sum(len(r) for r in responses)
                        == len(unseen_prompts_batch)
                        == len(unseen_histories_batch)
                    )
                    unseen_prompts_batch = orig_unseen_prompts_batch
                    unseen_histories_batch = orig_unseen_histories_batch
            elif model_name in LLAMA_MODELS:
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Reload cache for better concurrency. Otherwise multiple query processes can overwrite
            # each other
            cache = {}
            if not skip_cache:
                if os.path.exists(cache_file):
                    cache = pickle.load(open(cache_file, "rb"))
            for prompt, his, response in zip(
                unseen_prompts_batch, unseen_histories_batch, responses
            ):
                key = prompt2key(prompt, his)
                cache[key] = response
            if not skip_cache:
                pickle.dump(cache, open(cache_file, "wb"))

    return [cache[prompt2key(prompt, his)] for prompt, his in zip(prompts, histories)]

def get_cost(input, outputs, model_name=None, history=None):
    if model_name not in MODEL2COST:
        return 0
    costs = MODEL2COST[model_name]
    enc = tiktoken.encoding_for_model(model_name)
    input_tokens = len(enc.encode(input))
    if isinstance(outputs, str):
        outputs = [outputs]
    for output in outputs:
        output_tokens = len(enc.encode(output))
        cost = input_tokens * costs["input"] + output_tokens * costs["output"]
        if history is not None:
            for his in history:
                if his["role"] == "user":
                    cost += len(enc.encode(his["content"])) * costs["input"]
                elif his["role"] == "assistant":
                    cost += len(enc.encode(his["content"])) * costs["input"]
    return cost / 1000


def format_history(query, response):
    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
