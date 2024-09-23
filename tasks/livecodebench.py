import copy
import logging
import re
import numpy as np
import ast
from utils.python_utils import execute_function, extract_function_names
from tqdm import tqdm

from prompts.livecodebench import (
    input_prompt,
    cot_input_prompt,
    coc_input_prompt,
    rule_input_prompt,
    rule_with_feedback_input_prompt,
    output_prompt,
    cot_output_prompt,
    coc_output_prompt,
    rule_output_prompt,
    rule_with_feedback_output_prompt,
)
from tasks.base import Task
from utils.format_utils import format_grid, format_list, str_to_list, unformat_grid
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)
PRINT_NUM = 3
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

class LiveCodeBench(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_code_examples(self, data):
        return data["code"][: self.n_train]
    def get_input_examples(self, data):
        return data["input"][: self.n_train]
    def get_output_examples(self, data):
        return data["output"][: self.n_train]

    def get_all_examples(self, split, idxs=None):
        if idxs is None:
            idxs = list(range(len(self.data)))
        if split == "code":
            return [self.get_code_examples(self.data[i]) for i in idxs]
        elif split == "input":
            return [self.get_input_examples(self.data[i]) for i in idxs]
        elif split == "output":
            return [self.get_output_examples(self.data[i]) for i in idxs]
        else:
            raise ValueError(f"Invalid split: {split}")

    def extract_rules(self, response):
        pattern = r'Analysis:(.*?)Answer:'
        results = re.findall(pattern, response, re.DOTALL)
        if not results:
            return response
        return results[-1].strip()

    def apply_all_rules(self, idxs, all_rules, all_examples):
        pass

class LiveCodeBenchInput(LiveCodeBench):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = input_prompt
        self.cot_prompt = cot_input_prompt
        self.coc_prompt = coc_input_prompt
        self.rule_prompt = rule_input_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_input_prompt

    def extract_input_prediction(self, text, func_name):
        pattern = fr'{func_name}\((.*?)\)'
        results = re.findall(pattern, text)
        if not results:
            return ""
        return results[-1]

    def get_metrics(self, codes: list, pred_inputs: list, all_outputs: list) -> dict:
        all_pred_outputs = []
        for code, pred_input, outputs in\
                tqdm(zip(codes, pred_inputs, all_outputs),
                     desc="Evaluating results", total=len(codes)):
            input = copy.deepcopy([pred_input])
            pred_outputs = execute_function(code, input)
            all_pred_outputs.append(pred_outputs[0])
        acc = np.mean([a == safe_literal_eval(o) for a, o in zip(all_pred_outputs, all_outputs)])
        output_dict = {
            "test_instance_acc": acc,
            "test_acc": acc,
        }
        return output_dict

    def eval_test_from_rule(self, responses):
        codes = self.get_all_examples("code")
        outputs = self.get_all_examples("output")
        func_names = [extract_function_names(code)[-1] for code in codes]
        assert all([response is not None for response in responses])
        answers = [self.extract_input_prediction(response, func_name) for response, func_name in zip(responses, func_names)]
        output_dict = self.get_metrics(codes, answers, outputs)
        return output_dict
    def get_feedback(self, code, output, rule, pred_input, input):
        p_input = copy.deepcopy([pred_input])
        pred_outputs = execute_function(code, p_input)[0]
        if safe_literal_eval(output) != pred_outputs:
            func_name = extract_function_names(code)[-1]
            return self.rule_with_feedback_prompt.format(code=code, func_name=func_name, output=output, rule=rule, p_input=p_input)
        return ""

    def get_best_outputs(self, responses, codes):
        func_names = [extract_function_names(code)[-1] for code in codes]
        if self.n == 1:
            responses = [self.extract_input_prediction(r, fn) for r, fn in zip(responses, func_names)]
            return responses
        responses = [
            [str(self.extract_input_prediction(r, fn)) for r in res] for res, fn in zip(responses, func_names)
        ]
        best_outputs = [max(set(res), key=res.count) for res in responses]
        return best_outputs
    def eval_io(self):
        all_codes = self.get_all_examples("code")
        all_outputs = self.get_all_examples("output")

        prompts = []
        idxs = []
        for i, (code, output) in enumerate(zip(all_codes, all_outputs)):
            func_name = extract_function_names(code)[-1]
            if self.rule_type == 'io':
                prompt = self.io_prompt.format(code=code, func_name=func_name, output=output)
            elif self.rule_type == 'cot':
                prompt = self.cot_prompt.format(code=code, func_name=func_name, output=output)
            elif self.rule_type == 'coc':
                prompt = self.coc_prompt.format(code=code, func_name=func_name, output=output)
            else:
                raise ValueError(f"Invalid rule type: {self.rule_type}")
            prompts.append(prompt)
            idxs.append(i)
        responses = self.query(prompts, idxs, histories=None)
        responses = self.get_best_outputs(responses, all_codes)
        metrics = self.get_metrics(all_codes, responses, all_outputs)
        self.metrics.append(metrics)

    def eval_rule(self):
        all_codes = self.get_all_examples("code")
        all_outputs = self.get_all_examples("output")

        prompts = []
        idxs = []
        for i, (code, output) in enumerate(zip(all_codes, all_outputs)):
            func_name = extract_function_names(code)[-1]
            prompt = self.rule_prompt.format(code=code, func_name=func_name, output=output)
            prompts.append(prompt)
            idxs.append(i)
        idx_to_response = [None for _ in range(len(self.data))]

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            responses = self.query(prompts, idxs, histories=histories)
            if self.n > 1:
                all_train_examples = self.get_all_examples("train", idxs)
                logger.info(f"Reranking {len(all_train_examples)} train examples...")
                if self.verbose:
                    logger.info(f"Responses before reranking:")
                    for res in responses[:PRINT_NUM]:
                        logger.info(res)
                responses = self.get_best_responses(idxs, all_train_examples, responses)
            for idx, response in zip(idxs, responses):
                idx_to_response[idx] = response

            rules = [self.extract_rules(response) for response in responses]
            self.add_rules(idxs, rules)

            if self.max_iter > 1:
                self.add_histories("user", idxs, prompts)
                self.add_histories("assistant", idxs, responses)


                all_inputs = self.get_all_examples("input", idxs)
                all_codes = self.get_all_examples("code", idxs)
                all_outputs = self.get_all_examples("output", idxs)
                all_func_names = [extract_function_names(code)[-1] for code in all_codes]
                all_train_inputs = [self.extract_input_prediction(response, func_name) for response, func_name in zip(responses, all_func_names)]
                prompts = []
                new_idxs = []
                for idx, rule, code, output, train_inputs, inputs in\
                        tqdm(zip(idxs, rules, all_codes, all_outputs, all_train_inputs, all_inputs),
                             desc="Generating feedback", total=len(idxs)):
                    prompt = self.get_feedback(code, output, rule, train_inputs, inputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_test_from_rule(idx_to_response)
            self.metrics.append(metrics)
class LiveCodeBenchOutput(LiveCodeBench):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = output_prompt
        self.cot_prompt = cot_output_prompt
        self.coc_prompt = coc_output_prompt
        self.rule_prompt = rule_output_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_output_prompt

    def extract_output_prediction(self, text):
        pattern = r'==\s*(.*?)\s*```'
        results = re.findall(pattern, text)
        if not results:
            return ""
        return results[-1]
    def get_metrics(self, answers: list) -> dict:
        all_outputs = self.get_all_examples("output")
        acc = np.mean([safe_literal_eval(a) == ast.literal_eval(o) for a, o in zip(answers, all_outputs)])
        output_dict = {
            "test_instance_acc": acc,
            "test_acc": acc,
        }
        return output_dict

    def eval_test_from_rule(self, responses):
        assert all([response is not None for response in responses])
        answers = [self.extract_output_prediction(response) for response in responses]
        output_dict = self.get_metrics(answers)
        return output_dict
    def get_feedback(self, code, input, rule, p_output, output):
        if safe_literal_eval(p_output) != ast.literal_eval(output):
            return self.rule_with_feedback_prompt.format(code=code, input=input, rule=rule, p_output=p_output)
        return ""

    def get_best_outputs(self, responses):
        if self.n == 1:
            responses = [self.extract_output_prediction(r) for r in responses]
            return responses
        responses = [
            [str(self.extract_output_prediction(r)) for r in res] for res in responses
        ]
        best_outputs = [max(set(res), key=res.count) for res in responses]
        return best_outputs
    def eval_io(self):
        all_codes = self.get_all_examples("code")
        all_inputs = self.get_all_examples("input")

        prompts = []
        idxs = []
        for i, (code, input) in enumerate(zip(all_codes, all_inputs)):
            if self.rule_type == 'io':
                prompt = self.io_prompt.format(code=code, input=input)
            elif self.rule_type == 'cot':
                prompt = self.cot_prompt.format(code=code, input=input)
            elif self.rule_type == 'coc':
                prompt = self.coc_prompt.format(code=code, input=input)
            prompts.append(prompt)
            idxs.append(i)
        responses = self.query(prompts, idxs, histories=None)
        responses = self.get_best_outputs(responses, )
        metrics = self.get_metrics(responses)
        self.metrics.append(metrics)

    def eval_rule(self):
        all_codes = self.get_all_examples("code")
        all_inputs = self.get_all_examples("input")

        prompts = []
        idxs = []
        for i, (code, input) in enumerate(zip(all_codes, all_inputs)):
            prompt = self.rule_prompt.format(code=code, input=input)
            prompts.append(prompt)
            idxs.append(i)
        idx_to_response = [None for _ in range(len(self.data))]

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            responses = self.query(prompts, idxs, histories=histories)
            if self.n > 1:
                all_train_examples = self.get_all_examples("train", idxs)
                logger.info(f"Reranking {len(all_train_examples)} train examples...")
                if self.verbose:
                    logger.info(f"Responses before reranking:")
                    for res in responses[:PRINT_NUM]:
                        logger.info(res)
                responses = self.get_best_responses(idxs, all_train_examples, responses)
            for idx, response in zip(idxs, responses):
                idx_to_response[idx] = response

            rules = [self.extract_rules(response) for response in responses]
            self.add_rules(idxs, rules)

            if self.max_iter > 1:
                self.add_histories("user", idxs, prompts)
                self.add_histories("assistant", idxs, responses)

                all_train_outputs = [self.extract_output_prediction(response) for response in responses]
                all_outputs = self.get_all_examples("output", idxs)
                prompts = []
                new_idxs = []
                for idx, rule, code, input, train_outputs, outputs in zip(
                        idxs, rules, all_codes, all_inputs, all_train_outputs, all_outputs
                ):
                    prompt = self.get_feedback(code, input, rule, train_outputs, outputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_test_from_rule(idx_to_response)
            self.metrics.append(metrics)
