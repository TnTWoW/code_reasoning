import logging
from utils.python_utils import extract_function_names
from tqdm import tqdm
import copy
import ast
from utils.python_utils import execute_function

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
    input_feedback_prompt,
    output_feedback_prompt,
)
from tasks.iobase import IOBase

logger = logging.getLogger(__name__)
PRINT_NUM = 3

class LiveCodeBenchInput(IOBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = input_prompt
        self.cot_prompt = cot_input_prompt
        self.coc_prompt = coc_input_prompt
        self.rule_prompt = rule_input_prompt
        self.rule_with_feedback_input_prompt = rule_with_feedback_input_prompt
        self.input_feedback_prompt = input_feedback_prompt
        self.language = "python"

    def get_input_feedback(self, code, output, rule, pred_input, input):
        p_input = copy.deepcopy([pred_input])
        code = "from typing import List\nfrom collections import defaultdict\nfrom numpy import inf\nimport numpy\nimport pandas\n" + code
        pred_outputs = execute_function(code, p_input)[0]
        if self.safe_literal_eval(output) != pred_outputs:
            func_name = extract_function_names(code)[0]
            if self.rule_type == "io":
                return self.input_feedback_prompt.format(code=code, func_name=func_name, output=output, p_input=p_input, language=self.language)
            else:
                return self.rule_with_feedback_input_prompt.format(code=code, func_name=func_name, output=output, rule=rule, p_input=p_input, language=self.language)
        return ""
    def eval_io(self):
        all_codes = self.get_all_examples("code")
        all_outputs = self.get_all_examples("output")

        prompts = []
        idxs = []

        for i, (code, output) in enumerate(zip(all_codes, all_outputs)):
            func_name = extract_function_names(code)[0]
            if self.rule_type == 'io':
                prompt = self.io_prompt.format(code=code, func_name=func_name, output=output, language=self.language)
            elif self.rule_type == 'cot':
                prompt = self.cot_prompt.format(code=code, func_name=func_name, output=output, language=self.language)
            elif self.rule_type == 'coc':
                prompt = self.coc_prompt.format(code=code, func_name=func_name, output=output, language=self.language)
            else:
                raise ValueError(f"Invalid rule type: {self.rule_type}")
            prompts.append(prompt)
            idxs.append(i)

        idx_to_response = [None for _ in range(len(self.data))]
        if self.mode == "generate":
            self.load_model()

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            if self.mode == "generate":
                responses = self.generate(prompts, idxs, histories=histories)
            else:
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
                all_func_names = [extract_function_names(code)[0] for code in all_codes]
                all_train_inputs = [self.extract_input_prediction(response, func_name) for response, func_name in zip(responses, all_func_names)]
                prompts = []
                new_idxs = []
                for idx, rule, code, output, train_inputs, inputs in\
                        tqdm(zip(idxs, rules, all_codes, all_outputs, all_train_inputs, all_inputs),
                             desc="Generating feedback", total=len(idxs)):
                    prompt = self.get_input_feedback(code, output, rule, train_inputs, inputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_input_from_rule(idx_to_response)
            self.metrics.append(metrics)

    def eval_rule(self):
        all_codes = self.get_all_examples("code")
        all_outputs = self.get_all_examples("output")

        prompts = []
        idxs = []
        for i, (code, output) in enumerate(zip(all_codes, all_outputs)):
            func_name = extract_function_names(code)[0]
            prompt = self.rule_prompt.format(code=code, func_name=func_name, output=output, language=self.language)
            prompts.append(prompt)
            idxs.append(i)
        idx_to_response = [None for _ in range(len(self.data))]

        if self.mode == "generate":
            self.load_model()

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            if self.mode == "generate":
                responses = self.generate(prompts, idxs, histories=histories)
            else:
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
                all_func_names = [extract_function_names(code)[0] for code in all_codes]
                all_train_inputs = [self.extract_input_prediction(response, func_name) for response, func_name in zip(responses, all_func_names)]
                prompts = []
                new_idxs = []
                for idx, rule, code, output, train_inputs, inputs in\
                        tqdm(zip(idxs, rules, all_codes, all_outputs, all_train_inputs, all_inputs),
                             desc="Generating feedback", total=len(idxs)):
                    prompt = self.get_input_feedback(code, output, rule, train_inputs, inputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_input_from_rule(idx_to_response)
            self.metrics.append(metrics)
class LiveCodeBenchOutput(IOBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = output_prompt
        self.cot_prompt = cot_output_prompt
        self.coc_prompt = coc_output_prompt
        self.rule_prompt = rule_output_prompt
        self.rule_with_feedback_output_prompt = rule_with_feedback_output_prompt
        self.output_feedback_prompt = output_feedback_prompt
        self.language = "python"

    def get_output_feedback(self, code, input, rule, p_output, output):
        if self.safe_literal_eval(p_output) != ast.literal_eval(output):
            func_name = extract_function_names(code)[0]
            if self.rule_type == "io":
                return self.output_feedback_prompt.format(code=code, func_name=func_name, input=input, p_output=p_output, language=self.language)
            else:
                return self.rule_with_feedback_output_prompt.format(code=code, func_name=func_name, input=input, rule=rule, p_output=p_output, language=self.language)
        return ""
    def eval_io(self):
        all_codes = self.get_all_examples("code")
        all_inputs = self.get_all_examples("input")

        prompts = []
        idxs = []
        for i, (code, input) in enumerate(zip(all_codes, all_inputs)):
            if self.rule_type == 'io':
                prompt = self.io_prompt.format(code=code, input=input, language=self.language)
            elif self.rule_type == 'cot':
                prompt = self.cot_prompt.format(code=code, input=input, language=self.language)
            elif self.rule_type == 'coc':
                prompt = self.coc_prompt.format(code=code, input=input, language=self.language)
            prompts.append(prompt)
            idxs.append(i)

        idx_to_response = [None for _ in range(len(self.data))]
        if self.mode == "generate":
            self.load_model()

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)

            if self.mode == "generate":
                responses = self.generate(prompts, idxs, histories=histories)
            else:
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
                    prompt = self.get_output_feedback(code, input, rule, train_outputs, outputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_output_from_rule(idx_to_response)
            self.metrics.append(metrics)

    def eval_rule(self):
        all_codes = self.get_all_examples("code")
        all_inputs = self.get_all_examples("input")

        prompts = []
        idxs = []
        for i, (code, input) in enumerate(zip(all_codes, all_inputs)):
            prompt = self.rule_prompt.format(code=code, input=input, language=self.language)
            prompts.append(prompt)
            idxs.append(i)
        idx_to_response = [None for _ in range(len(self.data))]

        if self.mode == "generate":
            self.load_model()

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            
            if self.mode == "generate":
                responses = self.generate(prompts, idxs, histories=histories)
            else:
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
                    prompt = self.get_output_feedback(code, input, rule, train_outputs, outputs)
                    if prompt == "":
                        continue
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_output_from_rule(idx_to_response)
            self.metrics.append(metrics)
