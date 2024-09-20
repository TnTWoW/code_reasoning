import logging
import re
import numpy as np

from prompts.cruxeval import (
    input_prompt,
    cot_input_prompt,
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

class CruxEvalInput(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = input_prompt


class CruxEvalOutput(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = output_prompt
        self.cot_prompt = cot_output_prompt
        self.coc_prompt = coc_output_prompt
        self.rule_prompt = rule_output_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_output_prompt

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

    def extract_output_prediction(self, text):
        pattern = r'==\s*(.*?)\s*```'
        results = re.findall(pattern, text)
        if not results:
            return text
        return results[-1]
    def apply_all_rules(self, idxs, all_rules, all_examples):
        pass
    def get_metrics(self, answers: list) -> dict:
        all_outputs = self.get_all_examples("output")
        acc = np.mean([eval(a) == eval(o) for a, o in zip(answers, all_outputs)])
        output_dict = {
            "test_instance_acc": acc,
            "test_acc": acc,
        }
        return output_dict
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
        responses = self.get_best_outputs(responses)
        responses = [self.extract_output_prediction(r) for r in responses]
        metrics = self.get_metrics(responses)
        self.metrics.append(metrics)

