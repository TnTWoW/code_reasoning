import copy
import logging
import re
import numpy as np
import ast
from tasks.base import Task
from utils.python_utils import execute_function, extract_function_names
from tqdm import tqdm


logger = logging.getLogger(__name__)
PRINT_NUM = 3

class IOBase(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_code_examples(self, data):
        return data["code"][: self.n_train]
    def get_input_examples(self, data):
        return data["input"][: self.n_train]
    def get_output_examples(self, data):
        return data["output"][: self.n_train]
    def safe_literal_eval(self, value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

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

    def get_input_metrics(self, codes: list, pred_inputs: list, all_outputs: list) -> dict:
        all_pred_outputs = []
        for code, pred_input, outputs in\
                tqdm(zip(codes, pred_inputs, all_outputs),
                     desc="Evaluating results", total=len(codes)):
            code = "from typing import List\nfrom collections import defaultdict\nfrom numpy import inf\nimport numpy\nimport pandas\n" + code
            input = copy.deepcopy([pred_input])
            pred_outputs = execute_function(code, input)
            all_pred_outputs.append(pred_outputs[0])
        accs = [a == self.safe_literal_eval(o) for a, o in zip(all_pred_outputs, all_outputs)]
        acc = np.mean(accs)
        output_dict = {
            "test_instance_acc": acc,
            "test_acc": acc,
            "test_accs": accs,
        }
        return output_dict

    def extract_input_prediction(self, text, func_name):
        pattern = fr'assert\s+{func_name}\((.*?)\)'
        results = re.findall(pattern, text)
        if not results:
            return ""
        return results[-1]

    def get_best_inputs(self, responses, codes):
        func_names = [extract_function_names(code)[0] for code in codes]
        if self.n == 1:
            responses = [self.extract_input_prediction(r, fn) for r, fn in zip(responses, func_names)]
            return responses
        responses = [
            [str(self.extract_input_prediction(r, fn)) for r in res] for res, fn in zip(responses, func_names)
        ]
        best_outputs = [max(set(res), key=res.count) for res in responses]
        return best_outputs

    def eval_input_from_rule(self, responses):
        codes = self.get_all_examples("code")
        outputs = self.get_all_examples("output")
        func_names = [extract_function_names(code)[0] for code in codes]
        assert all([response is not None for response in responses])
        answers = [self.extract_input_prediction(response, func_name) for response, func_name in zip(responses, func_names)]
        output_dict = self.get_input_metrics(codes, answers, outputs)
        return output_dict

    def extract_output_prediction(self, text):
        pattern = r'==\s*(.*?)\s*```'
        results = re.findall(pattern, text)
        if not results:
            return ""
        return results[-1]
    def get_output_metrics(self, answers: list) -> dict:
        all_outputs = self.get_all_examples("output")
        accs = [self.safe_literal_eval(a) == ast.literal_eval(o) for a, o in zip(answers, all_outputs)]
        acc = np.mean(accs)
        output_dict = {
            "test_instance_acc": acc,
            "test_acc": acc,
            "test_accs": accs,
        }
        return output_dict

    def eval_output_from_rule(self, responses):
        assert all([response is not None for response in responses])
        answers = [self.extract_output_prediction(response) for response in responses]
        output_dict = self.get_output_metrics(answers)
        return output_dict

    def get_best_outputs(self, responses):
        if self.n == 1:
            responses = [self.extract_output_prediction(r) for r in responses]
            return responses
        responses = [
            [str(self.extract_output_prediction(r)) for r in res] for res in responses
        ]
        best_outputs = [max(set(res), key=res.count) for res in responses]
        return best_outputs