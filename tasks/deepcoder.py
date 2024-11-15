import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from prompts.arc import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    python_rule_prompt,
    rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_with_feedback_prompt,
)
from prompts.deepcoder import few_shot_prompt, fewshot_coc_prompt, few_shot_rule_prompt, rule_to_python_prompt
from tasks.base import PythonTask
from utils.format_utils import str_to_list
from utils.query_utils import CLAUDE_MODELS
import utils.deepcoder_dsl as deepcoder_dsl 
from utils.query_utils import get_cost, query_batch_struct
from utils.format_utils import flatten, unflatten

import tqdm

import numpy as np

import logging
import types
import copy
import collections

from typing import Any, Union
import utils.dsl as robustfill_dsl
ROBUSTFILL_ENUMS = [
    robustfill_dsl.Type, robustfill_dsl.Case, robustfill_dsl.Boundary,
]

# `inputs` is a dict for DeepCoder (name to list of values for each example), or
# a list for RobustFill (values for each example).
DatasetElement = collections.namedtuple(
    'DatasetElement',
    ['inputs', 'outputs', 'dsl_program', 'python_program'])

ROBUSTFILL_FUNCTIONS = [
		'Const', 'SubStr', 'GetSpan', 'GetToken', 'ToCase', 'Replace', 'Trim',
		'GetUpto', 'GetFrom', 'GetFirst', 'GetAll', 'Substitute', 'SubstituteAll',
		'Remove', 'RemoveAll',
]


logger = logging.getLogger(__name__)

PRINT_NUM = 3
    
CACHE_FILE = "query_cache.pkl"
HISTORY_FILE = "history.jsonl"

class DeepCoder(PythonTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_name in CLAUDE_MODELS:
            self.io_prompt = io_prompt_with_format
            self.rule_to_output_prompt = rule_to_output_prompt_with_format
        else:
            self.io_prompt = io_prompt
            self.rule_to_output_prompt = rule_to_output_prompt

        if self.rule_type == "python":
            self.rule_prompt = python_rule_prompt
        else:
            self.rule_prompt = rule_prompt
        self.example_prompt = example_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_prompt
        self.feedback_prompt = feedback_prompt
        self.rule_to_python_prompt = rule_to_python_prompt

    def eval_test_from_rule(self, responses):
        assert all([response is not None for response in responses])
        rules = [response for response in responses]
        all_test_examples = self.get_all_examples("test")
        idxs = list(range(len(all_test_examples)))
        logger.info(f"Applying rules to {len(all_test_examples)} test examples...")
        all_test_outputs = self.apply_all_rules(idxs, rules, all_test_examples)
        output_dict = self.get_metrics("test", all_test_outputs)
        return output_dict

    # def get_rule(self, response):
    #     result = []
    #     json_response = eval(response)
    #     for i, step in enumerate(json_response['steps'], start=1):
    #         subrule = step['Subrule']
    #         input_data = step['input']
    #         output_data = step['output']
    #         result.append(f"Step {i}: {subrule} Input: {input_data}, Output: {output_data}")
    #
    #     # Append the rule at the end
    #     result.append(f"Rule: {json_response['rule']}")
    #
    #     # Join the list into a single string
    #     final_output = '. '.join(result)
    #     return final_output

    def canonicalize(self, grid):
        try:
            if isinstance(grid, list) and isinstance(grid[0], list):
                return [[int(x) if x is not None else 0 for x in row] for row in grid]
            elif isinstance(grid, list):
                return [int(x) for x in grid]
            elif isinstance(grid, str):
                return str_to_list(grid)
            return grid
        except Exception:
            return grid
        
    def struct_query(self, prompts, idxs, n=None, temperature=None, histories=None):
        n = self.n if n is None else n
        temperature = self.temperature if temperature is None else temperature
        responses = query_batch_struct(
            prompts,
            self.model_name,
            system_msg=self.system_msg,
            cache_file=self.cache_file,
            history_file=self.history_file,
            histories=histories,
            n=n,
            temperature=temperature,
        )
        prompt2key = lambda p, h: (
            p,
            self.model_name,
            self.system_msg,
            tuple([tuple(e.items()) for e in h]) if h is not None else None,
            temperature,
            n,
        )
        assert len(idxs) == len(prompts) == len(responses)
        for i, (idx, prompt, response) in enumerate(zip(idxs, prompts, responses)):
            history = histories[i] if histories is not None else None
            key = prompt2key(prompt, history)
            if key not in self.cache:
                self.cache[key] = response
                cost = get_cost(
                    prompt, response, model_name=self.model_name, history=history
                )
                self.cost += cost
            self.interactions[idx].append(
                {
                    "query": prompt,
                    "response": response,
                    "history": copy.deepcopy(history),
                    "n": n,
                    "temperature": temperature,
                    "system_msg": self.system_msg,
                }
            )
        return responses
    
    def get_train_examples(self, data):

        input_data, output_data = data[0].inputs, data[0].outputs 

        return({"input": input_data, "output": output_data})
        

    def get_test_examples(self, data):

        input_data, output_data = data[2].inputs, data[2].outputs 

        return({"input": input_data, "output": output_data})
    
    def get_all_examples(self, split, idxs=None):
        if idxs is None:
            idxs = list(range(len(self.data)))
        if split == "train":
            return [self.get_train_examples(self.data[i]) for i in idxs]
        elif split == "test":
            return [self.get_test_examples(self.data[i]) for i in idxs]
        else:
            raise ValueError(f"Invalid split: {split}")

    def eval_rule(self):
        prompts = []
        num_train = len(self.data)
        for train_index in range(num_train):
            train_set, few_shot_examples, test_set = self.data[train_index]
            # prompts.append(few_shot_prompt(few_shot_examples, train_set, 'deepcoder'))
            prompts.append(few_shot_rule_prompt(few_shot_examples, train_set, 'deepcoder'))
            # prompts.append(fewshot_coc_prompt(train_set))


        idxs = list(range(len(self.data)))
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
                # responses = self.struct_query(prompts, idxs, histories=histories)
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

            rules = [self.get_rule(response) for response in responses]
            # rules = [response for response in responses]
            
            self.add_rules(idxs, rules)

            if self.eval_every > 0 and i % self.eval_every == 0:
                metrics = self.eval_test_from_rule(idx_to_response)
                self.metrics.append(metrics)

            if self.max_iter > 1:
                all_train_examples = self.get_all_examples("train", idxs)
                logger.info(
                    f"Applying rules to {len(all_train_examples)} train examples for feedback..."
                )
                if self.verbose:
                    logger.info(f"Rules:")
                    for rule in rules[:3]:
                        logger.info(rule)
                all_train_outputs = self.apply_all_rules(
                    idxs, rules, all_train_examples
                )
                self.add_histories("user", idxs, prompts)
                self.add_histories("assistant", idxs, responses)

                prompts = []
                new_idxs = []
                for idx, rule, train_examples, train_outputs in zip(
                    idxs, rules, all_train_examples, all_train_outputs
                ):
                    feedback = self.get_feedback(train_examples, train_outputs)
                    rule = self.format_rule(rule)
                    if self.verbose:
                        logger.info(f"Feedback:")
                        logger.info(feedback)
                    if feedback == "":
                        continue
                    train_examples = self.format_examples(train_examples)
                    prompt = self.rule_with_feedback_prompt.format(
                        examples=train_examples, rule=rule, feedback=feedback
                    )
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_test_from_rule(idx_to_response)
            self.metrics.append(metrics)

    def get_best_responses(self, all_idxs, all_examples, all_responses):
        # all_idxs: [idx1, idx2]
        # all_examples: [[ex1, ex2], [ex1, ex2]]
        # responses [[rule1_for_examples1, rule2_for_examples1], [rule1_for_examples2, rule2_for_examples2]]
        assert len(all_examples) == len(all_responses) == len(all_idxs)
        # all_examples_flatten: [examples1, examples1, examples2, examples2]
        all_examples_flatten, all_responses_flatten = flatten(
            all_examples, all_responses
        )
        all_idxs_flatten, _ = flatten(all_idxs, all_responses)
        all_rules_flatten = [
            response for response in all_responses_flatten
        ]
        all_outputs_flatten = self.apply_all_rules(
            all_idxs_flatten,
            all_rules_flatten,
            all_examples_flatten,
        )
        accs = [
            self.eval_one(examples, outputs)[0]
            for examples, outputs in zip(all_examples_flatten, all_outputs_flatten)
        ]
        accs = unflatten(accs, all_responses)
        best_idx = [acc.index(max(acc)) for acc in accs]
        best_responses = [
            responses[idx] for idx, responses in zip(best_idx, all_responses)
        ]
        if self.verbose:
            for best_idx, best_response, acc in zip(best_idx, best_responses, accs):
                logger.info(f"Best rule: {best_response}, acc: {acc[best_idx]}")
        return best_responses

    def flatten(self, list1, list2):
        """Flatten and match a list and a nested list.

        Args:
            list1: [ex1, ex2, ex3, ...]
            list2: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]

        Returns:
            flatten_list1: [ex1, ex1, ex2, ex2, ...]
            flatten_list2: [r1_ex1, r2_ex1, r1_ex2, r2_ex2, ...]
        """
        assert len(list1) == len(list2)
        flatten_list1 = []
        flatten_list2 = []
        for ex, nested_ex in zip(list1, list2):
            for item in nested_ex:
                flatten_list1.append(ex)
                flatten_list2.append(item)
        return flatten_list1, flatten_list2

    def unflatten(self, flatten_list3, list2):
        """Unflatten a flatten list to match a nested list.

        Args:
            flatten_list3: [r1_ex1, r2_ex1, r1_ex2, r2_ex2, ...]
            list2: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]

        Returns:
            nested_list3: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]
        """
        list3 = []
        index = 0
        for inner_list in list2:
            length = len(inner_list)
            list3.append(flatten_list3[index : index + length])
            index += length
        return list3
    
    # def apply_all_rules(self, idxs, all_rules, all_examples, program_name: str = 'program'):
    #     # if self.interpreter_type == "lm":
    #     #     return self.apply_all_rules_with_lm(idxs, all_rules, all_examples)
    #     # if self.rule_type != "python":
    #     #     all_rules = self.rules_to_programs(idxs, all_rules)
    #     #     programs = [extract_program(rule) for rule in all_rules]
    #     # else:
    #     all_outputs = []
    #     namespace = get_namespace('deepcoder')
    #
    #     for dsl_program, inputs, i in zip(all_rules, all_examples, range(len(all_rules))):
    #         namespace_copy = namespace.copy()
    #         keys = inputs['input'].keys()
    #         call_code = f'{program_name}({", ".join(keys)})'
    #         program_code = extract_program(dsl_program)
    #         try:
    #             exec(program_code, namespace_copy)  # pylint: disable=exec-used
    #         except Exception as e:  # pylint: disable=bare-except
    #             print(f"An error occurred:{e}")
    #
    #
    #         result = []
    #         for i in range(len(inputs['output'])):
    #             for input_name, input_values in inputs['input'].items():
    #                 namespace_copy[input_name] = input_values[i]
    #                     # Call the solution function.
    #             try:
    #                 output = eval(call_code, namespace_copy)  # pylint: disable=eval-used
    #             except:
    #                 output = None
    #             result.append(output)
    #
    #         all_outputs.append(result)
    #
    #     return all_outputs

    def apply_all_rules(self, idxs, all_rules, all_examples, program_name: str = 'program'):
        if self.interpreter_type == "lm":
            return self.apply_all_rules_with_lm(idxs, all_rules, all_examples)
        if self.rule_type != "python":
            all_rules = self.rules_to_programs(idxs, all_rules)

        programs = [extract_program(rule) for rule in all_rules]
        all_outputs = []
        namespace = get_namespace('deepcoder')

        for dsl_program, inputs, i in zip(programs, all_examples, range(len(all_rules))):
            namespace_copy = namespace.copy()
            keys = inputs['input'].keys()
            call_code = f'{program_name}({", ".join(keys)})'
            program_code = extract_program(dsl_program)
            try:
                exec(program_code, namespace_copy)  # pylint: disable=exec-used
            except Exception as e:  # pylint: disable=bare-except
                print(f"An error occurred:{e}")

            result = []
            for i in range(len(inputs['output'])):
                for input_name, input_values in inputs['input'].items():
                    namespace_copy[input_name] = input_values[i]
                    # Call the solution function.
                try:
                    output = eval(call_code, namespace_copy)  # pylint: disable=eval-used
                except:
                    output = None
                result.append(output)

            all_outputs.append(result)

        return all_outputs
    
    def rules_to_programs(self, idxs, all_rules):
        prompts = [self.rule_to_python_prompt.format(rule=rule) for rule in all_rules]
        if self.mode == "generate":
            responses = self.generate(prompts, idxs, histories=None)
        else:
            responses = self.query(prompts, idxs, n=1, temperature=0, histories=None)
        return responses

    def get_feedback(self, examples, outputs):
        feedbacks, inputs, targets = [], examples['input'], examples['output']
        for pred, i in zip(outputs, range(len(outputs))):
            target = targets[i]

            input = {key: inputs[key][i] for key in inputs}

            pred = self.format_output(pred)

            if pred != target:
                feedback = self.feedback_prompt.format(
                    input=input, target=target, output=pred
                )
                feedbacks.append(feedback)
        return "\n".join(feedbacks)
    
    def format_examples(self, examples):
        example_strs = []

        for i in range(len(examples['output'])):
            input = {key: examples['input'][key][i] for key in examples['input']}
            output = examples['output'][i]
            example_strs.append(self.example_prompt.format(input=input, output=output))

        return "\n".join(example_strs)

    def format_input(self, input):
        return input['input']
    
    def eval_one(self, examples, outputs):
        # examples: [ex1, ex2]
        # outputs: [output1, output2]
        try:
            if type(outputs) == str:
                outputs = eval(outputs)
            outputs = [self.format_output(output) for output in outputs]
        except:
            return 0.0, [0.0 for _ in range(len(examples['output']))]
        targets = [self.format_output(ex) for ex in examples['output']]
        accs = []
        for pred, target in zip(outputs, targets):
            accs.append(float(pred == target))
        acc = np.mean(accs)
        return acc, accs
    
    def eval_io(self):
        all_train_examples = self.get_all_examples("train")
        all_test_examples = self.get_all_examples("test")
        prompts = []
        idxs = []
        for idx, (train_examples, test_examples) in enumerate(
            zip(all_train_examples, all_test_examples)
        ):
            train_examples = self.format_examples(train_examples)
            test_input = self.format_input(test_examples)
            prompts.append(
                self.io_prompt.format(
                    examples=train_examples, test_input=test_input
                )
            )
            idxs.append(idx)
        responses = self.query(prompts, idxs, histories=None)
        responses = self.get_best_outputs(responses)
        responses = [self.extract_prediction(r) for r in responses]
        # test_outputs = unflatten(responses, all_test_examples)
        metrics = self.get_metrics("test", responses)
        self.metrics.append(metrics)

def get_namespace(dataset_type: str) -> dict[str, Any]:
  """Gets a namespace with the dsl loaded."""
  dsl_object = types.SimpleNamespace()
  if dataset_type == 'deepcoder':
    for lambda_ in deepcoder_dsl.LAMBDAS:
      setattr(dsl_object, lambda_.name, lambda_.func)
    for op in deepcoder_dsl.OPERATIONS:
      setattr(dsl_object, op.token, op.func)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return {'dsl': dsl_object}


def get_num_examples(inputs: Union[list[str], dict[str, Any]],
                     dataset_type: str) -> int:
  """Returns the number of examples in the inputs."""
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    inputs_dict = inputs
    num_examples = len(list(inputs_dict.values())[0])
    assert all(len(v) == num_examples for v in inputs_dict.values())
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return num_examples

def extract_program(response):
    pattern = r"```python\s*([\s\S]+?)\s*```"
    matches = re.findall(pattern, response)
    matches = [match for match in matches if "def" in match]
    if matches:
        return "\n".join(matches)
    return response
