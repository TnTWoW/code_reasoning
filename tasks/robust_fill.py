from prompts.arc import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    python_rule_prompt,
    rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_to_python_prompt,
    rule_with_feedback_prompt,
)
from prompts.robustfill import few_shot_prompt
from tasks.base import Task
from utils.format_utils import str_to_list
from utils.query_utils import CLAUDE_MODELS
from utils.query_utils import get_cost, query_batch_struct
from utils.format_utils import unflatten

import logging
import types
import copy
import collections
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

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

class RobustFill(Task):
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

    def get_rule(self, response):
        result = []
        json_response = eval(response)
        for i, step in enumerate(json_response['steps'], start=1):
            subrule = step['Subrule']
            input_data = step['input']
            output_data = step['output']
            result.append(f"Step {i}: {subrule} Input: {input_data}, Output: {output_data}")

        # Append the rule at the end
        result.append(f"Rule: {json_response['rule']}")

        # Join the list into a single string
        final_output = '. '.join(result)
        return final_output

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
        train_examples = []
        for input, output in zip(data[0].inputs, data[0].outputs):
            train_examples.append({"input": input, "output": output})
        
        return train_examples

    def get_test_examples(self, data):
        test_examples = []
        for input, output in zip(data[2].inputs, data[2].outputs):
            test_examples.append({"input": input, "output": output})
        return test_examples
    
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
            prompts.append(few_shot_prompt(few_shot_examples, train_set))

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

            # rules = [self.get_rule(response) for response in responses]
            rules = [response for response in responses]
            
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

    def eval_io(self):
        all_train_examples = self.get_all_examples("train")
        all_test_examples = self.get_all_examples("test")
        prompts = []
        idxs = []
        for idx, (train_examples, test_examples) in enumerate(
            zip(all_train_examples, all_test_examples)
        ):
            train_examples = self.format_examples(train_examples)
            for test_example in test_examples:
                test_input = self.format_input(test_example["input"])
                prompts.append(
                    self.io_prompt.format(
                        examples=train_examples, test_input=test_input
                    )
                )
                idxs.append(idx)
        responses = self.query(prompts, idxs, histories=None)
        responses = self.get_best_outputs(responses)
        responses = [self.extract_prediction(r) for r in responses]
        test_outputs = unflatten(responses, all_test_examples)
        metrics = self.get_metrics("test", test_outputs)
        self.metrics.append(metrics)

    def apply_all_rules(self, idxs, all_rules, all_examples, program_name: str = 'program'):

        call_code = f'{program_name}(x)'
        namespace = get_namespace('robustfill')

        all_outputs = []

        for program, inputs in zip(all_rules, all_examples):
            program_code = extract_program(program)
            try:
                exec(program_code, namespace)  # pylint: disable=exec-used
            except Exception as e:  # pylint: disable=bare-except
                print(f"An error occurred:{e}")
            
            outputs = []
            for input in inputs:
                namespace_copy = namespace.copy()
                # Assign the argument values.
                namespace_copy['x'] = input['input']
                # Call the solution function.
                try:
                    output = eval(call_code, namespace_copy)  # pylint: disable=eval-used
                except:  # pylint: disable=bare-except
                    output = None
                
                outputs.append(output)

            all_outputs.append(outputs)
        return all_outputs
    
def get_namespace(dataset_type: str) -> dict[str, Any]:
    """Gets a namespace with the dsl loaded."""
    dsl_object = types.SimpleNamespace()
    for function_name in ROBUSTFILL_FUNCTIONS:
        if function_name == 'Const':
            op_class = robustfill_dsl.ConstStr
            wrapper = lambda c, op_class=op_class: op_class(c)(None)
        else:
            op_class = getattr(robustfill_dsl, function_name)
            wrapper = lambda x, *args, op_class=op_class: op_class(*args)(x)
        setattr(dsl_object, function_name, wrapper)
    for enum_class in ROBUSTFILL_ENUMS:
        setattr(dsl_object, enum_class.__name__, enum_class)
    return {'dsl': dsl_object}


def get_num_examples(inputs: Union[list[str], dict[str, Any]],
                     dataset_type: str) -> int:
  """Returns the number of examples in the inputs."""
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    inputs_dict = inputs
    num_examples = len(list(inputs_dict.values())[0])
    assert all(len(v) == num_examples for v in inputs_dict.values())
  elif dataset_type == 'robustfill':
    assert isinstance(inputs, list), f'RobustFill inputs: {inputs}'
    num_examples = len(inputs)
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

