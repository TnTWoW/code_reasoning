import collections
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from typing import Any
import utils.dsl as robustfill_dsl

DatasetElement = collections.namedtuple(
		'DatasetElement',
		['inputs', 'outputs', 'dsl_program', 'python_program'])

ROBUSTFILL_FUNCTIONS = [
		'Const', 'SubStr', 'GetSpan', 'GetToken', 'ToCase', 'Replace', 'Trim',
		'GetUpto', 'GetFrom', 'GetFirst', 'GetAll', 'Substitute', 'SubstituteAll',
		'Remove', 'RemoveAll',
]

ROBUSTFILL_ENUMS = [
		robustfill_dsl.Type, robustfill_dsl.Case, robustfill_dsl.Boundary,
]

def dsl_description() -> str:
	"""Returns the description for the DSL module."""
	dsl_purpose = 'manipulating strings'
	function_details = ', '.join(ROBUSTFILL_FUNCTIONS)
	constant_details = ', '.join(
		[robustfill_dsl.to_python(obj)	# pylint: disable=g-complex-comprehension
			for e in ROBUSTFILL_ENUMS for obj in e])
	return (
		f'The `dsl` module is a custom library for {dsl_purpose}. It contains '
		'the following functions:\n\n'
		f'{function_details}\n\n'
		+ (
			'Additionally, the module defines the following constants:\n\n'
			f'{constant_details}\n\n'
			if constant_details else '') +
		'Below are example programming problems using the `dsl` module, with'
		' input-output test cases illustrating their behavior to help you understand.\n\nImportant:'
		' All programs begin with ```python and end with ``` alone.\n\n'
	)

def get_num_examples(inputs: list[str] | dict[str, Any]) -> int:
	"""Returns the number of examples in the inputs."""
	assert isinstance(inputs, list), f'RobustFill inputs: {inputs}'
	num_examples = len(inputs)
	return num_examples

def get_prompt_prefix(dataset_element: DatasetElement) -> str:
	"""Gets a prefix of the prompt describing one dataset element."""
	s = 'Generate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\n'
	s += 'Input-output test cases:\n'
	for i in range(get_num_examples(dataset_element.inputs)):
		s += f'	Case {i + 1}. '
		s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
	s += '\nProgram:\n```python\n'
	return s

def get_test_prompt(dataset_element: DatasetElement) -> str:
	"""Gets a prefix of the prompt describing one dataset element."""
	s = 'Below are some example help you understand the `dsl` module. Now your task is to generate the dsl program that maps the following inputs to their corresponding outputs First, analysis the program step by step, then write the corresponding code  [BEGIN PROBLEM]\n'
	s += 'Input-output test cases:\n'
	for i in range(get_num_examples(dataset_element.inputs)):
		s += f'	Case {i + 1}. '
		s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
	s += '\nAnalysis: <Your analysis>\nProgram:\n```python\n# Your code here'
	return s

def get_prompt_suffix(dataset_element: DatasetElement) -> str:
	return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'

def get_prompt(dataset_element: DatasetElement) -> str:
	return (get_prompt_prefix(dataset_element)
					+ get_prompt_suffix(dataset_element))

def few_shot_prompt(few_shot_examples: list[DatasetElement],
										test_problem: DatasetElement,
										) -> str:
	prompt_parts = [dsl_description()]
	prompt_parts.extend(get_prompt(d) for d in few_shot_examples)
	prompt_parts.append(get_test_prompt(test_problem))
	return '\n'.join(prompt_parts)
