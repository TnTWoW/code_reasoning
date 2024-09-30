import collections
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from typing import Any

import utils.deepcoder_dsl as deepcoder_dsl

DatasetElement = collections.namedtuple(
		'DatasetElement',
		['inputs', 'outputs', 'dsl_program', 'python_program'])

_DEEPCODER_FUNCTION_IMPLS = [
    '''
def Map(f, xs):
  return [f(x) for x in xs]
''',
    '''
def Filter(f, xs):
  return [x for x in xs if f(x)]
''',
    '''
def Count(f, xs):
  return len([x for x in xs if f(x)])
''',
    '''
def ZipWith(f, xs, ys):
  return [f(x, y) for (x, y) in zip(xs, ys)]
''',
    '''
def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys
''',
]

_DEEPCODER_LAMBDA_IMPLS = '''
PLUS_ONE = lambda x: x + 1
MINUS_ONE = lambda x: x - 1
TIMES_TWO = lambda x: x * 2
DIV_TWO = lambda x: x // 2
NEGATE = lambda x: -x
SQUARE = lambda x: x ** 2
TIMES_THREE = lambda x: x * 3
DIV_THREE = lambda x: x // 3
TIMES_FOUR = lambda x: x * 4
DIV_FOUR = lambda x: x // 4
IS_POSITIVE = lambda x: x > 0
IS_NEGATIVE = lambda x: x < 0
IS_EVEN = lambda x: x % 2 == 0
IS_ODD = lambda x: x % 2 == 1
ADD = lambda x, y: x + y
SUBTRACT = lambda x, y: x - y
MULTIPLY = lambda x, y: x * y
MIN = lambda x, y: min(x, y)
MAX = lambda x, y: max(x, y)
'''.strip()

def get_num_examples(inputs: list[str] | dict[str, Any],
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

def dsl_description(dataset_type: str, version: int) -> str:
  """Gets a description of the DSL for prompting."""
  if dataset_type == 'deepcoder':
    dsl_purpose = 'manipulating lists of integers'
    if version == 1:
      function_details = ', '.join(
          [op.token for op in deepcoder_dsl.OPERATIONS])
      constant_details = ', '.join(
          [lambda_.name for lambda_ in deepcoder_dsl.LAMBDAS])
    elif version == 2:
      function_details = '\n\n'.join(
          [i.strip() for i in _DEEPCODER_FUNCTION_IMPLS])
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 3 or version == 5:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 4:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = None
    else:
      raise ValueError(f'Unhandled version: {version}')
    
  return (
      f'The `dsl` module is a custom library for {dsl_purpose}. It contains '
      'the following functions:\n\n'
      f'{function_details}\n\n'
      + (
          'Additionally, the module defines the following constants:\n\n'
          f'{constant_details}\n\n'
          if constant_details else '') +
      'Below are example programming problems using the `dsl` module, with'
      ' input-output test cases illustrating their behavior.\n\nImportant:'
      ' All programs begin with ```python and end with ``` alone.\n\n'
  )

def get_prompt_prefix(dataset_element: DatasetElement,
                      dataset_type: str) -> str:
  """Gets a prefix of the prompt describing one dataset element."""
  s = '[BEGIN PROBLEM]\n'
  s += 'Input-output test cases:\n'
  for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
    s += f'  Case {i + 1}. '
    if dataset_type == 'deepcoder':
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
      s += f' --> {dataset_element.outputs[i]}\n'
    elif dataset_type == 'robustfill':
      s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  s += '\nProgram:\n```python\n'
  return s


def get_prompt_suffix(dataset_element: DatasetElement) -> str:
  return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'


def get_prompt(dataset_element: DatasetElement, dataset_type: str) -> str:
  return (get_prompt_prefix(dataset_element, dataset_type)
          + get_prompt_suffix(dataset_element))

def few_shot_prompt(few_shot_examples: list[DatasetElement],
                    test_problem: DatasetElement,
                    dataset_type: str,
                    version: int=1) -> str:
  prompt_parts = [dsl_description(dataset_type, version=version)]
  prompt_parts.extend(get_prompt(d, dataset_type) for d in few_shot_examples)
  prompt_parts.append(get_prompt_prefix(test_problem, dataset_type))
  return '\n'.join(prompt_parts)