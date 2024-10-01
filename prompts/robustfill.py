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

coc_prompt = """
The `dsl` module is a custom library for manipulating strings. It contains the following functions:
\nConst, SubStr, GetSpan, GetToken, ToCase, Replace, Trim, GetUpto, GetFrom, GetFirst, GetAll, Substitute, SubstituteAll, Remove, RemoveAll
\nAdditionally, the module defines the following constants:
\ndsl.Type.NUMBER, dsl.Type.WORD, dsl.Type.ALPHANUM, dsl.Type.ALL_CAPS, dsl.Type.PROP_CASE, dsl.Type.LOWER, dsl.Type.DIGIT, dsl.Type.CHAR, dsl.Case.PROPER, dsl.Case.ALL_CAPS, dsl.Case.LOWER, dsl.Boundary.START, dsl.Boundary.END
\nBelow are example programming problems using the `dsl` module, with step by step state input-output test cases illustrating their behavior to help you understand.
Task description: Map the given inputs to their corresponding outputs.

Q:
Input:  "[%0600)1801??;97?/63"
Output: "600)1801??;97?/63.[%0600)1801??;97?/f"

Input: "?[]1@5]584?,610?"
Output: "@5]584?,610?.?[]1@5]584?,f?"

A:
# TRACE START
state: {{}}
explanation: Python execution.
delta state: {{'Input': "[%0600)1801??;97?/63"}, {'Input': "?[]1@5]584?,610?"}
line: dsl.GetFrom(x, dsl.Type.DIGIT)
explanation: Python execution.
delta state: {{'Input': "[%0600)1801??;97?/63", 'Output': "600)1801??;97?/63"}, {'Input': "?[]1@5]584?,610?", 'Output': "@5]584?,610?"}}
line: dsl.Const(\'.\')
explanation: Python execution.
delta state: {{'Input': "600)1801??;97?/63", 'Output': "600)1801??;97?/63."}, 'Input': "@5]584?,610?", 'Output': "@5]584?,610?."}}
line: dsl.Substitute(x, dsl.Type.NUMBER, -1, \'f\')
explanation: Python execution.
delta state: {{'Input': "600)1801??;97?/63.", 'Output': "600)1801??;97?/63.[%0600)1801??;97?/f"}, {'Input': "@5]584?,610?.", 'Output': "@5]584?,610?.?[]1@5]584?,f?"}}
# TRACE END
```python\ndef program(x):
  parts = [
    dsl.GetFrom(x, dsl.Type.DIGIT),
    dsl.Const(\'.\'),
    dsl.Substitute(x, dsl.Type.NUMBER, -1, \'f\'),
  ]
  return \'\'.join(parts)
```

Q:
Input: "[:15D.e5o7[1&45Da[["
Output: ":15D.e5o7[1&5Da"

Input:"[.nVNO[]kf\'9911[:In"
Output:".nVNO[]kf\'1[:"

A:
# TRACE START
state: {{}}
explanation: Python execution.
delta state: {{'Input': "[:15D.e5o7[1&45Da[["},{'Input': "[.nVNO[]kf\'9911[:In"}}
line: dsl.GetSpan(x, dsl.Type.CHAR, 1, dsl.Boundary.END, dsl.Type.NUMBER, -1, dsl.Boundary.START)
explanation: Python execution.
delta state: {{'Input': "[:15D.e5o7[1&45Da[[", 'Output': ":15D.e5o7[1&"}, {'Input': "[.nVNO[]kf\'9911[:In", 'Output': ".nVNO[]kf\'"}}
line: dsl.SubStr(x, -5, -3)
explanation: Python execution.
delta state: {{'Input': ":15D.e5o7[1&", 'Output': ":15D.e5o7[1&5Da"},{'Input': ".nVNO[]kf\'", 'Output': ".nVNO[]kf\'1[:"}}
# TRACE END
```python\ndef program(x):
  parts = [
      dsl.GetSpan(x, dsl.Type.CHAR, 1, dsl.Boundary.END, dsl.Type.NUMBER, -1, dsl.Boundary.START),
      dsl.SubStr(x, -5, -3),
  ]
  return \'\'.join(parts)
```

"""
rule_prompt = """Generate a rule that maps the following inputs to their corresponding outputs.

{examples}

Please format your rule as follows:

Rule: <Your rule>"""

rule_to_python_prompt = """
The `dsl` module is a custom library for manipulating strings. It contains the following functions:
\nConst, SubStr, GetSpan, GetToken, ToCase, Replace, Trim, GetUpto, GetFrom, GetFirst, GetAll, Substitute, SubstituteAll, Remove, RemoveAll
\nAdditionally, the module defines the following constants:
\ndsl.Type.NUMBER, dsl.Type.WORD, dsl.Type.ALPHANUM, dsl.Type.ALL_CAPS, dsl.Type.PROP_CASE, dsl.Type.LOWER, dsl.Type.DIGIT, dsl.Type.CHAR, dsl.Case.PROPER, dsl.Case.ALL_CAPS, dsl.Case.LOWER, dsl.Boundary.START, dsl.Boundary.END
\nBelow are example programming problems using the `dsl` module, with input-output test cases illustrating their behavior to help you understand.
\nImportant: All programs begin with ```python and end with ``` alone.

\nGenerate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\nInput-output test cases:
\tCase 1. ",/a@eya\'Md.!ov." --> "aeyadov"
\tCase 2. "&gPnb(py,]r8.&J4JZ." --> "gnbpyr"
\tCase 3. ",\'gw6q..pfl"G"otn." --> "gwqpflotn"
\tCase 4. "."uaR@H..gjih,[q" --> "uagjihq"
\nProgram:
```python\ndef program(x):
  parts = [
      dsl.GetAll(dsl.Trim(x), dsl.Type.LOWER),
  ]
  return \'\'.join(parts)
```
[END PROBLEM]

\nGenerate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\nInput-output test cases:
\tCase 1. ""4##]#))" --> "4"
\tCase 2. ")###)]{143" --> "143"
\tCase 3. "])#)##}7" --> "7"
\tCase 4. "))]###&61" --> "61"
\nProgram:
```python\ndef program(x):
  parts = [
      dsl.GetAll(dsl.GetSpan(x, dsl.Type.NUMBER, -1, dsl.Boundary.START, dsl.Type.NUMBER, 1, dsl.Boundary.END), dsl.Type.CHAR),
  ]
  return \'\'.join(parts)
```
[END PROBLEM]

\nGenerate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\nInput-output test cases:
\tCase 1. "[%0600)1801??;97?/63" --> "600)1801??;97?/63.[%0600)1801??;97?/f"
\tCase 2. "?[]1@5]584?,610?" --> "@5]584?,610?.?[]1@5]584?,f?"
\tCase 3. "[#53&2\'6?\'7550??" --> "3&2\'6?\'7550??.[#53&2\'6?\'f??"
\tCase 4. "[???)4?2862!614}343" --> "?2862!614}343.[???)4?2862!614}f"
\nProgram:
```python\ndef program(x):
  parts = [
      dsl.GetFrom(x, dsl.Type.DIGIT),
      dsl.Const(\'.\'),
      dsl.Substitute(x, dsl.Type.NUMBER, -1, \'f\'),
  ]
  return \'\'.join(parts)
```
[END PROBLEM]

\nGenerate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\nInput-output test cases:
\tCase 1. "[:15D.e5o7[1&45Da[[" --> ":15D.e5o7[1&5Da"
\tCase 2. "[.nVNO[]kf\'9911[:In" --> ".nVNO[]kf\'1[:"
\tCase 3. "[!r9Yw[[;145$Tz)DOZi" --> "!r9Yw[[;)DO"
\tCase 4. "/dtI[!Q:Dxy[[9[" --> "dtI[!Q:Dxy[[y[["
\nProgram:
```python\ndef program(x):
  parts = [
      dsl.GetSpan(x, dsl.Type.CHAR, 1, dsl.Boundary.END, dsl.Type.NUMBER, -1, dsl.Boundary.START),
      dsl.SubStr(x, -5, -3),
  ]
  return \'\'.join(parts)
```
[END PROBLEM]
Below are some examples to help you understand the `dsl` module. Now your task is to generate the dsl program that maps the following inputs to their corresponding outputs.  Write a Python function `program` for the following rule. the input and output are both string.
"""

def fewshot_coc_prompt(test_problem: DatasetElement) -> str:
	"""Returns the prompt for the CoC problem."""
	s = coc_prompt + '\n\nQ:'
	for i in range(get_num_examples(test_problem.inputs)):
		s += f'"Input: {test_problem.inputs[i]}" \nOutput: "{test_problem.outputs[i]}"\n'
	s += '\nA:\n'
	return s

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
def rule_dsl_description() -> str:
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
			if constant_details else '') + ''
	)
def get_num_examples(inputs: list[str] | dict[str, Any]) -> int:
	"""Returns the number of examples in the inputs."""
	assert isinstance(inputs, list), f'RobustFill inputs: {inputs}'
	num_examples = len(inputs)
	return num_examples

def get_prompt_prefix(dataset_element: DatasetElement) -> str:
	"""Gets a prefix of the prompt describing one dataset element."""
	# s = 'Generate the dsl program that maps the following inputs to their corresponding outputs. [BEGIN PROBLEM]\n'
	s = '. [BEGIN PROBLEM]\n'
	s += 'Input-output test cases:\n'
	for i in range(get_num_examples(dataset_element.inputs)):
		s += f'	Case {i + 1}. '
		s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
	s += '\nProgram:\n```python\n'
	return s

def get_test_prompt(dataset_element: DatasetElement) -> str:
	"""Gets a prefix of the prompt describing one dataset element."""
	s = 'Below are some example help you understand the `dsl` module. Now your task is to generate the dsl program that maps the following inputs to their corresponding outputs.[BEGIN PROBLEM]\n'
	s += 'Input-output test cases:\n'
	for i in range(get_num_examples(dataset_element.inputs)):
		s += f'	Case {i + 1}. '
		s += f'"Input:{dataset_element.inputs[i]}" Output:"{dataset_element.outputs[i]}"\n'
	s += '\n"Generate a rule that maps the following inputs to their corresponding outputs. Please format your rule as follows: Rule: <Your rule>'
	return s

def get_rule_test_prompt(dataset_element: DatasetElement) -> str:
	"""Gets a prefix of the prompt describing one dataset element."""
	s = ''
	s += 'Input-output test cases:\n'
	for i in range(get_num_examples(dataset_element.inputs)):
		s += f'	Case {i + 1}. '
		s += f'"Input:{dataset_element.inputs[i]}" Output:"{dataset_element.outputs[i]}"\n'
	s += '\n"Generate a rule that maps the below inputs to their corresponding outputs. Please format your rule as follows: Rule: <Your rule>'
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

def few_shot_rule_prompt(test_problem: DatasetElement,
										) -> str:
	prompt_parts = [rule_dsl_description()]
	# prompt_parts.extend(get_prompt(d) for d in few_shot_examples)
	prompt_parts.append(get_rule_test_prompt(test_problem))
	return '\n'.join(prompt_parts)