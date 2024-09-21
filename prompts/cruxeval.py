input_prompt = """You will be given a function f and an output in the form f(??) == output. Find any input such that 
executing f on the input leads to the given output. There may be multiple answers, but you should only output one. 
Complete the assertion with one such input that will produce the output when executing the function.

```python
def f(s1, s2):
    return s1+s2
# assert f(??) == "banana"
```
Answer:
```python
assert f("ba", "nana") == "banana"
```

```python
{code}
# assert f(??) == {output}
```
Answer:
```python
"""

cot_input_prompt = """You will be given a function f and an output in the form f(??) == output. Your task is to find 
any input such that executing f on the input leads to the given output. There may be multiple answers, 
but only output one. First, think step by step. Express your answer as a passing assertion containing the input and 
the given output.

'''python
def f(x):
    return x + 1
# assert f(??) == 17
```
Let's analyze the code step by step:
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion.
We know that f(??) == 17. Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 

Answer:
```python
assert f(16) == 17
```

```python
{code}
# assert f(??) == {output}
```
Let's analyze the code step by step:
"""

output_prompt = """You are given a Python function and an assertion containing an input to the function. Complete the 
assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the 
provided code on the given input, even if the function is incorrect or incomplete. Provide the full assertion with 
the correct output, following the examples.

```python
def f(s):
    return s + "a"
# assert f("x9j") == ??
```
Answer:
```python
assert f("x9j") == "x9ja"
```

```python
{code}
# assert f({input}) == ??
```
Answer:
```python
"""

cot_output_prompt = """You are given a Python function and an assertion containing an input to the function. Complete 
the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing 
the provided code on the given input, even if the function is incorrect or incomplete. Execute the program step by 
step before arriving at an answer, and provide the full assertion with the correct output, following the examples.

```python
def f(s):
    s = s + s
    return "b" + s + "a"
# assert f("hi") == ??
```
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".

Answer:
```python
assert f("hi") == "bhihia"
```

```python
{code}
# assert f({input}) == ??
```
Let's execute the code step by step:
"""

output_feedback_prompt = """The predicted output {p_output} does not match the actual execution result {output}.
Please regenerate the result for f({input})
 
```python
{code}
# assert f({input}) == ??
```
Answer:
```python
"""

coc_output_prompt = """You are given a Python function and an assertion containing an input to the function. Complete 
the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing 
the provided code on the given input, even if the function is incorrect or incomplete. Execute the program step by 
step before arriving at an answer, and provide the full assertion with the correct output, following the examples.

```python
def f(s):
    s = s + s
    result = "b" + s + "a"
    return result
# assert f("hi") == ??
```
[TRACE]
state: {{}}
line: f("hi")
explanation: Python execution.
delta state: {{'s': 'hi'}}
line: s = s + s
explanation: Python execution.
delta state: {{'s': 'hihi'}}
line: result = "b" + s + "a"
explanation: Python execution.
delta state: {{'result': 'bhihia'}}
line: return result
explanation: Python execution.
delta state: {{}}
[/TRACE]
Answer:
```python
assert f("hi") == "bhihia"
```

```python
{code}
# assert f({input}) == ??
```
[TRACE]
"""

rule_output_prompt = """You are given a piece of Python function. Please analyze the functionality of the Python 
code step by step, and based on the input and the function's purpose, provide the full assertion with the 
correct output.

'''python
{code}
# assert f({input}) == ??
'''

Please provide the analysis and the assertion in the following format:
Analysis: <Your analysis>
Answer:
```python
assert f({input}) == <Your output>
```
"""

rule_with_feedback_output_prompt = """You are given a piece of Python function:

'''python
{code}
'''

And the following is your analysis of the code: {rule}

The predicted output {p_output} based on your analysis does not match the actual execution result. Please reflect 
on the potential errors in the analysis steps and provide the full assertion with the correct output.

Please provide the analysis and the assertion in the following format:
Analysis: <Your analysis>
Answer:
```python
assert f({input}) == <Your output>
```
"""

