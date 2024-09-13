# Reference: https://github.com/langchain-ai/langchain/blob/4a07fba9f0f7d949c9c5cb057a4d09c7db1dfb42/libs/langchain/langchain/utilities/python.py

import collections
import copy
import logging
import math
import multiprocessing
import re
import types
from collections import Counter

logger = logging.getLogger(__name__)

TIMEOUT = 2


class PythonExecutor:
    """Simulates a standalone Python Executor."""

    def __init__(self, globals=None, locals=None):
        self.globals = globals if globals is not None else {}
        self.locals = locals if locals is not None else {}

        # Import some useful modules
        self.globals["math"] = math
        self.globals["Counter"] = Counter
        self.globals["collections"] = collections
        self.globals["copy"] = copy

    @classmethod
    def worker(
        cls,
        command,
        globals,
        queue,
    ):
        try:
            local_namespace = {}
            exec(command, globals, local_namespace)
            result = local_namespace.get("result")
            if isinstance(result, types.GeneratorType):
                result = list(result)
            if isinstance(result, list):
                result = [
                    list(x) if isinstance(x, types.GeneratorType) else x for x in result
                ]
            queue.put(result)
        except Exception as e:
            queue.put(repr(e))

    def run(self, command, timeout=3):
        """Run command with own globals/locals and returns the result.
        Timeout after the specified number of seconds."""

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return None
        else:
            self.worker(command, self.globals, queue)
        # get the result from the worker function
        return queue.get()


def extract_program(response):
    pattern = r"```python\s*([\s\S]+?)\s*```"
    matches = re.findall(pattern, response)
    matches = [match for match in matches if "def" in match]
    if matches:
        return "\n".join(matches)
    return response


def extract_function_names(function_string):
    matches = re.findall(r"def (\w+)\(", function_string)
    return matches


def execute_function(function_string, inputs, timeout=TIMEOUT, verbose=False):
    executor = PythonExecutor()

    outputs = []
    function_names = extract_function_names(function_string)
    try:
        fn_name = "fn" if "fn" in function_names else function_names[-1]
    except:
        return [None] * len(inputs)

    for inp in inputs:
        output = executor.run(
            f"{function_string}\nresult = {fn_name}({inp})", timeout=timeout
        )
        if verbose:
            if output is None:
                print(f"Timeout on {inp} using:")
                print(function_string)
        outputs.append(output)
    return outputs
