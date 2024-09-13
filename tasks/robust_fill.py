import logging
import re

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
from tasks.base import PythonTask
from utils.format_utils import format_grid, format_list, str_to_list, unformat_grid
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)

class RobustFill(PythonTask):
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

    def format_input(self, input):
        return input

    def format_output(self, output):
        return output

    def get_rule(self, response):
        return response

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

    def format_input(self, input):
        return "\n" + format_grid(input, row_sep="\n", sep=", ")

    def format_output(self, output):
        grid = self.canonicalize(output)
        try:
            if isinstance(grid, list) and isinstance(grid[0], list):
                return format_grid(grid, row_sep="\n", sep=", ")
            elif isinstance(grid, list):
                return format_list(grid, sep=", ")
            return grid
        except Exception:
            return grid

    def get_rule(self, response):
        return response

    def get_feedback(self, response):
        return response

    def get_example(self, response):
        return response

    def get_io(self, response):
        return response

    def get_rule_with_feedback(self, response):
        return response

    def get_rule_to_output(self, response):
        return response

    def get_rule_to_python(self, response):
        return response

    def get_example_prompt(self):
        return self.example_prompt

    def get_feedback_prompt(self):
        return self.feedback_prompt