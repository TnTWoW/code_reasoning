import logging
import re

from prompts.cruxeval import (
    input_prompt,
    cot_input_prompt,
    output_prompt,
    cot_output_prompt,
)
from tasks.base import PythonTask
from utils.format_utils import format_grid, format_list, str_to_list, unformat_grid
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)

class CruxEvalInput(PythonTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io_prompt = input_prompt
        self.rule_to_output_prompt = rule_to_output_prompt

        if self.rule_type == "python":
            if 'Llama' in self.model_name:
                self.rule_prompt = llama_python_rule_prompt
                self.rule_with_feedback_prompt = llama_rule_with_feedback_prompt
                self.coc_prompt = coc_prompt
            else:
                self.rule_prompt = python_rule_prompt
                self.rule_with_feedback_prompt = rule_with_feedback_prompt
                self.coc_prompt = coc_prompt
        elif self.rule_type == "noisy":
            self.rule_prompt = noisy_rule_prompt
            self.rule_with_feedback_prompt = nosiy_rule_with_feedback_prompt
        else:
            self.rule_prompt = structured_rule_prompt
            self.rule_with_feedback_prompt = rule_with_feedback_prompt
        self.example_prompt = example_prompt
        self.feedback_prompt = feedback_prompt
        self.rule_to_python_prompt = rule_to_python_prompt