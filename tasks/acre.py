import ast
import logging
import re
from collections import defaultdict

from prompts.acre import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    nl_rule_prompt,
    nl_rule_to_output_prompt,
    nl_rule_to_output_prompt_with_format,
    nl_rule_with_feedback_prompt,
    rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_with_feedback_prompt,
)
from tasks.base import Task
from utils.format_utils import format_list
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)


class ACRE(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_name in CLAUDE_MODELS:
            self.io_prompt = io_prompt_with_format
            if self.rule_type == "nl":
                self.rule_prompt = nl_rule_prompt
                self.rule_to_output_prompt = nl_rule_to_output_prompt_with_format
                self.rule_with_feedback_prompt = nl_rule_with_feedback_prompt
            else:
                self.rule_prompt = rule_prompt
                self.rule_to_output_prompt = rule_to_output_prompt_with_format
                self.rule_with_feedback_prompt = rule_with_feedback_prompt
        else:
            self.io_prompt = io_prompt
            if self.rule_type == "nl":
                self.rule_prompt = nl_rule_prompt
                self.rule_to_output_prompt = nl_rule_to_output_prompt
                self.rule_with_feedback_prompt = nl_rule_with_feedback_prompt
            else:
                self.rule_prompt = rule_prompt
                self.rule_to_output_prompt = rule_to_output_prompt
                self.rule_with_feedback_prompt = rule_with_feedback_prompt

        self.example_prompt = example_prompt
        self.feedback_prompt = feedback_prompt

    def format_input(self, input):
        return format_list(input)

    def format_output(self, output):
        return output

    def rule_to_outputs(self, rule, examples):
        matches = re.findall(r"\{[^{}]*\}", rule)
        if len(matches) != 1:
            logger.info(f"Multiple matches found: {matches}")
        rule = matches[-1]
        try:
            rule = ast.literal_eval(rule)
        except:
            logger.info(f"Failed to parse rule: {rule}")
            return [None] * len(examples)
        rule = {k.replace("object", "").strip(): v for k, v in rule.items()}
        state_to_objs = defaultdict(set)
        rule_objects = set()
        for objs, state in rule.items():
            objs = objs.split(",")
            for obj in objs:
                obj = obj.strip()
                state_to_objs[state].add(obj)
                rule_objects.add(obj)
        all_objects = set()
        outputs = []
        for example in examples:
            objects = set([str(x) for x in example["input"]])
            all_objects |= objects
            if objects.intersection(state_to_objs["on"]):
                outputs.append("on")
            elif objects.intersection(state_to_objs["undetermined"]):
                outputs.append("undetermined")
            elif objects.issubset(state_to_objs["off"]):
                outputs.append("off")
            else:
                outputs.append("undetermined")
        if not all_objects.issubset(rule_objects):
            missing_objects = all_objects - rule_objects
            logger.info(f"Rule does not cover all objects: missing {missing_objects}")
        return outputs

    def apply_all_rules(self, idxs, all_rules, all_examples):
        if self.rule_type == "nl":
            assert self.interpreter_type == "lm"
            return self.apply_all_rules_with_lm(idxs, all_rules, all_examples)
        all_outputs = []
        for rule, examples in zip(all_rules, all_examples):
            outputs = self.rule_to_outputs(rule, examples)
            all_outputs.append(outputs)
        return all_outputs
