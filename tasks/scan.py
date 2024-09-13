import logging
import re

from timeout_decorator import TimeoutError, timeout
from tqdm import tqdm

from prompts.scan import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    nl_rule_prompt,
    nl_rule_to_output_prompt,
    nl_rule_to_output_prompt_with_format,
    rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_with_feedback_prompt,
)
from tasks.base import Task
from utils.grammar.qcfg_rule import get_nts, get_num_nts, rule_from_string
from utils.grammar_utils import canonicalize_rule, rule_to_parses
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)

TIMEOUT = 5


class SCAN(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_name in CLAUDE_MODELS:
            self.io_prompt = io_prompt_with_format
            if self.rule_type == "nl":
                self.rule_prompt = nl_rule_prompt
                self.rule_to_output_prompt = nl_rule_to_output_prompt_with_format
            else:
                self.rule_prompt = rule_prompt
                self.rule_to_output_prompt = rule_to_output_prompt_with_format
        else:
            self.io_prompt = io_prompt
            if self.rule_type == "nl":
                self.rule_prompt = nl_rule_prompt
                self.rule_to_output_prompt = nl_rule_to_output_prompt
            else:
                self.rule_prompt = rule_prompt
                self.rule_to_output_prompt = rule_to_output_prompt

        self.example_prompt = example_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_prompt
        self.feedback_prompt = feedback_prompt

    def format_input(self, input):
        return input

    def format_output(self, output):
        return output

    def get_rule(self, response):
        # Extract rules
        rule_pattern = r"Rule (\d+): (.*?)(?:\n|$)"
        rule_matches = re.findall(rule_pattern, response, re.DOTALL)
        rule_matches = sorted(rule_matches, key=lambda x: int(x[0]))

        if self.rule_type == "nl":
            rules = [rule[1] for rule in rule_matches]
            return rules

        # Extract priorities
        priority_pattern = r"Priority (\d+): (\d+)"
        priority_matches = re.findall(priority_pattern, response)
        priority_matches = sorted(priority_matches, key=lambda x: int(x[0]))

        idx2rule = {rule[0]: rule[1] for rule in rule_matches}
        idx2priority = {priority[0]: priority[1] for priority in priority_matches}

        combs = []
        for idx, rule in idx2rule.items():
            if idx not in idx2priority:
                idx2priority[idx] = 0
            combs.append((rule, idx2priority[idx]))
        return combs

    def format_rule(self, rules):
        if self.rule_type == "nl":
            rule_strs = [f"Rule {i + 1}: {rule}" for i, rule in enumerate(rules)]
        else:
            rule_strs = [
                f"Rule {i + 1}: {rule}\nPriority {i + 1}: {priority}"
                for i, (rule, priority) in enumerate(rules)
            ]
        return "\n".join(rule_strs)

    @timeout(TIMEOUT)
    def rule_to_parses(self, rules, examples):
        rule_to_priority = {}
        for rule, priority in rules:
            try:
                qcfg_rule = rule_from_string(canonicalize_rule(rule))
                if len(get_nts(qcfg_rule.source)) == get_num_nts(qcfg_rule.source):
                    rule_to_priority[qcfg_rule] = int(priority)
                else:
                    logger.info("Invalid rule: %s" % rule)
            except:
                logger.info("Cannot parse rule: %s" % rule)
        all_parses = rule_to_parses(rule_to_priority, examples)
        return all_parses

    def apply_all_rules(self, idxs, all_rules, all_examples):
        if self.rule_type == "nl":
            assert self.interpreter_type == "lm"
            return self.apply_all_rules_with_lm(idxs, all_rules, all_examples)
        all_outputs = []
        total = len(all_examples)
        for rule, examples in tqdm(
            zip(all_rules, all_examples), desc="Applying rules", total=total
        ):
            try:
                parses = self.rule_to_parses(rule, examples)
            except TimeoutError:
                logger.info(f"Timeout using {rule}:")
                parses = [None] * len(examples)
            all_outputs.append(parses)
        return all_outputs
