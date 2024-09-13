import re
from collections import Counter

from utils.grammar.qcfg_parser import parse
from utils.grammar.qcfg_rule import ROOT_RULE_KEY, apply_target


def canonicalize_rule(rule_str):
    lhs, rhs = rule_str.split("->")
    lhs_tokens = lhs.strip().strip('"').split(" ")
    rhs_tokens = rhs.strip().strip('"').split(" ")
    lhs_nts = []
    for token in lhs_tokens:
        if token.startswith("##") and token not in lhs_nts:
            lhs_nts.append(token)
    rhs_nts = []
    for token in rhs_tokens:
        if token.startswith("##") and token not in rhs_nts:
            rhs_nts.append(token)
    nt_to_idx = {nt: idx + 1 for idx, nt in enumerate(lhs_nts)}
    lhs_tokens = [f"NT_{nt_to_idx[nt]}" if nt in nt_to_idx else nt for nt in lhs_tokens]
    rhs_tokens = [f"NT_{nt_to_idx[nt]}" if nt in nt_to_idx else nt for nt in rhs_tokens]
    return " ".join(lhs_tokens) + " ### " + " ".join(rhs_tokens)


def strip_brackets(text):
    text = re.sub(r"\([^()]*\)", "", text).strip()
    if text.startswith("<") and text.endswith(">"):
        text = re.sub(r"^<|>$", "", text)
    return " ".join(text.split())


def postprocess_token(tokens):
    new_tokens = []
    for token in tokens:
        if token.startswith("##") and token[len("##") :].lower() == token[len("##") :]:
            new_tokens.append(token[len("##") :])
        else:
            new_tokens.append(token)
    return new_tokens


def postprocess_rule(rule):
    rule = strip_brackets(rule)
    source, target = rule.split(" -> ")
    new_source = postprocess_token(source.split(" "))
    new_target = postprocess_token(target.split(" "))
    rule = " ".join(new_source) + " -> " + " ".join(new_target)
    return rule


def get_score(rule_to_priority, rule, children):
    score = 0.0
    for child in children:
        parent_priority = rule_to_priority[rule]
        child_priority = rule_to_priority[child.rule]
        application_score = float(parent_priority < child_priority)
        score += application_score
        score += child.score
    return score


def get_root_score(rule_to_priority, child):
    return get_score(rule_to_priority, ROOT_RULE_KEY, [child])


def get_node_fn(rule_to_priority):
    """Return node_fn."""

    def node_fn(unused_span_begin, unused_span_end, rule, children):
        return ScoredChartNode(rule_to_priority, rule, children)

    return node_fn


def postprocess_cell_fn(nodes):
    return nodes


class ScoredChartNode(object):
    """Represents node in chart."""

    def __init__(self, rule_to_priority, rule, children):
        self.score = get_score(rule_to_priority, rule, children)
        self.rule = rule
        self.children = children

    def __str__(self):
        return "%s (%s)" % (self.rule, self.score)

    def __repr__(self):
        return self.__str__()

    def target_string(self):
        """Construct target string recursively."""
        return apply_target(self.rule, [node.target_string() for node in self.children])


def rule_to_parses(rule_to_priority, examples):
    qcfg_rules = list(rule_to_priority.keys())
    rule_to_priority[ROOT_RULE_KEY] = 0
    all_parses = []
    node_fn = get_node_fn(rule_to_priority)
    for example in examples:
        tokens = example["input"].split()
        nodes = parse(
            tokens,
            qcfg_rules,
            node_fn=node_fn,
            postprocess_cell_fn=postprocess_cell_fn,
        )

        if not nodes:
            all_parses.append(None)
        else:
            targets_and_scores = [
                (node.target_string(), get_root_score(rule_to_priority, node))
                for node in nodes
            ]
            occurrences = Counter(target for target, _ in targets_and_scores)
            # Sort by priority, then by occurrences (as majority vote), then by length, then by string.
            sort_key = lambda x: (
                -x[1],
                -occurrences[x[0]],
                len(x[0].split(" ")),
                x[0],
            )
            targets_and_scores = sorted(targets_and_scores, key=sort_key)
            best_target, _ = targets_and_scores[0]
            all_parses.append(best_target)
    return all_parses
