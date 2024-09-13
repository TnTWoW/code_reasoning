# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data structures for representing Quasi-Synchronous CFG (QCFG) rules.

Currently, both terminal and non-terminal symbols are simply represented
as strings, with special strings reserved for non-terminals.

QCFG rules used by NQG follow the following restrictions:
- There is only one non-terminal symbol, `NT`
Therefore, we only need to reserve two strings to represent indexed
non-terminals.

We also expect all rules to be normalized as follows: a non-terminal with index
2 should never appear before a non-terminal with index 1 in the source
sequence.

Note that this data structure could potentially be improved:
1. A more flexible representation for terminal and non-terminal symbols
would avoid possible collisions between terminal and non-terminal symbols,
and allow for representing QCFGs that do not conform to the restrictions above.
2. Representing symbols as integers rather than strings may have computational
benefits for various operations over QCFG rules.

Reference: https://github.com/google-research/language/blob/master/language/compgen/csl/qcfg/qcfg_rule.py
"""

import re
import typing
from dataclasses import dataclass  # pylint: disable=g-importing-member

# Used for separating source and target sequences for string formatting.
SEPARATOR = "###"

NON_TERMINAL_PREFIX = "NT_"
ROOT_RULE_KEY = "ROOT_RULE_KEY"


# The LHS non-terminal is always assumed to be `NT` so is not represented.
@dataclass(frozen=True, order=True)
class QCFGRule:
    """Class for QCFG rule."""

    # Tuple of source symbols (strings).
    source: typing.Tuple[str, Ellipsis]
    # Tuple of target symbols (strings).
    target: typing.Tuple[str, Ellipsis]
    # The number of unique non-terminal indexes (0, 1, or 2).
    arity: int

    def __str__(self):
        return "%s %s %s" % (" ".join(self.source), SEPARATOR, " ".join(self.target))

    def __repr__(self):
        return str(self)


# Represents the non-terminal symbol `NT` with linked index i
@dataclass(frozen=True, order=True)
class NT:
    index: int

    def __str__(self):
        return "%s%d" % (NON_TERMINAL_PREFIX, self.index)

    def __repr__(self):
        return str(self)


def get_nt_index(nt):
    """Return the index of the NT."""
    if isinstance(nt, NT):
        return nt.index
    elif isinstance(nt, str):
        digits = re.findall(r"\d+", nt)
        assert len(digits) == 1
        return int(digits[0])
    else:
        raise ValueError("Unsupported NT type %s" % type(nt))


def is_nt(token):
    """Return whether the token is NT or not."""
    if not token:
        return False
    if isinstance(token, NT):
        return True
    elif isinstance(token, str) and token.startswith(NON_TERMINAL_PREFIX):
        return True
    return False


# TODO(petershaw): What should this max index be?
NT_SET = frozenset([str(NT(idx)) for idx in range(10)])


def is_nt_fast(token):
    """Return whether the token is NT or not if token is a string.."""
    return token in NT_SET


def is_single_nt(symbols):
    """Return whether the symbols are single NT."""
    return get_num_nts(symbols) == 1 and len(symbols) == 1


def get_nts(tokens):
    """Return a list of symbols that are NT."""
    return [token for token in tokens if is_nt(token)]


def get_num_nts(tokens):
    """Return the number of NTs in the list."""
    return len(set(get_nts(tokens)))


def is_allowed(symbols, allowed_terminals):
    """Returns True if the token is a nonterminal or an allowed terminal."""
    for token in symbols:
        if token not in NT_SET and token not in allowed_terminals:
            return False
    return True


def _get_arity(source):
    """Return the arity of the rule."""
    nts = set(get_nts(source))
    nt_indices = [get_nt_index(nt) for nt in nts]
    arity = len(nts)
    if sorted(nt_indices) != list(range(1, arity + 1)):
        raise ValueError("Source is unnormalized: %s" % source)
    return arity


def rule_from_string(rule_str):
    """Parse rule in format 'source SEPARATOR target'."""
    splits = rule_str.split(SEPARATOR)
    if len(splits) != 2:
        raise ValueError("Invalid rule string: %s" % rule_str)
    source_str, target_str = splits
    source = source_str.strip().split()
    target = target_str.strip().split()
    arity = _get_arity(source)
    return QCFGRule(tuple(source), tuple(target), arity)


def apply_target(rule, substitutions):
    """Return target string with non-terminals replaced with substitutions."""
    if rule.arity != len(substitutions):
        raise ValueError(
            "Rule (%s) arity does not match substitutions: %s" % (rule, substitutions)
        )
    output = []
    for token in rule.target:
        if is_nt(token):
            index = get_nt_index(token)
            output.append(substitutions[index - 1])
        else:
            output.append(token)
    return " ".join(output)


def _swap_nt_order(rhs, old_to_new):
    new_rhs = []
    for symbol in rhs:
        if is_nt_fast(symbol):
            old_idx = get_nt_index(symbol)
            new_rhs.append(str(NT(old_to_new[old_idx])))
        else:
            new_rhs.append(symbol)
    return tuple(new_rhs)


def canonicalize_nts(source, target, arity):
    """Follows convention of source indexes being in order."""
    source_nts = []
    for token in source:
        if is_nt(token) and token not in source_nts:
            source_nts.append(token)
    if len(set(source_nts)) != arity:
        raise ValueError("Bad arity 2 source: %s" % (source,))
    old_to_new = {get_nt_index(nt): idx + 1 for idx, nt in enumerate(source_nts)}
    source = _swap_nt_order(source, old_to_new)
    target = _swap_nt_order(target, old_to_new)
    return source, target


def make_rule(source, target):
    """Canoncalize NT indexes and return QCFGRule."""
    arity = len(get_nts(source))
    source, target = canonicalize_nts(source, target, arity)
    return QCFGRule(tuple(source), tuple(target), arity)
