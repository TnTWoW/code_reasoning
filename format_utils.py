import ast
import re


def format_list(list, sep=", ", bracket=False):
    rep = sep.join([str(x) for x in list])
    if bracket:
        return "[" + rep + "]"
    return rep


def format_grid(grid, row_sep="\n", sep=", "):
    return row_sep.join([format_list(row, sep, bracket=True) for row in grid])


def unformat_grid(grid_string, row_sep="\n", sep=", "):
    grid_string = grid_string.replace(f"]{row_sep}[", "], [")
    try:
        nested_list = ast.literal_eval(grid_string)
        nested_list = [[str(x) for x in row] for row in nested_list]
        return nested_list
    except:
        rows = re.findall(r"\[.*?\]", grid_string)
        nested_list = []
        for row in rows:
            row = row[1:-1]
            nested_list.append([item for item in row.split(sep)])
        return nested_list


def flatten(list1, list2):
    """Flatten and match a list and a nested list.

    Args:
        list1: [ex1, ex2, ex3, ...]
        list2: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]

    Returns:
        flatten_list1: [ex1, ex1, ex2, ex2, ...]
        flatten_list2: [r1_ex1, r2_ex1, r1_ex2, r2_ex2, ...]
    """
    assert len(list1) == len(list2)
    flatten_list1 = []
    flatten_list2 = []
    for ex, nested_ex in zip(list1, list2):
        for item in nested_ex:
            flatten_list1.append(ex)
            flatten_list2.append(item)
    return flatten_list1, flatten_list2


def unflatten(flatten_list3, list2):
    """Unflatten a flatten list to match a nested list.

    Args:
        flatten_list3: [r1_ex1, r2_ex1, r1_ex2, r2_ex2, ...]
        list2: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]

    Returns:
        nested_list3: [[r1_ex1, r2_ex1], [r1_ex2, r2_ex2, ...], ...]
    """
    list3 = []
    index = 0
    for inner_list in list2:
        length = len(inner_list)
        list3.append(flatten_list3[index : index + length])
        index += length
    return list3


def extract_response(prefixes, response):
    patterns = [f"{prefix}: (.*)" for prefix in prefixes]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            assert len(matches) == 1
            return matches[0].strip()


def str_to_list(s):
    try:
        s = re.sub(r"\b0+(\d)", r"\1", s)
        return ast.literal_eval(s)
    except:
        return s
