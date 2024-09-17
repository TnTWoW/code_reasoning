# TODO: define an API

import re

# Listing 5: code for all Properties acting on single Strings.
props_str = (
    ("str",),
    [
        lambda str: str == "",  # is empty?
        lambda str: len(str) == 1,  # is single char?
        lambda str: len(str) <= 5,  # is short string?
        lambda str: str.islower(),  # is lowercase?
        lambda str: str.isupper(),  # is uppercase?
        lambda str: " " in str,  # contains space?
        lambda str: "," in str,  # contains comma?
        lambda str: "." in str,  # contains period?
        lambda str: "-" in str,  # contains dash?
        lambda str: "/" in str,  # contains slash?
        lambda str: bool(re.search(r"\d", str)),  # contains digits?
        lambda str: bool(re.match(r"\d+$", str)),  # only digits?
        lambda str: bool(re.search(r"\d", str)),  # contains digits?
        lambda str: bool(re.match(r"\d+$", str)),  # only digits?
        lambda str: bool(re.search(r"[a-zA-Z]", str)),  # contains letters?
        lambda str: bool(re.match(r"[a-zA-Z]+$", str)),  # only letters?
    ],
)

# Listing 6: code for all Properties acting on single Integers.
props_int = (
    ("int",),
    [
        lambda integer: integer == 0,  # is zero?
        lambda integer: integer == 1,  # is one?
        lambda integer: integer == 2,  # is two?
        lambda integer: integer < 0,  # is negative?
        lambda integer: 0 < integer and integer <= 3,  # is small integer?
        lambda integer: 3 < integer and integer <= 9,  # is medium integer?
        lambda integer: 9 < integer,  # is large integer?
    ],
)

# Listing 7: code for all Properties acting on a String and the output String.
props_str2str = (
    ("str", "str"),
    [
        lambda str, outputStr: str in outputStr,  # output contains input?
        lambda str, outputStr: outputStr.startswith(str),  # output starts with input?
        lambda str, outputStr: outputStr.endswith(str),  # output ends with input?
        lambda str, outputStr: outputStr in str,  # input contains output?
        lambda str, outputStr: str.startswith(outputStr),  # input starts with output?
        lambda str, outputStr: str.endswith(outputStr),  # input ends with output?
        lambda str, outputStr: str.lower()
        in outputStr.lower(),  # output contains input ignoring case?
        lambda str, outputStr: outputStr.lower().startswith(
            str.lower()
        ),  # output starts with input ignoring case?
        lambda str, outputStr: outputStr.lower().endswith(
            str.lower()
        ),  # output ends with input ignoring case?
        lambda str, outputStr: outputStr.lower()
        in str.lower(),  # input contains output ignoring case?
        lambda str, outputStr: str.lower().startswith(
            outputStr.lower()
        ),  # input starts with output ignoring case?
        lambda str, outputStr: str.lower().endswith(
            outputStr.lower()
        ),  # input ends with output ignoring case?
        lambda str, outputStr: str == outputStr,  # input equals output?
        lambda str, outputStr: str.lower()
        == outputStr.lower(),  # input equals output ignoring case?
        lambda str, outputStr: len(str)
        == len(outputStr),  # input same length as output?
        lambda str, outputStr: len(str) < len(outputStr),  # input shorter than output?
        lambda str, outputStr: len(str) > len(outputStr),  # input longer than output?
    ],
)

# Listing 8: code for all Properties acting on an Integer and the output String.
props_int2str = (
    ("int", "str"),
    [
        lambda integer, outputStr: integer
        < len(outputStr),  # is less than output length?
        lambda integer, outputStr: integer
        <= len(outputStr),  # is less or equal to output length?
        lambda integer, outputStr: integer
        == len(outputStr),  # is equal to output length?
        lambda integer, outputStr: integer
        >= len(outputStr),  # is greater or equal to output length?
        lambda integer, outputStr: integer
        > len(outputStr),  # is greater than output length?
        lambda integer, outputStr: abs(integer - len(outputStr))
        <= 1,  # is very close to output length?
        lambda integer, outputStr: abs(integer - len(outputStr))
        <= 3,  # is close to output length?
    ],
)

llProps = [props_str, props_int, props_str2str, props_int2str]
