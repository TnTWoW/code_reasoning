stringprogs = [
    # add decimal point if not present
    'IF(EQUALS(-1, FIND(".", var_0)), CONCATENATE(var_0, ".0"), var_0)',
    # add plus sign to positive integers
    'IF(EXACT(LEFT(var_0, 1), "-"), var_0, CONCATENATE("+", var_0))',
    # append AM or PM to the hour depending on if it’s morning
    'CONCATENATE(LEFT(var_0, MINUS(FIND(":",var_0), 1)), IF(EXACT(var_1, "morning"), " AM", " PM"))',
    # fix capitalization of city and state
    "CONCATENATE(LEFT(PROPER(var_0), MINUS(LEN(var_0), 1)), UPPER(RIGHT(var_0, 1)))",
    # capitalize the first word and lowercase the rest
    "REPLACE(LOWER(var_0), 1, 1, UPPER(LEFT(var_0, 1)))",
    # whether the first string contains the second
    "TO_TEXT(GE(FIND(var_1, var_0), 0))",
    # whether the first string contains the second, ignoring case
    "TO_TEXT(GE(FIND(LOWER(var_1), LOWER(var_0)), 0))",
    # count the number of times the second string appears in the first
    'TO_TEXT(DIVIDE(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, var_1, ""))), LEN(var_1)))',
    # create email address from name and company
    'LOWER(CONCATENATE(LEFT(var_0, 1), CONCATENATE(var_1, CONCATENATE("@", CONCATENATE(var_2, ".com")))))',
    # change DDMMYYYY date to MM/DD/YYYY
    'CONCATENATE(MID(var_0, 3, 2), CONCATENATE("/", REPLACE(var_0, 3, 2, "/")))',
    # change YYYY-MM-DD date to YYYY/MM/DD
    'SUBSTITUTE(var_0, "-", "/")',
    # change YYYY-MM-DD date to MM/DD
    'SUBSTITUTE(RIGHT(var_0, 5), "-", "/")',
    # extract the part of a URL between the 2nd and 3rd slash
    'MID(var_0, ADD(FIND("//", var_0), 2), MINUS(MINUS(FINDI("/", var_0, 9), FIND("/", var_0)), 2))',
    # extract the part of a URL starting from the 3rd slash
    'RIGHT(var_0, ADD(1, MINUS(LEN(var_0), FINDI("/", var_0, ADD(FIND("//", var_0), 2)))))',
    # get first name from second column
    'LEFT(var_1, MINUS(FIND(" ", var_1), 1))',
    # whether the string is lowercase
    'IF(EXACT(var_0, LOWER(var_0)), "true", "false")',
    # get last name from first column
    'RIGHT(var_0, MINUS(LEN(var_0), FIND(" ", var_0)))',
    # output "Completed" if 100%, "Not Yet Started" if 0%, "In Progress" otherwise
    'IF(EXACT(var_0,"100%"), "Completed", IF(EXACT(var_0,"0%"), "Not Yet Started", "In Progress"))',
    # enclose negative numbers in parentheses
    'IF(EXACT(LEFT(var_0, 1), "-"), CONCATENATE(SUBSTITUTE(var_0, "-", "("), ")"), var_0)',
    # pad text with spaces to a given width
    'CONCATENATE(REPEAT(" ", MINUS(VALUE(var_1), LEN(var_0))), var_0)',
    # pad number with 0 to width
    'CONCATENATE(REPEAT("0", MINUS(5, LEN(var_0))), var_0)',
    # the depth of a path, i.e., count the number of /
    'TO_TEXT(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, "/", ""))))',
    # extract the rest of a word given a prefix
    "RIGHT(var_0, MINUS(LEN(var_0), LEN(var_1)))",
    # prepend Mr. to last name
    'CONCATENATE("Mr. ", RIGHT(var_0, MINUS(LEN(var_0), FIND(" ", var_0))))',
    # prepend Mr. or Ms. to last name depending on gender
    'CONCATENATE(IF(EXACT(var_1, "male"), "Mr. ", "Ms. "), RIGHT(var_0, MINUS(LEN(var_0), FIND(" ", var_0))))',
    # remove leading and trailing spaces and tabs, and lowercase
    "TRIM(LOWER(var_0))",
    # replace <COMPANY> in a string with a given company name
    'SUBSTITUTE(var_0, "<COMPANY>", var_1)',
    # replace com with org
    'SUBSTITUTEI(var_0, "com", "org", 1)',
    # select the first string, or the second if the first is NONE
    'IF(EXACT(var_0, "NONE"), var_1, var_0)',
    # select the longer of 2 strings, defaulting to the first if equal length
    "IF(GT(LEN(var_1), LEN(var_0)), var_1, var_0)",
    # whether the two strings are exactly equal, yes or no
    'IF(EXACT(var_0, var_1), "yes", "no")',
    # whether the two strings are exactly equal ignoring case, yes or no
    'IF(EXACT(LOWER(var_0), LOWER(var_1)), "yes", "no")',
    # length of string
    "TO_TEXT(LEN(var_0))",
    # extract the rest of a word given a suffix
    "LEFT(var_0, MINUS(LEN(var_0), LEN(var_1)))",
    # swap the case of a string that is entirely uppercase or lowercase
    "IF(EXACT(var_0, LOWER(var_0)), UPPER(var_0), LOWER(var_0))",
    # truncate and add ... if longer than 15 characters
    'IF(GT(LEN(var_0), 15), CONCATENATE(LEFT(var_0, 15), "..."), var_0)',
    # create acronym from two words in one cell
    'CONCATENATE(LEFT(var_0, 1), MID(var_0, ADD(FIND(" ", var_0), 1), 1))',
    # create capitalized acronym from two words in one cell
    'UPPER(CONCATENATE(LEFT(var_0, 1), MID(var_0, ADD(FIND(" ", var_0), 1), 1)))',
]

input = [
    "",
    " ",
    "1",
    "12",
    "12.",
    "12.0",
    "0",
    ".3",
    "15e-15",
    "-1",
    "-12",
    "-12.0",
    "12:00",
    "12:01",
    "Cambridge, ma",
    "cambridge, ma",
    "CAMBRIDGE, MA",
    "Cambridge ma",
    "cambridge ma",
    "CAMBRIDGE MA",
    "rome",
    "Rome",
    "rome of Italy",
    "Rome of Italy",
    "hello world",
    "HELLO WORLD",
    "Hello World",
    "hellO WORLD",
    "02032022",
    "2022-03-02",
    "https://github.com/zaxtax/bustle/",
    "https://github.com/zaxtax/bustle#readme",
    "0%",
    "100%",
    "50%",
    "34.3%",
    "(3 + 4 - 5)",
    "(3 + 4 + -5)",
    "(3 + 4) - 5",
    "(3 + 4) + -5",
    "morning",
    "afternoon",
    "hello",
    "world",
    "HELLO",
    "WORLD",
    "HELLo",
    "hellO",
    "o",
    "l",
    "z",
    "Gerald Jay Sussman",
    "Gerry Sussman",
    "Gerald J. Sussman",
    "Gerald J Sussman",
    "3",
    "4",
    "5",
    "male",
    "female",
    "<COMPANY>",
    "<COMPANY>, inc",
    "NONE",
    "Github",
    "1", "2.0", "hello", "-1", "-1.0", "-", "hello-you"
]


def all_inputs(n, inps=[[]]):
    if n == 0:
        return list(zip(*inps))
    else:
        return all_inputs(n - 1, [[x] + inp for x in input for inp in inps])


def test():
    from stringdsl import StringDsl
    from dslparser import parse, printer

    sl = StringDsl()
    dummy_inp = ["hello" for i in range(3)]
    inpss = [all_inputs(i) for i in range(4)]
    for prog in stringprogs:
        print("parsing", prog)
        ast = parse(sl, prog)
        print("ast", ast)
        txt = printer(sl, ast)
        print("print", txt)
        ast2 = parse(sl, txt)
        txt2 = printer(sl, ast2)
        assert ast == ast2

        inps = inpss[sl.numInputs(ast)]
        try:
            v = sl.eval(ast, dummy_inp)
            print("value", v)

            sl.evalIO(ast, inps)
        except:
            continue


if __name__ == "__main__":
    print("running tests...")
    test()
    print("done")
