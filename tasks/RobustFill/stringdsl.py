import itertools
from dsl import Dsl


class StringDsl(Dsl):
    def __init__(self, supportExtraConstants=False, fewOps=False, progs=[]):
        Name = "string manipulation DSL"

        # note: because we don't support overloading, we name
        # SubstituteI instead of overloading Substitute, and
        # FindI instead of overloading Find`
        FewOps = [
            "Concatenate",
            "Left",
            "Right",
            #"Mid",
            #"Replace",
            #"Trim",
            #"Repeat",
            #"Substitute",
            #"SubstituteI",
            #"To_Text",
            #"Lower",
            #"Upper",
            #"Proper",
            "If",
            #"Add",
            #"Minus",
            #"Divide",
            #"Find",
            #"FindI",
            #"Len",
            "Exact",
            #"Equals",
            #"GT",
            #"GE",
            #"IsNumber",
            #"Value",
        ]
        Ops = [
            "Concatenate",
            "Left",
            "Right",
            "Mid",
            "Replace",
            "Trim",
            "Repeat",
            "Substitute",
            "SubstituteI",
            "To_Text",
            "Lower",
            "Upper",
            "Proper",
            "If",
            "Add",
            "Minus",
            "Divide",
            "Find",
            "FindI",
            "Len",
            "Exact",
            "Equals",
            "GT",
            "GE",
            "IsNumber",
            "Value",
        ]

        Types = ["str", "int", "bool"]

        self.Ops = FewOps if fewOps else Ops
        self.Types = Types
        self.Name = Name
        self.supportExtraConstants = supportExtraConstants
        self.Ex = (("Concatenate", [("input", 0), ("input", 1)]),
                   [["hello", "world"], ["you", "domination"]],
                   ["helloyou", "worlddomination"],)
        super().__init__()
        self.progConstants = self.extractAllConstantStrings(progs)

    def execute(self, op, x):
        if op == "Concatenate":
            return x[0] + x[1]
        elif op == "Left":
            return x[0][: x[1]]
        elif op == "Right":
            return x[0][len(x[0]) - x[1] :]
        elif op == "Mid":
            return x[0][x[1] : x[1] + x[2]]
        elif op == "Replace":
            a = x[0]
            start = x[1]
            length = x[2]
            r = x[3]
            ret = a[:start] + r + a[start + length :]
            return ret
        elif op == "Trim":
            return x[0].strip()
        elif op == "Repeat":
            return x[0] * x[1]
        elif op == "Substitute":
            return x[0].replace(x[1], x[2])
        elif op == "SubstituteI":
            return x[0].replace(x[1], x[2], x[3])
        elif op == "To_Text":
            return str(x[0])
        elif op == "Lower":
            return x[0].lower()
        elif op == "Upper":
            return x[0].upper()
        elif op == "Proper":
            return x[0].title()
        elif op == "If":
            return x[1] if x[0] else x[2]
        elif op == "Add":
            return x[0] + x[1]
        elif op == "Minus":
            return x[0] - x[1]
        elif op == "Divide":
            return x[0] // x[1]
        elif op == "Find":
            return x[1].find(x[0])
        elif op == "FindI":
            return x[1].find(x[0], x[2])
        elif op == "Len":
            return len(x[0])
        elif op == "Exact":
            return x[0] == x[1]
        elif op == "Equals":
            return x[0] == x[1]
        elif op == "GT":
            return x[0] > x[1]
        elif op == "GE":
            return x[0] >= x[1]
        elif op == "IsNumber":
            return x[0].isnumeric()
        elif op == "Value":
            return int(x[0])
        else:
            assert False

    def types(self, op):
        s = "str"
        i = "int"
        b = "bool"

        if op == "Concatenate":
            return (s, (s, s))
        elif op in ["Left", "Right"]:
            return (s, (s, i))
        elif op == "Mid":
            return (s, (s, i, i))
        elif op == "Replace":
            return (s, (s, i, i, s))
        elif op == "Trim":
            return (s, (s,))
        elif op == "Repeat":
            return (s, (s, i))
        elif op == "Substitute":
            return (s, (s, s, s))
        elif op == "SubstituteI":
            return (s, (s, s, s, i))
        elif op == "To_Text":
            return (s, (i,))
        elif op in ["Lower", "Upper", "Proper"]:
            return (s, (s,))
        elif op == "If":
            return (s, (b, s, s))
        elif op in ["Add", "Minus", "Divide"]:
            return (i, (i, i))
        elif op == "Find":
            return (i, (s, s))
        elif op == "FindI":
            return (i, (s, s, i))
        elif op == "Len":
            return (i, (s,))
        elif op == "Exact":
            return (b, (s, s))
        elif op in ["Equals", "GT", "GE"]:
            return (b, (i, i))
        elif op == "IsNumber":
            return (b, (s,))
        elif op == "Value":
            return (i, (s,))
        else:
            assert False, "op %s is undefined in types" % op

    def constantVs(self, N, t, cs):
        return [(t, (c, [c for _ in range(N)])) for c in cs]

    def constantIn(self, c, O):
        return any(c in o for o in O)

    def constantMissing(self, c, I, O):
        return any((c in i) and (c not in o) for i,o in zip(I, O))

    def extractConstants(self, I, O, It, Ot):
        extraConstants = []
        if self.supportExtraConstants:
            extraConstants = [
                ",", ".", "!", "?", "(", ")", "|", "[", "]", "<", ">",
                "{", "}", "-", "+", "_", "/", "$", "#", ":",";", "@","%", "O"
            ]
        N = len(O)
        intVs = self.constantVs(N, "int", [0, 1, 2, 3, 5, 9, 15, 99])
        strVs = self.constantVs(
            N,
            "str",
            [" "] +
            self.progConstants +
            #[c for c in self.progConstants if self.constantMissing(c, O, I[0]) or self.constantMissing(c, I[0], O)] +
            extraConstants
            #[" "] + extraConstants
        )
        # TODO: string constants extracted from I/O examples
        return intVs + strVs

    def extractConstantStrings(self, x):
        if type(x) is tuple or type(x) is list:
            if x[0] == "input":
                return []
            else:
                return itertools.chain(*(self.extractConstantStrings(e) for e in x[1]))
        elif type(x) is str:
            return [x]
        else:
            return []
    def extractAllConstantStrings(self, xs):
        from dslparser import parse
        r = itertools.chain(*(self.extractConstantStrings(parse(self, x)) for x in xs if self.numInputs(parse(self, x)) == 1))
        return list(set(r))

    def inferType(self, v):
        if type(v) is str:
            return "str"
        elif type(v) is int:
            return "int"
        elif type(v) is bool:
            return "bool"
        else:
            assert False

import stringprogs
stringdsl = StringDsl(progs=stringprogs.stringprogs)

def test():
    from softcheck import softcheck
    from bustle import bustle, propertySignatureSize
    from bustle import probe_bustle, PSol_cost
    from stringprops import llProps
    from model import Rater, loadModel
    import torch
    from llm import bustle_llm, generateDeltaWeightAll, generateOpWeightTable

    str2 = ("str", ("str",))
    str3 = ("str", ("str", "str"))

    sl = StringDsl(progs=[])
    slx = StringDsl(True, progs=[])
    MsInit = {
        (("str",), "str", "str"): Rater(
            2 * propertySignatureSize(("str",), "str", llProps)
        )
    }
    MsTrained = loadModel()

    for (desc, sl, llProps, Ms, llm_gen) in [
            ('no ML', sl, None, None, None),
            ('LLM (per problem)', sl, None, None, generateOpWeightTable),
            ('LLM', sl, None, None, generateDeltaWeightAll),
            ('init ML', sl, llProps, MsInit, None),
            ('trained ML', sl, llProps, MsTrained, None),
            ('trained ML, extended DSL', slx, llProps, MsTrained, None)
            ]:
        print(desc)

        print("test 1: ", end='')
        assert ("Left", [("input", 0), 1]) == bustle_llm(
            "Consider the problem of taking the first character of a string.",
            llm_gen,
            sl, str2, [["hello", "world"]], ["h", "w"], llProps, Ms,
            print_stats=True
        )
        print("test 2: ", end='')
        assert ("Right", [("input", 0), 1]) == bustle_llm(
            "Consider the probelm of taking the last character of a string.",
            llm_gen,
            sl, str2, [["hello", "world"]], ["o", "d"], llProps, Ms,
            print_stats=True
        )
        print("test 3: ", end='')
        assert ("Concatenate", [("input", 0), ("input", 1)]) == bustle_llm(
            "Consider the problem of concatenating two strings.",
            llm_gen,
            sl, str3,
            [["hello", "world"], ["you", "domination"]],
            ["helloyou", "worlddomination"],
            llProps, Ms,
            print_stats=True
        )
        print("test 4: ", end='')
        assert (
            "Concatenate", [("input", 0), ("Concatenate", [" ", ("input", 1)])],
        ) == bustle_llm(
            "Consider the problem of concatenating two strings with a space in between.",
            llm_gen,
            sl, str3,
            [["hello", "world"], ["you", "domination"]],
            ["hello you", "world domination"],
            llProps, Ms,
            print_stats=True
        )
        if llProps is not None:
            continue
        print("test 5: ", end='')
        softcheck(
            bustle_llm(
                "Consider the problem of concatenating the first character and the last character of a string.",
                llm_gen,
                sl, str2,
                [["hello", "world", "domination", "yes"]], ["ho", "wd", "dn", "ys"],
                llProps, Ms,
                print_stats=True
            ), [
                ("Concatenate", [("Left", [("input", 0), 1]), ("Right", [("input", 0), 1])]),
                ('Replace', [('input', 0), 1, ('Minus', [0, 2]), ''])
            ])

        # PSol = probe_bustle(
        #         sl, str2,
        #         [["hello", "world", "domination", "yes"]], ["ho", "wd", "dn", "ys"],
        #         llProps, Ms,
        #         llm("Consider the problem of concatenating the first character and the last character of a string.") if llm is not None else None,
        #         print_stats=True, N=5
        #     )[1]
        # cost = PSol_cost(sl, PSol)
        # print(cost)

if __name__ == "__main__":
    print("running tests...")
    test()
    print("done")
