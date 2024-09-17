from dslparser import printer
from llm_outlines import gen

debug = False
choices = ["A", "B", "C", "D", "E"]
choice_stats = {c:0 for c in choices}

def grading_instructions():
    prompt = ""
    prompt += "Answer with one of the grades A, B, C, D, E.\n"
    prompt += "A means very likely to appear.\n"
    prompt += "B means somewhat likely to appear.\n"
    prompt += "C means somewhat unlikely to appear.\n"
    prompt += "D means unlikely to appear.\n"
    prompt += "E means very unlikely to appear.\n"
    prompt += "So answer with one of the grades A, B, C, D, E.\n"
    prompt += "Your grade is:"
    return prompt

def generateOpWeightTable(desc, dsl, It, Ot, I, O):
    table = {}
    prompt = dsl.desc(sayOps=True)
    prompt += "\n"
    n = len(O)
    prompt += desc
    prompt += "\n"
    prompt += f"You have to generate this function `f` with input/output as follows on {n} examples:\n"
    prompt += dsl.io_print(I, O)
    preambule = prompt
    for op in dsl.Ops:
        prompt = preambule
        prompt += f"Can you guess whether the operation `{op}` is used in the solution to the function `f` to synthesize?\n"
        prompt += grading_instructions()
        if debug:
            print("PROMPT:")
            print(prompt)
        r = gen(prompt, choices)
        if debug:
            print(r)
        table[dsl.toOp(op)] = choices.index(r)
    def inner(dsl, It, Ot, Vt, I, O, V):
        (e, _) = V
        if type(e) is tuple or type(e) is list:
            (op, _) = e
            if op == 'input':
                return 0
            return table[dsl.toOp(op)]
        else:
            return 0
    print('Op weight table is', table)
    return inner

def generateDeltaWeightAll(desc, dsl, It, Ot, I, O):
    return generateDeltaWeight(desc)

def generateDeltaWeight(desc):
    def inner(dsl, It, Ot, Vt, I, O, V):
        global choice_stats
        prompt = dsl.desc()
        prompt += "\n"
        n = len(O)
        prompt += desc
        prompt += "\n"
        prompt += f"You have to generate this function `f` with input/output as follows on {n} examples:\n"
        prompt += dsl.io_print(I, O)
        expr = printer(dsl, V[0])
        res = V[1]
        outres = O
        prompt += f"Can you guess whether the DSL expression `{expr}` is likely to appear as a sub-expression in a solution in our DSL to the function `f` to synthesize. The expression has the following output on the inputs above: {res}, while the final solution should have the following output: {outres}.\n"
        prompt += grading_instructions()
        if debug:
            print("PROMPT:")
            print(prompt)
        r = gen(prompt, choices)
        if debug:
            print(r)
        choice_stats[r] += 1
        return choices.index(r)
    return inner

def bustle_llm(desc, gen, dsl, typeSig, I, O, llProps=None, Ms=None, **kwargs):
    from bustle import bustle
    Ot, It = typeSig
    llm = gen(desc, dsl, It, Ot, I, O) if gen is not None else None
    return bustle(dsl, typeSig, I, O, llProps, Ms, llm=llm, **kwargs)

def test():
    from arithdsl import ArithDsl
    from bustle import bustle   
    al = ArithDsl()
    int2 = ("int", ("int",))
    for gen in [generateOpWeightTable, generateDeltaWeightAll]:
        r = bustle_llm(
            'Consider the problem of substracting one, then doubling.',
            gen,
            al, int2, [[1, 2, 3, 4]], [0, 2, 4, 6])
        print(printer(al, r))
        print(choice_stats)

if __name__ == "__main__":
    print("running tests...")
    debug = True
    test()
    print("done")
