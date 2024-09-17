from dslparser import printer

def makeOpCache(Ops):
    cache = {}
    for op in Ops:
        cache[op.lower()] = op
    return cache

class Dsl:
    def __init__(self):
        self.opCache = makeOpCache(self.Ops)

    def eval(self, x, inp):
        if type(x) is tuple or type(x) is list:
            if x[0] == "input":
                return inp[x[1]]
            else:
                return self.execute(x[0], [self.eval(e, inp) for e in x[1]])
        else:
            return x

    def evalIO(self, exp, inps):
        return [self.eval(exp, inp) for inp in zip(*inps)]

    def numInputs(self, x):
        if type(x) is tuple or type(x) is list:
            if x[0] == "input":
                return x[1] + 1
            else:
                return max(self.numInputs(e) for e in x[1])
        else:
            return 0

    def size(self, x):
        if type(x) is tuple or type(x) is list:
            if x[0] == "input":
                return 1
            else:
                return 1 + sum(self.size(e) for e in x[1])
        else:
            return 1

    def all_ops(self, x):
        if type(x) is tuple or type(x) is list:
            if x[0] == "input":
                return set()
            else:
                return set({x[0]}).union(*[self.all_ops(e) for e in x[1]])
        else:
            return set()

    def isOp(self, token):
        return token.lower() in self.opCache

    def toOp(self, token):
        return self.opCache[token.lower()]

    def arity(self, op):
        return len(self.argtypes(op))

    def argtypes(self, op):
        _, ts = self.types(op)
        return ts

    def returntype(self, op):
        t, _ = self.types(op)
        return t

    def desc(self, sayOps=False):
        prompt = ""
        ops = ",".join([f"`{op}`" for op in self.Ops])
        (V0, I, O) = self.Ex
        n = len(O)
        expr = printer(self, V0)
        prompt = f"The {self.Name} has operations {ops}.\n"
        prompt += f"As an example, consider a function with the following input/output on {n} examples:\n"
        prompt += self.io_print(I, O)
        prompt += f"A possible solution expression is {expr}.\n"
        if sayOps:
            prompt += f"So the operations in that expression should be graded as very likely to occur: {self.all_ops(V0)}.\n"
        return prompt

    def io_print(self, I, O):
        prompt = ""
        n = len(O)
        for i in range(0, n):
            ins = [repr(x[i]) for x in I]
            out = repr(O[i])
            prompt += f"{i}. ({','.join(ins)}) -> {out}\n"
        return prompt


def test():
    from arithdsl import ArithDsl

    al = ArithDsl()
    assert 2 == al.eval(("add", [("input", 0), 1]), [1])
    assert [2, 3, 4] == al.evalIO(("add", [("input", 0), 1]), [[1, 2, 3]])
