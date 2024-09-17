from dsl import Dsl


class ArithDsl(Dsl):
    def __init__(self):
        Ops = ["add", "mul", "div", "neg", "lt", "if"]
        Types = ["int", "bool"]
        Name = "arithmetic expression DSL"

        self.Ops = Ops
        self.Types = Types
        self.Name = Name
        self.Ex = (("add", [("input", 0), 1]), [[1, 2, 3]], [2, 3, 4])
        super().__init__()

    def execute(self, op, args):
        if op == "add":
            return args[0] + args[1]
        elif op == "mul":
            return args[0] * args[1]
        elif op == "div":
            #return args[0] // args[1]
            return args[0] / args[1]
        elif op == "neg":
            return -args[0]
        elif op == "lt":
            return args[0] < args[1]
        elif op == "if":
            return args[1] if args[0] else args[2]
        else:
            assert False

    def types(self, op):
        if op in ["neg"]:
            return ("int", ("int",))
        elif op in ["add", "mul", "div"]:
            return ("int", ("int", "int"))
        elif op in ["lt"]:
            return ("bool", ("int", "int"))
        elif op in ["if"]:
            return ("int", ("bool", "int", "int"))
        else:
            assert False

    def extractConstants(self, I, O, It, Ot):
        cs = [0, 1]
        return [("int", (c, [c for _ in range(len(O))])) for c in cs]

    def inferType(self, v):
        if type(v) is int:
            return "int"
        elif type(v) is bool:
            return "bool"
        else:
            assert False
