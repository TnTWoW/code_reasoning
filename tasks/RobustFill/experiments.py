from dslparser import parse, printer
from stringdsl import stringdsl
from bustle import bustle, propertySignatureSize
import stringprogs
from stringprops import llProps
from model import Rater, loadModel
import torch

from absl import app
from absl import flags
from absl import logging

WITH_ML = flags.DEFINE_boolean('ml', True, 'use ML')

def main(_):
    with_ml = WITH_ML.value
    sl = stringdsl
    #prog = stringprogs.stringprogs[1]
    #prog = 'IF(EXACT(LEFT(var_0, 1), "-"), var_0, CONCATENATE("+", var_0))'
    prog = 'CONCATENATE("+", var_0)'
    prog = 'IF(EXACT(var_0, "-"), var_0, CONCATENATE("+", var_0))'
    prog = 'IF(EXACT(var_0, "-"), "-", "+")'
    prog = 'IF(EXACT(var_0, "-"), var_0, CONCATENATE("+", var_0))'
    prog = 'IF(EXACT(LEFT(var_0, 1), "-"), var_0, CONCATENATE("+", var_0))'
    #prog = 'Concatenate(Left("+", Minus(0, Find("-", var_0))), var_0)'
    #prog = 'Concatenate(Replace("+", Find("-", var_0), 1, ""), var_0)'
    ast = parse(sl, prog)
    #I = [stringprogs.input]
    I = [["1", "2.0", "hello", "-1", "-1.0", "-", "hello-you"]]
    O = sl.evalIO(ast, I)
    str2 = ("str", ("str",))
    MsInit = {
        (("str",), "str", "str"): Rater(
            2 * propertySignatureSize(("str",), "str", llProps)
        )
    }
    MsTrained = loadModel()
    if with_ml:
        Ms = MsTrained
        llp = llProps
    else:
        Ms = None
        llp = None
    ast_found = bustle(sl, str2, I, O, llp, Ms, print_stats=True)
    prog_found = printer(sl, ast_found)
    print(prog_found)

if __name__ == '__main__':
    app.run(main)
