from bustle import propertySignature
from generate_dataset import subexpressions
from dslparser import parse
from stringdsl import stringdsl
import stringprogs
from stringprops import llProps
import torch

from absl import app
from absl import flags
from absl import logging

SUB1 = flags.DEFINE_string('sub1', '"s"', 'subexpression 1')
SUB2 = flags.DEFINE_string('sub2', '"t"', 'subexpression 2')
VT = flags.DEFINE_string("typ", "str", 'type of subexpressions')

def how_discriminating(dsl, sub1, sub2, Vt, Ot, I, O):
    V1 = dsl.evalIO(sub1, I)
    s_vo1 = propertySignature((V1,), (Vt,), O, Ot, llProps)
    V2 = dsl.evalIO(sub2, I)
    s_vo2 = propertySignature((V2,), (Vt,), O, Ot, llProps)
    return (s_vo1 - s_vo2)


def main(_):
    dsl = stringdsl
    prog = "REPLACE(LOWER(var_0), 1, 1, UPPER(LEFT(var_0, 1)))"
    ast = parse(dsl, prog)
    sub1 = parse(dsl, SUB1.value)
    sub2 = parse(dsl, SUB2.value)
    I = stringprogs.all_inputs(1)
    O = dsl.evalIO(ast, I)
    Ot = "str"
    Vt = VT.value
    print(how_discriminating(dsl, sub1, sub2, Vt, Ot, I, O))

if __name__ == '__main__':
    app.run(main)
