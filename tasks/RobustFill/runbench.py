from dslparser import parse, printer
from stringdsl import stringdsl
from bustle import bustle, propertySignatureSize
from bustle import probe_bustle, PSol_cost
import stringprogs
from stringprops import llProps
from model import Rater, loadModel
import torch

from absl import app
from absl import flags
from absl import logging

START = flags.DEFINE_integer('start', 0, 'starting index')

def main(_):
    sl = stringdsl
    I = stringprogs.all_inputs(1)
    #I = [["1", "2.0", "hello", "-1", "-1.0", "-", "hello-you"]]
    Ms = loadModel()
    
    str2 = ("str", ("str",))

    count = START.value
    asts = [parse(sl, prog) for prog in stringprogs.stringprogs]
    asts = [ast for ast in asts if sl.numInputs(ast)==1]
    for ast in asts[START.value:]:
        O = sl.evalIO(ast, I)
        prog = printer(sl, ast)
        print(str(count), 'bench', prog)
        count += 1

        PSol = probe_bustle(sl, str2, I, O, llProps, Ms, N=5)[1]
        print(PSol_cost(sl, PSol))

        ast_found = bustle(sl, str2, I, O, llProps, Ms, print_stats=True)
        prog_found = printer(sl, ast_found)
        print(prog_found)

if __name__ == '__main__':
    app.run(main)
