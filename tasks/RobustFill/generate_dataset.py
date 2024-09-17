from bustle import bustle, propertySignature
from stringdsl import stringdsl
import stringprogs
from stringprops import llProps
from model import loadModel
import itertools
import random
import string
from rich.progress import (
    track,
    BarColumn,
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import torch


def subexpressions(exp):
    r = []
    r += [exp]
    if type(exp) is tuple or type(exp) is list:
        if exp[0] == "input":
            return r
        r.extend(itertools.chain(*(subexpressions(s) for s in exp[1])))
    return r


def handle_expression(exp, all_search, search, dsl, inp):
    es = subexpressions(exp)
    pos_exp = random.choice(es)
    pos = dsl.evalIO(pos_exp, inp)
    neg = random.choice(all_search)
    while neg[0] in es:
        neg = random.choice(all_search)
    return (pos, neg[1])


def select_expression(all_search, search, dsl, inp):
    sample = random.choice(search)
    return build_sample(sample, all_search, search, dsl, inp)


def build_sample(sample, all_search, search, dsl, inp):
    exp = sample[0]
    o = sample[1]
    (pos, neg) = handle_expression(exp, all_search, search, dsl, inp)
    return ((inp, pos, o), (inp, neg, o))


def batch_dataset(dataset, llProps):
    desc = "Generating property signatures and Batching dataset ..."
    It = ("str",)
    Ot = "str"
    Ms = {}

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        for i, sample in enumerate(progress.track(dataset, description=desc)):
            pos, neg = sample
            for (ex, valence) in ((pos, 1.0), (neg, 0.0)):
                (I, V, O) = ex
                Vt = stringdsl.inferType(V[0])
                key = (It, Ot, Vt)
                s1 = propertySignature(I, It, O, Ot, llProps)
                s2 = propertySignature([V], (Vt,), O, Ot, llProps)
                s = torch.cat([s1, s2])

                if key not in Ms:
                    Ms[key] = [(s, torch.tensor([valence]))]
                else:
                    Ms[key].append((s, torch.tensor([valence])))
        return Ms


typ = ("str", ("str",))
# charset = string.printable[:-6]
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.-+/"


def generate_input(N=3, LB=5, UB=8):
    # TODO: maybe generate more interesting inputs that satisfy constraints
    #   such as index within range
    # numbers?
    # dates?
    # URLs?
    # words?
    inp = []
    for i in [random.randint(LB, UB) for _ in range(N)]:
        inp.append("".join([random.choice(charset) for _ in range(i)]))
    inp = inp + stringprogs.input
    return [inp]


def run_bustle_cache(dsl, typ, inp, N, src="cache_bustle.pt"):
    try:
        return torch.load(src)
    except FileNotFoundError:
        x = run_bustle(dsl, typ, inp, N)
        torch.save(x, src)
        return x

def run_bustle(dsl, typ, inp, N):
    Ms = loadModel()
    all_search = bustle(dsl, typ, inp, ["dummy" for _ in inp[0]], N=N, print_stats=True, Ms=Ms, llProps=llProps)
    search = [v for i in range(2, N) for v in all_search[i]["str"]]
    search_bool = [v for i in range(2, N) for v in all_search[i]["bool"]]
    search_int = [v for i in range(2, N) for v in all_search[i]["int"]]
    all_search = search + search_bool + search_int
    return search, all_search


def generate_dataset():
    return generate_dataset_cheat()


def generate_io_dataset():
    from dslparser import parse
    import json

    dsl = stringdsl
    one_input_expressions = [prog for prog in stringprogs.stringprogs if 'var_1' not in prog]

    progs = [parse(dsl, e) for e in one_input_expressions]

    data = {}
    random_inputs = generate_input(10, 2, 10)
    outputs = [dsl.evalIO(e, random_inputs) for e in progs]
    outputs = [o for o in outputs if type(o[0]) is str]

    inps = [random_inputs[0] for _ in outputs]
    for idx, (pro, inp, out) in enumerate(zip(one_input_expressions, inps, outputs)):
        data[idx] = {'program':pro, 'input': inp, 'output': out}

    with open('../../data/RobustFill.json', mode='w') as json_file:
        json.dump(data, json_file, indent=4)
    return data

def generate_dataset_cheat(only=None):
    from dslparser import parse

    dsl = stringdsl
    progs = [parse(dsl, prog) for prog in stringprogs.stringprogs]
    progs1 = [prog for prog in progs if dsl.numInputs(prog) == 1]
    if only is not None:
        progs1 = [prog for i,prog in enumerate(progs1) if i in only]
    exps = list(itertools.chain(*(subexpressions(prog) for prog in progs1)))

    N = 12
    N_search = 1
    N_selected = 4000
    data = []
    for i in range(N_search):
        inp = [stringprogs.input]
        print("")
        print("BUSTLE")
        search, all_search = run_bustle_cache(dsl, typ, inp, N)
        samples = [(e, dsl.evalIO(e, inp)) for e in exps]
        all_search = all_search + samples
        samples = [(e, o) for (e, o) in samples if type(o[0]) is str]
        print("")
        for sample in track(samples, description="Samples ..."):
            for i in range(N_selected):
                data.append(build_sample(sample, all_search, search, dsl, inp))
        for j in track(range(100*N_selected), description="Extra samples ..."):
            data.append(select_expression(all_search, search, dsl, inp))
        print()

    random.shuffle(data)
    return data


def generate_dataset_old(dsl=stringdsl):
    data = []
    N = 7
    N_search = 5
    N_selected = 1000
    for i in range(N_search):
        inp = generate_input()
        search, all_search = run_bustle(dsl, typ, inp, N)
        for j in range(N_selected):
            data.append(select_expression(all_search, search, dsl, inp))
        # for sample in search:
        #    data.append(build_sample(sample, all_search, dsl, inp))
    return data

if __name__ == '__main__':
    dataset = generate_io_dataset()