#!/usr/bin/env python2
"""
Usage:
    generate_io_samples.py [options] PROGRAM_TEXT

Example:
    $ ./generate_io_samples.py "a <- [int] | b <- int | c <- TAKE b a | d <- COUNT isEVEN c | e <- TAKE d a"

Options:
    -h --help             Show this screen.
    -N --number NUM       Number of I/O examples to generate. [default: 5]
    -L --length NUM       Length of generated lists. [default: 10]
    -V --value-range NUM  Range of values. [default: 512]
"""

import os
import sys
from collections import namedtuple, defaultdict
from math import ceil, sqrt

import numpy as np
from docopt import docopt

Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])
Program = namedtuple('Program', ['src', 'ins', 'out', 'fun', 'bounds'])


# HELPER FUNCTIONS
def type_to_string(t):
    if t == int:
        return 'int'
    if t == [int]:
        return '[int]'
    if t == bool:
        return 'bool'
    if t == [bool]:
        return '[bool]'
    raise ValueError('Type %s cannot be converted to string.' % t)


def scanl1(f, xs):
    if len(xs) > 0:
        r = xs[0]
        for i in range(len(xs)):
            if i > 0:
                r = f.fun(r, xs[i])
            yield r


##### Bound helpers:
def SQR_bounds(A, B):
    l = max(0, A)  # inclusive lower bound
    u = B - 1  # inclusive upper bound
    if l > u:
        return [(0, 0)]
    # now 0 <= l <= u
    # ceil(sqrt(l))
    # Assume that if anything is valid then 0 is valid
    return [(-int(sqrt(u)), ceil(sqrt(u + 1)))]


def MUL_bounds(A, B):
    return SQR_bounds(0, min(-(A + 1), B))


def scanl1_bounds(l, A, B, L):
    if l.src == '+' or l.src == '-':
        return [(A / L + 1, B / L)]
    elif l.src == '*':
        return [(int((max(0, A) + 1) ** (1.0 / L)), int((max(0, B)) ** (1.0 / L)))]
    elif l.src == 'MIN' or l.src == 'MAX':
        return [(A, B)]
    else:
        raise Exception('Unsupported SCANL1 lambda, cannot compute valid input bounds.')

##### LINQ language:
def get_language(V):
    Null = V
    lambdas = [
        Function('IDT', (int, int), lambda i: i, lambda args: [args]),

        Function('INC', (int, int), lambda i: i + 1, lambda args: [(args[0], args[1] - 1)]),
        Function('DEC', (int, int), lambda i: i - 1, lambda args: [(args[0] + 1, args[1])]),
        Function('SHL', (int, int), lambda i: i * 2, lambda args: [((args[0] + 1) / 2, args[1] / 2)]),
        Function('SHR', (int, int), lambda i: int(float(i) / 2), lambda args: [(2 * args[0], 2 * args[1])]),
        Function('doNEG', (int, int), lambda i: -i, lambda args: [(-args[1] + 1, -args[0] + 1)]),
        Function('MUL3', (int, int), lambda i: i * 3, lambda args: [((args[0] + 2) / 3, args[1] / 3)]),
        Function('DIV3', (int, int), lambda i: int(float(i) / 3), lambda args: [args]),

        Function('MUL4', (int, int), lambda i: i * 4, lambda args: [((args[0] + 3) / 4, args[1] / 4)]),
        Function('DIV4', (int, int), lambda i: int(float(i) / 4), lambda args: [args]),
        Function('SQR', (int, int), lambda i: i * i, lambda args: SQR_bounds(args[0], args[1])),
        #Function('SQRT',    (int, int),          lambda i: int(sqrt(i)),                              lambda (A, B): [(max(0, A*A), B*B)]),

        Function('isPOS', (int, bool), lambda i: i > 0, lambda args: [args]),
        Function('isNEG', (int, bool), lambda i: i < 0, lambda args: [args]),
        Function('isODD', (int, bool), lambda i: i % 2 == 1, lambda args: [args]),
        Function('isEVEN', (int, bool), lambda i: i % 2 == 0, lambda args: [args]),

        Function('+', (int, int, int), lambda i, j: i + j, lambda args: [(args[0] / 2 + 1, args[1] / 2)]),
        Function('-', (int, int, int), lambda i, j: i - j, lambda args: [(args[0] / 2 + 1, args[1] / 2)]),
        Function('*', (int, int, int), lambda i, j: i * j, lambda args: MUL_bounds(args[0], args[1])),
        Function('MIN', (int, int, int), lambda i, j: min(i, j), lambda args: [args]),
        Function('MAX', (int, int, int), lambda i, j: max(i, j), lambda args: [args]),
    ]

    LINQ = [
        Function('REVERSE', ([int], [int]), lambda xs: list(reversed(xs)), lambda args: [(args[0], args[1])]),
        Function('SORT', ([int], [int]), lambda xs: sorted(xs), lambda args: [(args[0], args[1])]),
        Function('TAKE', (int, [int], [int]), lambda n, xs: xs[:n], lambda args: [(0, args[2]), (args[0], args[1])]),
        Function('DROP', (int, [int], [int]), lambda n, xs: xs[n:], lambda args: [(0, args[2]), (args[0], args[1])]),
        Function('ACCESS', (int, [int], int), lambda n, xs: xs[n] if n >= 0 and len(xs) > n else Null,
                 lambda args: [(0, args[2]), (args[0], args[1])]),
        Function('HEAD', ([int], int), lambda xs: xs[0] if len(xs) > 0 else Null, lambda args: [(args[0], args[1])]),
        Function('LAST', ([int], int), lambda xs: xs[-1] if len(xs) > 0 else Null, lambda args: [(args[0], args[1])]),
        Function('MINIMUM', ([int], int), lambda xs: min(xs) if len(xs) > 0 else Null, lambda args: [(args[0], args[1])]),
        Function('MAXIMUM', ([int], int), lambda xs: max(xs) if len(xs) > 0 else Null, lambda args: [(args[0], args[1])]),
        Function('SUM', ([int], int), lambda xs: sum(xs), lambda args: [(args[0] / args[2] + 1, args[1] / args[2])]),
    ] + \
        [Function(
            'MAP ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(map(l.fun, xs)),
            lambda args, l=l: l.bounds((args[0], args[1]))
        ) for l in lambdas if l.sig == (int, int)] + \
        [Function(
            'FILTER ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(filter(l.fun, xs)),
            lambda args, l=l: [(args[0], args[1])],
        ) for l in lambdas if l.sig == (int, bool)] + \
        [Function(
            'COUNT ' + l.src,
            ([int], int),
            lambda xs, l=l: len(list(filter(l.fun, xs))),
            lambda args, l=l: [(-V, V)],
        ) for l in lambdas if l.sig == (int, bool)] + \
        [Function(
            'ZIPWITH ' + l.src,
            ([int], [int], [int]),
            lambda xs, ys, l=l: [l.fun(x, y) for x, y in zip(xs, ys)],
            lambda args, l=l: l.bounds((args[0], args[1])) + l.bounds((args[0], args[1])),
        ) for l in lambdas if l.sig == (int, int, int)] + \
       [Function(
           'SCANL1 ' + l.src,
           ([int], [int]),
           lambda xs, l=l: list(scanl1(l, xs)),
           lambda args, l=l: scanl1_bounds(l, args[0], args[1], args[2]),
       ) for l in lambdas if l.sig == (int, int, int)]

    return LINQ, lambdas

def compile(source_code, V, L, min_input_range_length=0):
    """ 
    Taken in a program source code, the integer range V and the tape lengths L,
    and produces a Program.
    If L is None then input constraints are not computed.
    """

    # Source code parsing into intermediate representation
    LINQ, _ = get_language(V)
    LINQ_names = [l.src for l in LINQ]

    input_types = []
    types = []
    functions = []
    pointers = []
    for line in source_code.split('\n'):
        instruction = line[5:]
        if instruction in ['int', '[int]']:
            input_types.append(eval(instruction))
            types.append(eval(instruction))
            functions.append(None)
            pointers.append(None)
        else:
            split = instruction.split(' ')
            command = split[0]
            args = split[1:]
            # Handle lambda
            if len(split[1]) > 1 or split[1] < 'a' or split[1] > 'z':
                command += ' ' + split[1]
                args = split[2:]
            f = LINQ[LINQ_names.index(command)]
            assert len(f.sig) - 1 == len(args), "Wrong number of arguments for %s" % command
            ps = [ord(arg) - ord('a') for arg in args]
            types.append(f.sig[-1])
            functions.append(f)
            pointers.append(ps)
            assert [types[p] == t for p, t in zip(ps, f.sig)]
    input_length = len(input_types)
    program_length = len(types)

    # Validate program by propagating input constraints and check all registers are useful
    limits = [(-V, V)] * program_length
    if L is not None:
        for t in range(program_length - 1, -1, -1):
            if t >= input_length:
                lim_l, lim_u = limits[t]
                new_lims = functions[t].bounds((lim_l, lim_u, L))
                num_args = len(functions[t].sig) - 1
                for a in range(num_args):
                    p = pointers[t][a]
                    limits[pointers[t][a]] = (max(limits[p][0], new_lims[a][0]),
                                              min(limits[p][1], new_lims[a][1]))
                    #print('t=%d: New limit for %d is %s' % (t, p, limits[pointers[t][a]]))
            elif min_input_range_length >= limits[t][1] - limits[t][0]:
                print('Program with no valid inputs: %s' % source_code)
                return None

    # for t in range(input_length, program_length):
    #     print('%s (%s)' % (functions[t].src, ' '.join([chr(ord('a') + p) for p in pointers[t]])))

    # Construct executor
    my_input_types = list(input_types)
    my_types = list(types)
    my_functions = list(functions)
    my_pointers = list(pointers)
    my_program_length = program_length

    def program_executor(args):
        # print '--->'
        # for t in range(input_length, my_program_length):
        #     print('%s <- %s (%s)' % (chr(ord('a') + t), my_functions[t].src, ' '.join([chr(ord('a') + p) for p in my_pointers[t]])))

        assert len(args) == len(my_input_types)
        registers = [None] * my_program_length
        for t in range(len(args)):
            registers[t] = args[t]
        for t in range(len(args), my_program_length):
            registers[t] = my_functions[t].fun(*[registers[p] for p in my_pointers[t]])
        return registers[-1]

    return Program(
        source_code,
        input_types,
        types[-1],
        program_executor,
        limits[:input_length]
    )


def generate_IO_examples(program, N, L, V):
    """ Given a programs, randomly generates N IO examples.
        using the specified length L for the input arrays. """
    input_types = program.ins
    input_nargs = len(input_types)

    # Generate N input-output pairs
    IO = []
    for _ in range(N):
        input_value = [None] * input_nargs
        for a in range(input_nargs):
            minv, maxv = program.bounds[a]
            if input_types[a] == int:
                input_value[a] = np.random.randint(minv, maxv)
            elif input_types[a] == [int]:
                input_value[a] = list(np.random.randint(minv, maxv, size=L))
            else:
                raise Exception("Unsupported input type " + input_types[a] + " for random input generation")
        output_value = program.fun(input_value)
        IO.append((input_value, output_value))
        assert (program.out == int and output_value <= V) or (program.out == [int] and len(output_value) == 0) or (
                    program.out == [int] and max(output_value) <= V)
    return IO


if __name__ == '__main__':
    args = docopt(__doc__)
    source = args['PROGRAM_TEXT']
    number = int(args['--number'])
    length = int(args['--length'])
    value_range = int(args['--value-range'])

    source = source.replace(' | ', '\n')
    program = compile(source, V=value_range, L=length)
    samples = generate_IO_examples(program, N=number, L=length, V=value_range)
    for (inputs, outputs) in samples:
        print("%s -> %s" % (inputs, outputs))
