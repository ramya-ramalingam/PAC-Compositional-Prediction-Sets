from lisp import *
import types

def get_grammar():
    rules = []
    symb = 'int'

    # higher-order combinators
    rules.append(('list-int', 'map', 'int->int', 'list-int'))
    rules.append(('list-int', 'filter', 'int->bool', 'list-int'))
    rules.append(('int', 'fold', 'int->int->int', 'list-int', 'const-int'))

    # function applicaion (currying)
    rules.append(('int->int', 'curry', 'int->int->int', 'const-int'))
    rules.append(('int->bool', 'curry', 'int->int->bool', 'const-int'))

    # functions
    rules.append(('int->int->int', 'plus'))
    rules.append(('int->int->bool', 'ge'))

    # inputs
    rules.append(('list-int', 'input'))
    
    # constants
    rules.append(('const-int', '0'))
    rules.append(('const-int', '1'))
    rules.append(('const-int', '2'))

    return Grammar(rules, symb)

def map_fn(args, inp):
    f, xs = args
    return [f(x) for x in xs]

def fold_fn(args, inp):
    f, xs, b = args
    for x in xs:
        b = f(x, b)
    return b

def filter_fn(args, inp):
    f, xs = args
    return [x for x in xs if f(x)]

def apply_fn(args, inp):
    f, x = args
    return f(x)

def curry_fn(args, inp):
    f, x = args
    return lambda y: f(x, y)

def input_fn(args, inp):
    if len(args) != 0:
        raise Exception()
    return inp

def plus_fn(args, inp):
    return lambda x, y: x + y

def ge_fn(args, inp):
    return lambda x, y: x >= y

def le_fn(args, inp):
    return lambda x, y: x <= y

def input_fn(args, inp):
    return inp

def get_fns():
    return {
        # higher order combinators
        'map': map_fn,
        'filter': filter_fn,
        'fold': fold_fn,

        # function application
        'apply': apply_fn,
        'curry': curry_fn,

        # functions
        'plus': plus_fn,
        'ge': ge_fn,
        'le': le_fn,

        # inputs
        'input': input_fn,

        # constants
        '0': lambda args, inp: 0,
        '1': lambda args, inp: 1,
        '2': lambda args, inp: 2,
        '100': lambda args, inp: 100,
        'MAX_WIDTH': lambda args, inp: 640,
        'MAX_HEIGHT': lambda args, inp: 480,
    }

if __name__ == '__main__':
    # Parameters
    depth = 4

    # Inputs
    grammar = get_grammar()
    fns = get_fns()
    inp = [2.0, 1.0, 2.0, 4.0]
    # Enumeration
    exprs = enum(depth, grammar)
    for expr in exprs:
        outp = execute(expr, inp, fns)
        print('{} {} = {}'.format(expr, inp, outp))
    print('{} total expressions'.format(len(exprs)))
    print()

    # Parsing and 
    program = '(fold plus (filter (curry ge 2) input) (fold plus input 0))'
    expr = parse(program)
    outp = execute(expr, inp, fns)
    # outp, outp_dict = execute_with_intermediate_values(expr, inp, fns, dict())
    print('Program: ' + program)
    print('Input: ' + str(inp))
    print('Output: ' + str(outp))

