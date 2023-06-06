from lisp import * 
import lisp_ex as concrete

def alpha(x):
    # Abstraction function
    if isinstance(x, bool):
        # checking for boolean type
        return int(x)
    elif isinstance(x, int):
        # checking for int type
        return (x,x)
    else:
        # otherwise, it will be a list
        return [(a, 1) for a in x]

def gamma(x):
    if isinstance(x, int):
        return bool(x)
    elif x == '?':
        return {0, 1}
    elif isinstance(x, tuple):
        val = {}
        for i in range(x[0], x[1] + 1):
            val.add(i)
        return val
    else:
        # otherwise, it will be a list
        return [a for (a, b) in x if b != 0]
    # Concretization function

# Used primarily in fold definition
def join(a1, a2):
    # Join of two abstract values
    # We assume for simplicity that we will only ever join integer values

    (l1, u1) = a1
    (l2, u2) = a2
    return (min(l1, l2), max(u1, u2))

# Arithmetic Operators 

def plus_curry_abs(args, inp):
    return lambda x, y: (x[0] + y[0], x[1] + y[1])

def ge_curry_abs(args, inp):
    return lambda x, y: 1 if x[0] >= y[1] else (0 if y[0] > x[1] else '?')

def le_curry_abs(args, inp):
    return lambda x, y: 1 if x[0] <= y[1] else (0 if y[0] < x[1] else '?')

def plus_fn_abs(args, inp):
    (l1, u1), (l2, u2) = args
    return (l1 + l2, u1 + u2)

def minus_fn_abs(args, inp):
    (l1, u1), (l2, u2) = args  
    return (l1 - u2, u2 - l1)

def ge_fn_abs(args, inp):
    (l1, u1), (l2, u2) = args
    if l1 >= u2:
        return 1
    elif l2 > u1:
        return 0
    else:
        return '?'

# Boolean operators

def or_fn_abs(args, inp):
    b1, b2 = args

    if b1 == '?':
        if b2 == 1:
            return 1
        else:
            return '?'
    elif b2 == '?':
        if b1 == 1:
            return 1
        else:
            return '?'
    else:
        return b1 or b2

def and_fn_abs(args, inp):
    b1, b2 = args
    if b1 =='?':
        if b2 == 1:
            return '?'
        else:
            return 0
    elif b2 == '?':
        if b1 == 1:
            return '?'
        else:
            return 0
    else:
        return b1 and b2

# Helper function for filter
def and_fn_helper(b1, b2):
    #b1, b2 = args

    if b1 == '?':
        if b2 == 0:
            return 0
        else:
            return '?'
    elif b2 == '?':
        if b1 == 0:
            return 0
        else:
            return '?'
    else:
        return int(b1 and b2)

def not_fn_abs(args, inp):
    b = args

    if b == '?':
        return '?'
    else:
        return int(not b)

# Apply, curry and input

def apply_fn(args, inp):
    f, x = args
    return f(x)

def curry_fn(args, inp):
    f, x = args
    return lambda y: f(x, y)

def input_fn(args, inp):
    return inp

# List combinators

def map_fn_abs(args, inp):
    f, xs = args
    return [(f(x), b) for (x, b) in xs]

def filter_fn_abs(args, inp):
    f, xs = args
    return [(x, and_fn_helper(b, f(x))) for (x, b) in xs if f(x) != 0]

def fold_fn_abs(args, inp):
    f, xs, b = args
    if len(xs) == 0:
        return b
    else:
        # Check boolean flag of first element of abstract list
        (x, flag) = xs[0]
        if flag == 1: 
            return fold_fn_abs((f, xs[1:], f(x, b)), inp)
        else:
            # In this case, b = '?'
            return join(
                fold_fn_abs((f, xs[1:], b), inp),
                fold_fn_abs((f, xs[1:], f(x, b)), inp)
            )

def f0(args, inp):
    return (0, 0)

def f1(args, inp):
    return lambda x, y: (1, 1)

def get_fns_abs():
    return {
        # higher order combinators
        'map': map_fn_abs,
        'filter': filter_fn_abs,
        'fold': fold_fn_abs,

        # function application
        'apply': apply_fn,
        'curry': curry_fn,

        # functions
        'plus': plus_curry_abs,
        'plus_uncurried': plus_fn_abs,
        'ge': ge_curry_abs,
        'ge_uncurried': ge_fn_abs,
        'le': le_curry_abs,

        # inputs
        'input': input_fn,

        # constant functions
        'f0': f0,
        'f1': f1,

        # constants
        '-100': lambda args, inp: (-100, -100),
        '100': lambda args, inp: (100, 100),
        '0': lambda args, inp: (0, 0),
        '1': lambda args, inp: (1, 1),
        '2': lambda args, inp: (2, 2),
        '10': lambda args, inp: (10, 10),
        '1,2': lambda args, inp: (1, 2),
        '1,3': lambda args, inp: (1, 3),
        '3,4': lambda args, inp: (3, 4),
        '3,5': lambda args, inp: (3, 5),
        'MAX_WIDTH': lambda args, inp: (640, 640),
        'MAX_HEIGHT': lambda args, inp: (480, 480),
    }

if __name__ == '__main__':

    fns = get_fns_abs()

    # Arithmetic operators test
    expr0 = parse('(plus_uncurried (1) (3,4))')
    inp0 = None
    outp0 = execute(expr0, inp0, fns)
    print('{} {} = {}'.format(expr0, inp0, outp0))

    expr1 = parse('(plus_uncurried (1,3) (3,4))')
    inp1 = None
    outp1 = execute(expr1, inp1, fns)
    print('{} {} = {}'.format(expr1, inp1, outp1))

    expr2 = parse('(ge_uncurried (1,2) (3,5))')
    inp2 = None
    outp2 = execute(expr2, inp2, fns)
    print('{} {} = {}'.format(expr2, inp2, outp2))

    expr3 = parse('(ge_uncurried (3,4) (3,5))')
    inp3 = None
    outp3 = execute(expr3, inp3, fns)
    print('{} {} = {}'.format(expr3, inp3, outp3))

    expr4 = parse('(ge_uncurried (1,3) (3,5))')
    inp4 = None
    outp4 = execute(expr4, inp4, fns)
    print('{} {} = {}'.format(expr4, inp4, outp4))

    expr5 = parse('(ge_uncurried (3,5) (1,3))')
    inp5 = None
    outp5 = execute(expr5, inp5, fns)
    print('{} {} = {}'.format(expr5, inp5, outp5))

    # Map test
    expr6 = parse('(map (curry plus 1) input)')
    inp6 = [((1,3), 1), ((5,6), '?')]
    outp6 = execute(expr6, inp6, fns)
    print('{} {} = {}'.format(expr6, inp6, outp6))

    # Filter test
    expr7 = parse('(filter (curry ge 2) input)')
    inp7 = [((3,6), '?'), ((0,1), 1), ((1,4), 1)]
    outp7, outp_dict = execute_with_intermediate_values(expr7, inp7, fns, dict())
    print(outp7)
    print(outp_dict)
    print('{} {} = {}'.format(expr7, inp7, outp7))

    # Fold test
    expr8 = parse('(fold plus input 0)')
    inp8 = [((3,6), '?'), ((0,1), 1), ((1,4), 1)]
    outp8 = execute(expr8, inp8, fns)
    print('{} {} = {}'.format(expr8, inp8, outp8))

    expr9 = parse('(fold plus input 0)')
    inp9 = [((1,2), 1), ((2,4), '?'), ((1,4), 1)]
    outp9 = execute(expr9, inp9, fns)
    print('{} {} = {}'.format(expr9, inp9, outp9))

    # Large expression test
    expr10 = parse('(fold plus (filter (curry ge 2) (map (curry plus 2) input)) 0)')
    inp10 = [((-3,-1), 1), ((2,2), 1), ((3,5), 1), ((-2,6), 1)]
    outp10 = execute(expr10, inp10, fns)
    # outp10, outp_dict10 = execute_with_intermediate_values(expr10, inp10, fns, dict())
    print('{} {} = {}'.format(expr10, inp10, outp10))

