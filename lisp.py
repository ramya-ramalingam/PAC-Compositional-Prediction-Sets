import itertools
import types

class Expr:
    def __init__(self, name, children=[]):
        self.name = name
        self.children = children

    def __str__(self):
        if len(self.children) == 0:
            return self.name
        else:
            return '(' + self.name + ''.join([' ' + str(child) for child in self.children]) + ')'

class Grammar:
    def __init__(self, rules, start):
        self.rules = {}
        for rule in rules:
            if rule[0] not in self.rules:
                self.rules[rule[0]] = []
            self.rules[rule[0]].append(rule[1:])
        self.start = start

def parse(s):
    toks = list(reversed(s.replace('(', ' ( ').replace(')', ' ) ').split()))
    if len(toks) == 1:
        if '(' in toks[0] or ')' in toks[0]:
            raise Exception()
        return Expr(toks[0])
    else:
        if toks.pop() != '(':
            raise Exception()
        return parse_helper(toks)

def parse_helper(toks):
    name = toks.pop()
    if '(' in name or ')' in name:
        raise Exception()
    children = []
    while True:
        tok = toks.pop()
        if tok == ')':
            return Expr(name, children)
        elif tok == '(':
            children.append(parse_helper(toks))
        else:
            children.append(Expr(tok))

def enum(depth, grammar, symb=None):
    if symb is None:
        symb = grammar.start
    if depth < 0:
        return []
    exprs = []
    for rule in grammar.rules[symb]:
        if len(rule) == 1:
            exprs.append(Expr(rule[0]))
        else:
            for children in itertools.product(*[enum(depth-1, grammar, child) for child in rule[1:]]):
                exprs.append(Expr(rule[0], children))
    return exprs

def execute(expr, inp, fns):
    args = []
    for i, child in enumerate(expr.children):
        args.append(execute(child, inp, fns))
    return fns[expr.name](args, inp)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

# Functions for extracting intermediate program point values

def execute_with_intermediate_values(expr, inp, fns, outp_dict):
    args = []
    for i, child in enumerate(expr.children):
        curr_arg, _ = execute_with_intermediate_values(child, inp, fns, outp_dict)
        args.append(curr_arg)
        # outp_dict[str(child)] = curr_arg
    outp = fns[expr.name](args,inp)
    # outp_dict[str(expr)] = outp
    # if not isinstance(outp, types.FunctionType)
    if type(outp) == int or type(outp) == float:
        if not is_number(str(expr)):
            outp_dict[str(expr)] = outp
    return outp, outp_dict

def execute_with_intermediate_values_abstract(expr, inp, fns, outp_dict):
    args = []
    for i, child in enumerate(expr.children):
        curr_arg, _ = execute_with_intermediate_values_abstract(child, inp, fns, outp_dict)
        args.append(curr_arg)
        # outp_dict[str(child)] = curr_arg
    outp = fns[expr.name](args,inp)
    # outp_dict[str(expr)] = outp
    if isinstance(outp, tuple):
        if not is_number(str(expr)):
            outp_dict[str(expr)] = outp
    return outp, outp_dict
