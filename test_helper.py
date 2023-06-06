
def get_size(interval):
    if interval == None:
        return 0
    else:
        return interval[1] - interval[0]

def covered(interval, val):
    if interval == None:
        return False
    if interval[0] <= val and val <= interval[1]:
        return True
    return False

def smaller(abstract, concrete):
    ab_size = abstract[1] - abstract[0]
    conc_size = concrete[1] - concrete[0]
    if ab_size <= conc_size:
        return abstract
    else:
        return concrete

def intersection(abstract, concrete):
    lower_bound_larger = max(abstract[0], concrete[0])
    upper_bound_smaller = min(abstract[1], concrete[1])
    if lower_bound_larger > upper_bound_smaller:
        return None
    else:
        return (lower_bound_larger, upper_bound_smaller)

def propagate_intersected_intervals(expr, inp, fns, conc_dict, abstract_dict):
    args = []
    for i, child in enumerate(expr.children):
        curr_arg, _ = propagate_intersected_intervals(child, inp, fns, conc_dict, abstract_dict)
        if str(child) in conc_dict:
            # print('before intersection')
            # print(curr_arg, conc_dict[str(child)])
            curr_arg = intersection(curr_arg, conc_dict[str(child)])
            # print('after intersection')
            # print(curr_arg)
        args.append(curr_arg)
    outp = fns[expr.name](args, inp)
    if isinstance(outp, tuple):
        if not is_number(str(expr)):
            abstract_dict[str(expr)] = outp
    return outp, abstract_dict