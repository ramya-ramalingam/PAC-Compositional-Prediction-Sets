from lisp import *
import lisp_ex as concrete
import lisp_ex_abs as abstract
from read_obj_det_imgs import get_bounds, is_small

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.stats import truncnorm

def get_pred_interval(res, delta):
    # Input:
    #       l - list of generated numbers for which to get an interval
    #       delta - 1 - confidence for prediction sets

    res_with_index = [(i, val) for i, val in enumerate(res)]
    
    sorted_res = sorted(res_with_index, key =  lambda x: -1*x[1])

    curr_sum = 0
    predIntervalList = list()
    # print(sortedProbList)

    for (j, prob) in sorted_res:
        predIntervalList.append(j)
        curr_sum += prob
        if curr_sum >= 1 - delta:
            break

    # print(predIntervalList)
    finalPredInterval = abstract.alpha(predIntervalList[0])

    for i in predIntervalList:
        finalPredInterval = abstract.join(finalPredInterval, abstract.alpha(i))

    return finalPredInterval


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

if __name__ == '__main__':
    # Basic set-up
    delta = 0.1
    filename_val = '/home/sangdonp/data/tmp/results_coco_val.pk'
    filename_test = '/home/sangdonp/data/tmp/results_coco_test.pk'
    min_IOU = 0.5
    small_area = 96**2
    small_count = 0
    abstractFns = abstract.get_fns_abs()
    concFns = concrete.get_fns()
    program = '(fold plus input 0)'
    expr = parse(program)

    # lc_quantile, hic_quantile = get_bounds(filename_val, min_IOU, delta)

    # print("Lowest confidence: ", str(-1 * lc_quantile))
    # print("Highest inconfidence: ", str(hic_quantile))

    with open(filename_val, 'rb') as f:
        testdata = pickle.load(f)

    time = list()
    abstractIntervalLengths = list()
    abstractIntervalCovered = list()
    concreteIntervalLengths = list()
    concreteIntervalCovered = list()

    lc_quantile = 0.08361497900839808
    hic_quantile = 0.9005308148229705
    w = 29.0

    lc_quantile_h = 0.7476653737587817
    hic_quantile_h = 0.6234039504194042

    for t, test in enumerate(testdata):
        # Don't include any images with small ground truth bounding boxes
        # flag_c = 0
        # for k, gt in enumerate(test['bbox_gt']):
        #     if is_small(gt, small_area):
        #         flag_c = 1
        #         break

        # if flag_c == 1:
        #     small_count += 1
        #     continue
            
        time.append(t)
        if t+1 % 200 == 0:
            print(t-1)
        abs_input = list()
        for i, score in enumerate(test['score_det_raw']):
            score_val = score.item()
            print(score_val)
            if score_val < lc_quantile:
                abs_input.append(((0,0), 1))
            elif score_val > hic_quantile:
                abs_input.append(((1,1), 1))
            else:
                abs_input.append(((0,1), 1))
        
        abstractOutput = execute(expr, abs_input, abstractFns) 
        
        # True number of image detections in image
        true_output = test['bbox_gt'].nelement() / 4
        # print('Actual number: ' + str(true_output))
        predicted_output = test['bbox_det_raw'].nelement() / 4
        concreteOutput = (predicted_output - w, predicted_output + w)

        abstractIntervalLengths.append(get_size(abstractOutput))
        abstractIntervalCovered.append(covered(abstractOutput, true_output))
        concreteIntervalLengths.append(get_size(concreteOutput))
        concreteIntervalCovered.append(covered(concreteOutput, true_output))
 
    print('Propagated interval length (average): ' + str(np.average(abstractIntervalLengths)))
    print('Propagated interval average coverage: ' + str(np.average(abstractIntervalCovered)))
    print('No. of small images: ' + str(small_count))

    # # Plot comparison in interval lengths between both methods
    plt.clf()
    plt.plot(time, abstractIntervalLengths, label = 'Propagated intervals', color = 'teal')
    plt.plot(time, concreteIntervalLengths, label = 'End-generated intervals', color = 'black')
    plt.xlabel('Interval Lengths Comparison')
    plt.legend()
    plt.title('Average interval length (propagated intervals): ' + str(np.average(abstractIntervalLengths)))
    plt.savefig('predintervals/intervalLengths/both_' +str(t) +'.png', dpi=1200)

    # # Plot comparison in coverage rate between both methods
    abstractIntervalCovered = np.array(abstractIntervalCovered)
    covered_abstract = [np.average(abstractIntervalCovered[:t+1]) for t in range(len(abstractIntervalCovered))]
    concreteIntervalCovered = np.array(concreteIntervalCovered)
    covered_concrete = [np.average(concreteIntervalCovered[:t+1]) for t in range(len(concreteIntervalCovered))]

    plt.clf()
    plt.ylabel("Coverage")
    plt.xlabel("Round")
    plt.plot(range(len(covered_abstract)), covered_abstract, label='Propagated intervals', color = 'teal')
    plt.plot(range(len(covered_concrete)), covered_concrete, label='End-generated intervals', color = 'black')

    plt.axhline(y= 1 - delta, c='r', linestyle='--', linewidth=2)
    plt.text(5000, 1 - delta, '  target coverage')
    plt.title('Marginal Coverage Comparison')
    # plt.ylim((0.2,1.05))
    plt.legend()
    plt.savefig('predintervals/intervalCoverage/both_' +str(t) +'.png', dpi=1200)
