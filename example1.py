from lisp import *
import lisp_ex as concrete
import lisp_ex_abs as abstract
from test_helper import *
from read_obj_det_imgs import get_bounds, get_end_interval_w

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.stats import truncnorm

if __name__ == '__main__':
    # Basic set-up
    delta = 0.1
    filename_val = '/home/sangdonp/data/tmp/results_coco_val.pk'
    filename_test = '/home/sangdonp/data/tmp/results_coco_test.pk'
    min_IOU = 0.5
    small_area = 96**2
    abstractFns = abstract.get_fns_abs()
    concFns = concrete.get_fns()
    program = '(fold plus input 0)'
    expr = parse(program)

    lc_quantile, hic_quantile = get_bounds(filename_val, min_IOU, delta)
    w = get_end_interval_w(filename_val, min_IOU, delta)

    # print("Lowest confidence: ", str(-1 * lc_quantile))
    # print("Highest inconfidence: ", str(hic_quantile))
    # print("End w:" + str(w))

    with open(filename_val, 'rb') as f:
        testdata = pickle.load(f)

    time = list()
    abstractIntervalLengths = list()
    abstractIntervalCovered = list()
    concreteIntervalLengths = list()
    concreteIntervalCovered = list()

    for t, test in enumerate(testdata):
        time.append(t)
        if t % 200 == 0:
            print(t)
        abs_input = list()
        # Constructing abstract input (list of prediction sets)
        for i, score in enumerate(test['score_det_raw']):
            score_val = score.item()
            if score_val < lc_quantile:
                abs_input.append(((0,0), 1))
            elif score_val > hic_quantile:
                abs_input.append(((1,1), 1))
            else:
                abs_input.append(((0,1), 1))
        
        # Get prediction interval at output (propagated interval)
        abstractOutput = execute(expr, abs_input, abstractFns) 
        
        # True number of image detections in image
        true_output = test['bbox_gt'].nelement() / 4
        # Predicted number of image detections in image
        predicted_output = test['bbox_det_raw'].nelement() / 4
        # Get prediction interval at output (end-generated interval)
        concreteOutput = (predicted_output - w, predicted_output + w)

        abstractIntervalLengths.append(get_size(abstractOutput))
        abstractIntervalCovered.append(covered(abstractOutput, true_output))
        concreteIntervalLengths.append(get_size(concreteOutput))
        concreteIntervalCovered.append(covered(concreteOutput, true_output))
 
    print('Propagated interval length (average): ' + str(np.average(abstractIntervalLengths)))
    print('Propagated interval average coverage: ' + str(np.average(abstractIntervalCovered)))
    print('End-generated interval length (average): ' + str(np.average(concreteIntervalLengths)))
    print('End-generated interval average coverage: ' + str(np.average(concreteIntervalCovered)))

    # # Plot comparison in interval lengths between both methods
    plt.clf()
    plt.plot(time, abstractIntervalLengths, label = 'Propagated intervals', color = 'teal')
    plt.plot(time, concreteIntervalLengths, label = 'End-generated intervals', color = 'black')
    plt.xlabel('Interval Lengths Comparison')
    plt.legend()
    plt.title('Average interval length (propagated intervals): ' + str(np.average(abstractIntervalLengths)))
    plt.savefig('predintervals/intervalLengths/newtest1' +'.png', dpi=1200)

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
    plt.savefig('predintervals/intervalCoverage/newtest1' +'.png', dpi=1200)
