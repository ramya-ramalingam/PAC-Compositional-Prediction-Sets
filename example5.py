# Counting number of people to the right of a car within 100 pixels of it (centers)

from lisp import *
import lisp_ex as concrete
import lisp_ex_abs as abstract
from test_helper import *
from read_obj_det_imgs import get_bounds, get_end_interval_w, get_bbox_center_bounds, get_center, get_left_images, get_bottom_images, get_top_images, get_right_images
from read_obj_det_imgs import left_images, bottom_images, top_images, right_images
from read_obj_det_imgs import get_person_left_of_car, person_left_car

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.stats import truncnorm
import sys
sys.setrecursionlimit(10000)

PERSON_INDEX = 1
CAR_INDEX = 3


# Given the set of interval lengths, plot them sorted in increasing interval length order of the intersected intervals (our method)
def plotSorted(time, abstract, concrete, intersected):
    overlapping = 0.5
    ordered_indices = np.argsort(intersected)

    ordered_abstract = [abstract[i] for i in ordered_indices]
    ordered_concrete = [concrete[i] for i in ordered_indices]
    ordered_intersected = [intersected[i] for i in ordered_indices]

    # Plot comparison in interval lengths between all methods
    plt.clf()
    plt.plot(time, ordered_abstract, label = 'Propagated intervals', color = 'red')
    plt.plot(time, ordered_concrete, label = 'End-generated intervals', color = 'black')
    plt.plot(time, ordered_intersected, label = 'Intersected intervals', color = 'green')
    plt.xlabel('Interval Lengths Comparison')
    plt.legend()
    plt.title('Average interval length (propagated intervals): ' + str(np.average(abstract)) + '\n' +
              'Average interval length (intersected intervals): ' + str(np.average(intersected)))
    plt.savefig('paper/intervalLengths/example4' +'.png', dpi=1200)

if __name__ == '__main__':
    # Basic set-up
    delta = 0.1
    filename_val = '/home/sangdonp/data/tmp/results_coco_val.pk'
    filename_test = '/home/sangdonp/data/tmp/results_coco_test.pk'
    min_IOU = 0.5
    num_nonzero = 0 # Keeping track of test images with non-zero true output
    abstractFns = abstract.get_fns_abs()
    concFns = concrete.get_fns()

    # Program to count number of people & cars within 100 pixels of each other (with people to the left of the cars)
    program = '(fold plus (map (curry f1 0) (filter (curry ge 100) (map tuple_dist (filter tuple_leq_x (list_prod input))))) 0)'
    expr = parse(program)

    pixels = 100
    def curr_person_left_car(img):
        return person_left_car(img, pixels)
    
    new_delta = delta / 2
    lc_quantile, hic_quantile = get_bounds(filename_val, min_IOU, new_delta) # Both this and the next need to be correct 
    width_w, height_w = get_bbox_center_bounds(filename_val, min_IOU, new_delta)
    w = get_end_interval_w(filename_val, min_IOU, delta, curr_person_left_car)

    lc_quantile2, hic_quantile2 = get_bounds(filename_val, min_IOU, new_delta / 2)
    width_w2, height_w2 = get_bbox_center_bounds(filename_val, min_IOU, new_delta / 2)
    w2 = get_end_interval_w(filename_val, min_IOU, delta / 2, curr_person_left_car)

    print("Lowest confidence: ", str(-1 * lc_quantile))
    print("Highest inconfidence: ", str(hic_quantile))
    print("Width bound: ", str(width_w))
    print("Height bound: ", str(height_w))

    with open(filename_val, 'rb') as f:
        testdata = pickle.load(f)

    time = list()
    abstractIntervalLengths = list()
    abstractIntervalCovered = list()
    concreteIntervalLengths = list()
    concreteIntervalCovered = list()
    intersectedIntervalLengths = list()
    intersectedIntervalCovered = list()
    true_labels = list()
    uncertainty_list = list()

    for t, test in enumerate(testdata):
        time.append(t)
        print(t)
        if (t % 200 == 0):
            print(t)
        person_list = list()
        person_list2 = list()
        car_list = list()
        car_list2 = list()
        abs_input = list()
        abs_input2 = list()
        # Constructing abstract input (list of prediction sets)
        for i, score in enumerate(test['score_det_raw']):
            x_coord, y_coord = get_center(test['bbox_det_raw'][i])
            width_interval = (x_coord - width_w, x_coord + width_w)
            width_interval2 = (x_coord - width_w2, x_coord + width_w2)
            height_interval = (y_coord - height_w, y_coord + height_w)
            height_interval2 = (y_coord -height_w2, y_coord + height_w2)
            score_val = score.item()
            if score_val < lc_quantile: # Not counted as detection
                pass
            elif score_val > hic_quantile:
                if test['label_det_raw'][i].item() == PERSON_INDEX:
                    person_list.append(((width_interval, height_interval), 1))
                    person_list2.append(((width_interval2, height_interval2), 1))
                elif test['label_det_raw'][i].item() == CAR_INDEX:
                    car_list.append(((width_interval, height_interval), 1))
                    car_list2.append(((width_interval2, height_interval2), 1))
            else:
                if test['label_det_raw'][i].item() == PERSON_INDEX:
                    person_list.append(((width_interval, height_interval), '?'))
                    person_list2.append(((width_interval2, height_interval2), '?'))
                elif test['label_det_raw'][i].item() == CAR_INDEX:
                    car_list.append(((width_interval, height_interval), '?'))
                    car_list2.append(((width_interval2, height_interval2), '?'))
        # Get prediction interval at output (propagated interval)
        abs_input = (person_list, car_list)
        abs_input2 = (person_list2, car_list2)
        abstractOutput = execute(expr, abs_input, abstractFns) 
        abstractOutput2 = execute(expr, abs_input2, abstractFns)
        
        # True number of image detections to the left of image (within 100 pixels)
        true_output = get_person_left_of_car(test['bbox_gt'], test['label_gt'], pixels)
        # Predicted number of image detections to the left of image (within 100 pixels)
        predicted_output = get_person_left_of_car(test['bbox_det_raw'], test['label_det_raw'], pixels)
        if (true_output > 0):
            num_nonzero += 1
        true_labels.append(true_output)
        uncertainty_list.append(np.abs(true_output - predicted_output))
        # Derive prediction interval at output (end-generated interval)
        concreteOutput = (predicted_output - w, predicted_output + w)
        concreteOutput2 = (predicted_output - w2, predicted_output + w)

        # Will be 'None' of the two input intervals do not intersect
        intersectedOutput = intersection(abstractOutput2, concreteOutput2)

        abstractIntervalLengths.append(get_size(abstractOutput))
        abstractIntervalCovered.append(covered(abstractOutput, true_output))
        concreteIntervalLengths.append(get_size(concreteOutput))
        concreteIntervalCovered.append(covered(concreteOutput, true_output))
        intersectedIntervalLengths.append(get_size(intersectedOutput))
        intersectedIntervalCovered.append(covered(intersectedOutput, true_output))
 
    print('Propagated interval length (average): ' + str(np.average(abstractIntervalLengths)))
    print('Propagated interval average coverage: ' + str(np.average(abstractIntervalCovered)))
    print('End-generated interval length (average): ' + str(np.average(concreteIntervalLengths)))
    print('End-generated interval average coverage: ' + str(np.average(concreteIntervalCovered)))
    print('Intersected interval length (average): ' + str(np.average(intersectedIntervalLengths)))
    print('Intersected interval average coverage: ' + str(np.average(intersectedIntervalCovered)))
    print('Number of images with non-zero true value: ' + str(num_nonzero))
    print('Variance of uncertainty scores: ', np.var(uncertainty_list))

    # Plotting true point output across test data
    plt.clf()
    plt.plot(time, true_labels, label = 'True point output', color = 'green')
    plt.legend()
    plt.title('Variance of true labels across test data')
    plt.savefig('paper/trueOutput/example4.png', dpi=1200)

    # Plotting uncertainty scores across test data
    ordered_indices = np.argsort(uncertainty_list)
    ordered_uncertainty = [uncertainty_list[i] for i in ordered_indices]
    plt.clf()
    plt.plot(time, ordered_uncertainty, label = 'Uncertainty', color = 'purple')
    plt.legend()
    plt.title('Variance of true labels across test data')
    plt.savefig('paper/uncertaintyScores/example4_sorted.png', dpi=1200)

    # # Plot comparison in interval lengths between both methods
    plt.clf()
    plt.plot(time, abstractIntervalLengths, label = 'Propagated intervals', color = 'red')
    plt.plot(time, concreteIntervalLengths, label = 'End-generated intervals', color = 'black')
    plt.plot(time, intersectedIntervalLengths, label = 'Intersected intervals', color = 'green')
    plt.xlabel('Interval Lengths Comparison')
    plt.legend()
    plt.title('Average interval length (propagated intervals): ' + str(np.average(abstractIntervalLengths)) + '\n' +
              'Average interval length (intersected intervals): ' + str(np.average(intersectedIntervalLengths)))
    plt.savefig('paper/intervalLengths/example4' +'.png', dpi=1200)

    # Do the same, but ordered by intersected interval length (our method)
    plotSorted(time, abstractIntervalLengths, concreteIntervalLengths, intersectedIntervalLengths)

    # # Plot comparison in coverage rate between both methods
    abstractIntervalCovered = np.array(abstractIntervalCovered)
    covered_abstract = [np.average(abstractIntervalCovered[:t+1]) for t in range(len(abstractIntervalCovered))]
    concreteIntervalCovered = np.array(concreteIntervalCovered)
    covered_concrete = [np.average(concreteIntervalCovered[:t+1]) for t in range(len(concreteIntervalCovered))]
    intersectedIntervalCovered = np.array(intersectedIntervalCovered)
    covered_intersected = [np.average(intersectedIntervalCovered[:t+1]) for t in range(len(intersectedIntervalCovered))]

    plt.clf()
    plt.ylabel("Coverage")
    plt.xlabel("Round")
    plt.plot(range(len(covered_abstract)), covered_abstract, label='Propagated intervals', color = 'red')
    plt.plot(range(len(covered_concrete)), covered_concrete, label='End-generated intervals', color = 'black')
    plt.plot(range(len(covered_intersected)), covered_intersected, label='Intersected intervals', color = 'green')

    plt.axhline(y= 1 - delta, c='r', linestyle='--', linewidth=2)
    plt.text(3000, 1 - delta, '  target coverage')
    plt.title('Marginal Coverage Comparison')
    # plt.ylim((0.2,1.05))
    plt.legend()
    plt.savefig('paper/intervalCoverage/example4' +'.png', dpi=1200)
