from lisp import *
import lisp_ex as concrete
import lisp_ex_abs as abstract
from test_helper import *
from read_obj_det_imgs import get_bounds, get_end_interval_w, get_bbox_center_bounds, get_center, get_left_images, get_bottom_images, get_top_images, get_right_images
from read_obj_det_imgs import left_images, bottom_images, top_images, right_images

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.stats import truncnorm


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
    plt.savefig('paper/intervalLengths/example2_left_ordered' +'.png', dpi=1200)

if __name__ == '__main__':
    # Basic set-up
    delta = 0.1
    filename_val = '/home/sangdonp/data/tmp/results_coco_val.pk'
    filename_test = '/home/sangdonp/data/tmp/results_coco_test.pk'
    min_IOU = 0.5
    small_area = 96**2
    abstractFns = abstract.get_fns_abs()
    concFns = concrete.get_fns()
    # Subtract 100 pixels from the x-coordinate of each bounding box, and filter the ones which are now less than 0 (within 100 pixels of left corner)
    # Then, map each of the remaining values to (1, 1) - indicating existence in the set of desired bounding boxes, and add them up. 
    program = '(fold plus (map (curry f1 0) (filter (curry ge 0) (map (curry plus -100) input))) 0)' # Left
    # Similar programs for each of the sides of the image
    program2 = '(fold plus (map (curry f1 0) (filter (curry ge 0) (map (curry plus -100) input))) 0)' # Bottom 
    program3 = '(fold plus (map (curry f1 0) (filter (curry le MAX_WIDTH) (map (curry plus 100) input))) 0)' # Right
    program4 = '(fold plus (map (curry f1 0) (filter (curry le MAX_HEIGHT) (map (curry plus 100) input))) 0)' # Top
    expr = parse(program)

    pixels = 100
    def curr_left_images(img):
        return left_images(img, pixels)
    def curr_bottom_images(img):
        return bottom_images(img, pixels)
    def curr_top_images(img):
        return top_images(img, pixels)
    def curr_right_images(img):
        return right_images(img, pixels)
    
    new_delta = delta / 2
    lc_quantile, hic_quantile = get_bounds(filename_val, min_IOU, new_delta) # Both this and the next need to be correct 
    width_w, height_w = get_bbox_center_bounds(filename_val, min_IOU, new_delta)
    w = get_end_interval_w(filename_val, min_IOU, delta, curr_left_images)

    lc_quantile2, hic_quantile2 = get_bounds(filename_val, min_IOU, new_delta / 2)
    width_w2, height_w2 = get_bbox_center_bounds(filename_val, min_IOU, new_delta / 2)
    w2 = get_end_interval_w(filename_val, min_IOU, delta / 2, curr_left_images)

    # print("Lowest confidence: ", str(-1 * lc_quantile))
    # print("Highest inconfidence: ", str(hic_quantile))
    # print("Width bound: ", str(width_w))
    # print("Height bound: ", str(height_w))

    with open(filename_val, 'rb') as f:
        testdata = pickle.load(f)

    time = list()
    abstractIntervalLengths = list()
    abstractIntervalCovered = list()
    concreteIntervalLengths = list()
    concreteIntervalCovered = list()
    intersectedIntervalLengths = list()
    intersectedIntervalCovered = list()

    for t, test in enumerate(testdata):
        time.append(t)
        if (t % 200 == 0):
            print(t)
        abs_input = list()
        abs_input2 = list()
        # Constructing abstract input (list of prediction sets)
        for i, score in enumerate(test['score_det_raw']):
            x_coord, y_coord = get_center(test['bbox_det_raw'][i])
            width_interval = (x_coord - height_w, x_coord + height_w)
            width_interval2 = (x_coord - height_w2, x_coord + height_w2)
            score_val = score.item()
            if score_val < lc_quantile: # Not counted as detection
                pass
            elif score_val > hic_quantile:
                abs_input.append((width_interval, 1))
                abs_input2.append((width_interval2, 1))
            else:
                abs_input.append((width_interval, '?'))
                abs_input2.append((width_interval2, '?'))
        # Get prediction interval at output (propagated interval)
        abstractOutput = execute(expr, abs_input, abstractFns) 
        abstractOutput2 = execute(expr, abs_input2, abstractFns)
        
        # True number of image detections to the left of image (within 100 pixels)
        true_output = get_left_images(test['bbox_gt'], pixels)
        # Predicted number of image detections to the left of image (within 100 pixels)
        predicted_output = get_left_images(test['bbox_det_raw'], pixels)
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

    # # Plot comparison in interval lengths between both methods
    plt.clf()
    plt.plot(time, abstractIntervalLengths, label = 'Propagated intervals', color = 'red')
    plt.plot(time, concreteIntervalLengths, label = 'End-generated intervals', color = 'black')
    plt.plot(time, intersectedIntervalLengths, label = 'Intersected intervals', color = 'green')
    plt.xlabel('Interval Lengths Comparison')
    plt.legend()
    plt.title('Average interval length (propagated intervals): ' + str(np.average(abstractIntervalLengths)) + '\n' +
              'Average interval length (intersected intervals): ' + str(np.average(intersectedIntervalLengths)))
    plt.savefig('paper/intervalLengths/example2_left' +'.png', dpi=1200)

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
    plt.savefig('paper/intervalCoverage/example2_left' +'.png', dpi=1200)
