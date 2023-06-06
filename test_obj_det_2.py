from lisp import *
import lisp_ex as concrete
import lisp_ex_abs as abstract
from read_obj_det_imgs import get_bounds, is_small, get_center

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.stats import truncnorm

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

def get_left_images(bboxes, pixel_no):
    num_gt = 0
    for i, val in enumerate(bboxes):
        curr_center = get_center(val)
        if curr_center[0] < pixel_no:
            num_gt += 1
    return num_gt

def get_top_images_gt(img, pixel_no):
    num_gt = 0
    for i, val in enumerate(img['bbox_gt']):
        curr_center = get_center(val)
        if curr_center[0] < pixel_no:
            num_gt += 1
    return num_gt

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
    # Subtract 100 pixels from the x-coordinate of each bounding box, and filter the ones which are now less than 0 (within 100 pixels of left corner)
    # Then, map each of the remaining values to (1, 1) - indicating existence in the set of desired bounding boxes, and add them up. 
    program = '(fold plus (map (curry f1 0) (filter (curry ge 0) (map (curry plus -100) input))) 0)'
    expr = parse(program)

    # inp = [((352.3031303830332, 386.5895392947012), 1), ((133.10622333957616, 167.39263225124415), 1), ((388.50830006076757, 422.79470897243556), 1), ((279.81312488986913, 314.0995338015371), 1), ((44.545323605079076, 78.8317325167471), '?'), ((202.7231285519785, 237.0095374636465), '?'), ((44.55697750522556, 78.84338641689358), '?'), ((314.0576469845957, 348.3440558962637), '?'), ((2.9702261038011564, 37.25663501546917), '?'), ((119.07188343479099, 153.358292346459), '?'), ((99.89254307224216, 134.17895198391017), '?'), ((269.1774742551035, 303.4638831667715), '?'), ((343.35682606174413, 377.6432349734121), '?'), ((122.32876896335544, 156.61517787502345), '?'), ((311.61332630588475, 345.89973521755275), '?'), ((188.63107990695897, 222.91748881862696), '?'), ((223.56571888400975, 257.85212779567775), '?'), ((102.14453052951755, 136.43093944118556), '?'), ((73.22718357517185, 107.51359248683987), '?'), ((135.39334797336522, 169.6797568850332), '?'), ((65.36363148166599, 99.65004039333401), '?'), ((263.9127800412363, 298.1991889529043), '?'), ((266.66236805393163, 300.9487769655996), '?'), ((392.1329948849863, 426.4194037966543), '?'), ((254.43493580295507, 288.72134471462306), '?'), ((45.4308464474863, 79.71725535915432), '?'), ((192.2960274167246, 226.5824363283926), '?'), ((95.72641682102146, 130.01282573268946), '?'), ((230.43759083225194, 264.72399974391993), '?'), ((190.4135811276621, 224.6999900393301), '?'), ((26.87620853855076, 61.162617450218775), '?'), ((8.977313274878885, 43.2637221865469), '?'), ((330.758651012916, 365.045059924584), '?'), ((9.568558449286844, 43.85496736095486), '?'), ((369.8222496457285, 404.1086585573965), '?'), ((355.4340660519785, 389.7204749636465), '?'), ((192.75512623264257, 227.04153514431056), '?'), ((308.57260059787694, 342.85900950954493), '?'), ((155.89686512424413, 190.18327403591212), '?'), ((148.76620411350194, 183.05261302516993), '?'), ((113.85195087863865, 148.13835979030665), '?'), ((244.85639881565038, 279.14280772731837), '?'), ((240.88679432346288, 275.17320323513087), '?'), ((81.91614841892185, 116.20255733058987), '?'), ((274.53959583713475, 308.82600474880275), '?'), ((261.8275444455332, 296.1139533572012), '?'), ((213.2783729977793, 247.56478190944728), '?'), ((0.23271941616001257, 34.51912832782803), '?'), ((71.6143410153574, 105.90074992702542), '?'), ((9.174067730445291, 43.460476642113306), '?'), ((343.63757252170507, 377.92398143337306), '?'), ((27.488545650977517, 61.77495456264553), '?'), ((97.63962864353122, 131.92603755519923), '?'), ((29.256764645118142, 63.54317355678616), '?'), ((321.7627708859629, 356.04917979763087), '?'), ((132.8468544430918, 167.13326335475978), '?'), ((194.23122333957616, 228.51763225124415), '?'), ((302.7097923703379, 336.99620128200587), '?'), ((151.92426990940038, 186.21067882106837), '?'), ((69.33476185275974, 103.62117076442776), '?'), ((372.9475395627207, 407.2339484743887), '?'), ((241.50763630344335, 275.79404521511134), '?'), ((191.05163502170507, 225.33804393337306), '?'), ((202.546248669166, 236.832657580834), '?'), ((234.36888813449804, 268.655297046166), '?'), ((324.30860828830663, 358.5950171999746), '?'), ((321.0872337765879, 355.37364268825587), '?'), ((116.33871006442966, 150.62511897609767), '?'), ((213.9647514767832, 248.25116038845118), '?'), ((270.22425770236913, 304.5106666140371), '?'), ((311.55883717014257, 345.84524608181056), '?'), ((78.58576893283787, 112.87217784450588), '?'), ((52.23151897861423, 86.51792789028225), '?'), ((365.6420738644785, 399.9284827761465), '?'), ((33.18992542697849, 67.47633433864651), '?'), ((15.880620235938455, 50.16702914760647), '?'), ((349.67701649143163, 383.9634254030996), '?'), ((155.083846325416, 189.370255237084), '?'), ((20.5713779874033, 54.857786899071314), '?'), ((370.9035026974863, 405.1899116091543), '?'), ((385.67172169162694, 419.95813060329493), '?'), ((89.37400364353122, 123.66041255519924), '?'), ((98.61135410739841, 132.89776301906642), '?'), ((115.83571171237888, 150.1221206240469), '?'), ((208.92536854221288, 243.21177745388087), '?'), ((295.1742546506113, 329.4606635622793), '?'), ((275.75033497287694, 310.03674388454493), '?'), ((341.93153309299413, 376.2179420046621), '?'), ((346.40821766330663, 380.6946265749746), '?'), ((203.95223926975194, 238.23864818141993), '?'), ((91.53227161838474, 125.81868053005276), '?')]
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
    width_w = 17.143204455834006
    height_w = 16.540258397943425
    w = 6.099494097807565

    # test no. = 2475

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
        print(t)
        abs_input = list()
        for i, score in enumerate(test['score_det_raw']):
            x_coord, y_coord = get_center(test['bbox_det_raw'][i])
            width_interval = (x_coord - width_w, x_coord + width_w)
            score_val = score.item()

            if score_val < lc_quantile: # Not counted as detection
                pass
                # abs_input.append((width_interval, 0))
            elif score_val > hic_quantile:
                abs_input.append((width_interval, 1))
            else:
                abs_input.append((width_interval, '?'))
        
        if t == 2157:
            print(abs_input)
        abstractOutput = execute(expr, abs_input, abstractFns) 
        
        # True number of image detections to the left of image (within 100 pixels)
        true_output = get_left_images(test['bbox_gt'], 100)
        # print('Actual number: ' + str(true_output))
        predicted_output = get_left_images(test['bbox_det_raw'], 100)
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
    plt.savefig('predintervals/intervalLengths/left_100_both_' +str(t) +'.png', dpi=1200)

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
    plt.savefig('predintervals/intervalCoverage/left_100_both' +str(t) +'.png', dpi=1200)
