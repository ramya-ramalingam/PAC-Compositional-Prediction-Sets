from collections import defaultdict
import pickle
import numpy as np
from main_det_coco import COCO_INSTANCE_CATEGORY_NAMES
from torchvision.ops import box_iou
import torch

MAX_WIDTH = 640
MAX_HEIGHT = 480

# Function to calculate area of a bounding box. Assumes input is tensor of the form 
# [xmin, ymin, xmax, ymax]
def get_area(bbox):
    if (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        print("Something wrong in area calculation")
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# Function to return center of bounding box as tuple (x_coord, y_coord). Assumes input is tensor of the form 
# [xmin, ymin, xmax, ymax]
def get_center(bbox):
    if (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        print("Something wrong in center calculation")
    x_coord = (bbox[0].item() + bbox[2].item()) / 2
    y_coord = (bbox[1].item() + bbox[3].item()) / 2
    return (x_coord, y_coord)

# For a given set of detections and ground truths, map each detection to a single ground truth 
# (based on highest IoU value). 
# Does not map detections which doesn't have an IOU >= min_IOU with any ground truth
# Returns:  - a list of indices dt_to_gt. dt_to_gt[i] = j means the ith detection maps to the jth ground truth. 
#           - a dictionary mapping each ground truth index to the set of detection indices mapped to it 
def map_det_to_gt(det_list, gt_list, min_IOU):
    det_to_gt = list()
    gt_to_det_dict = defaultdict(list)
    # If there are no ground truths, let each detection map to the value -1. 
    if gt_list.nelement() == 0:
        det_to_gt = [-1 for i in range(det_list.nelement())]
        return det_to_gt, gt_to_det_dict
    for i, det_val in enumerate(det_list):
        curr_IoU_list = list()
        for j, gt_val in enumerate(gt_list):
            # Using Torchvision IoU
            det_tensor_list = torch.empty(1,4)
            gt_tensor_list = torch.empty(1, 4)
            det_tensor_list[0] = det_val
            gt_tensor_list[0] = gt_val
            currIoU = box_iou(det_tensor_list, gt_tensor_list)
            curr_IoU_list.append(currIoU.item())
        maxIoU = np.max(curr_IoU_list)
        if (maxIoU < min_IOU):
            det_to_gt.append(-1) # Det not mapped to anything
        else:
            maxIoUIndex = np.argmax(curr_IoU_list)
            det_to_gt.append(maxIoUIndex)
            gt_to_det_dict[maxIoUIndex].append(i)
    
    return det_to_gt, gt_to_det_dict

# For a given set of detections and ground truths, map each ground truth backwards to a detection based on 
# confidence value
# Returns a list of indices gt_to_dt. gt_to_dt[j] = i means that the jth ground truth maps back to the ith detection
# if gt_to_dt[j] = -1, that means no detections were mapped to that ground truth. 
def map_gt_to_det(gt_list, gt_to_det_dict, det_score_list):
    gt_to_det = list()
    chosen_dets = list()
    for i, gt_val in enumerate(gt_list):
        det_backmap = gt_to_det_dict[i]
        det_backmap_with_score = [(i, det_score_list[i].item()) for i in det_backmap]
        if det_backmap_with_score == []: 
            # When no detections were mapped to the current ground truth bb
            gt_to_det.append(-1)
        else:
            # Pick the detection with the corresponding highest confidence score
            det_backmap_with_score.sort(key=lambda x: -1 * x[1])
            maxDtIndex = det_backmap_with_score[0][0]
            gt_to_det.append(maxDtIndex)
            chosen_dets.append(maxDtIndex)
    return gt_to_det, chosen_dets

# Returns a list l of size equal to number of raw detections. 
# l[i] = 1 if the ith raw detection was chosen, 0 otherwise
def fill_chosen_dets(chosen_dets, det_list):
    chosen_dets_filled = list()
    for i, val in enumerate(det_list):
        if i in chosen_dets:
            chosen_dets_filled.append(1)
        else:
            chosen_dets_filled.append(0)
    return chosen_dets_filled

# Return the "lowest confidence of correctness value" for a given set of detections and scores
def lowest_confidence(chosen_dets_filled, det_score_list):
    chosen_det_scores = [det_score_list[i].item() for i in range(len(chosen_dets_filled)) if chosen_dets_filled[i] == 1]
    # If this is empty, it means there were no chosen detections. 
    if chosen_det_scores == []:
        # Return 2 - trivially, we have covered all correct detections with any threshold
        return 2
    lowest_confidence = min(chosen_det_scores)
    return lowest_confidence

# Return the "highest confidence of incorrectness value" for a given set of detections and scores
def highest_inconfidence(chosen_dets_filled, det_score_list):
    unchosen_det_scores = [det_score_list[i].item() for i in range(len(chosen_dets_filled)) if chosen_dets_filled[i] == 0]
    # If this is empty, it implies that no detection was incorrect. 
    if unchosen_det_scores == []:
        # return -1 - trivially, we have covered all correct detections with any threshold
        return -1
    highest_inconfidence = max(unchosen_det_scores)
    return highest_inconfidence 

def get_conformal_quantile(residual_list, delta):
    calibration_size = len(residual_list)
    desired_quantile = np.ceil((1 - delta) * (calibration_size + 1)) / calibration_size
    chosen_quantile = np.minimum(1.0, desired_quantile)
    w = np.quantile(residual_list, chosen_quantile)
    return w

def get_c_width(delta, residuals):
    calibration_size = len(residuals)
    desired_quantile = np.ceil((1 - delta) * (calibration_size + 1)) / calibration_size
    chosen_quantile = np.minimum(1.0, desired_quantile)
    w = np.quantile(residuals, chosen_quantile)
    return w

def get_gt_to_det(img_result, min_IOU):
    det_to_gt, gt_to_det_dict = map_det_to_gt(img_result['bbox_det_raw'], img_result['bbox_gt'], min_IOU)
    # print(det_to_gt)
    # print(gt_to_det_dict)
    gt_to_det, chosen_dets = map_gt_to_det(img_result['bbox_gt'], gt_to_det_dict, img_result['score_det_raw'])

    return gt_to_det

# Returns true if bbox area is less than area, false otherwise
def is_small(bbox, area):
    bbox_area = get_area(bbox)
    if (bbox_area < area):
        return True
    else:
        return False

def get_left_images(bboxes, pixel_no):
    num_gt = 0
    for i, val in enumerate(bboxes):
        curr_center = get_center(val)
        if curr_center[0] < pixel_no:
            num_gt += 1
    return num_gt

def get_bottom_images(bboxes, pixel_no):
    num_gt = 0
    for i, val in enumerate(bboxes):
        curr_center = get_center(val)
        if curr_center[1] < pixel_no:
            num_gt += 1
    return num_gt

def get_top_images(bboxes, pixel_no):
    num_gt = 0
    for i, val in enumerate(bboxes):
        curr_center = get_center(val)
        if curr_center[1] >  MAX_HEIGHT - pixel_no:
            num_gt += 1
    return num_gt

def get_right_images(bboxes, pixel_no):
    num_gt = 0
    for i, val in enumerate(bboxes):
        curr_center = get_center(val)
        if curr_center[0] >  MAX_WIDTH - pixel_no:
            num_gt += 1
    return num_gt

# Function which does everything done in main() - for imports
def get_bounds(filename, min_IOU, delta):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    lowest_confidence_list = list()
    highest_inconfidence_list = list()
    lowest_confidence_neg = list()
    bad_list = list()
    delta = 0.1
    unused_count = 0
    small_count = 0
    x = 0
    for i, val in enumerate(results):

        if val['bbox_gt'].nelement() == 0 and val['bbox_det_raw'].nelement() == 0:
            # print('No ground truths: ' + str(i))
            unused_count += 1
            continue
    
        # 1. For each result, map every bounding box detection to a ground-truth bounding box. Do this on the basis of 
        #    IoU metric - Area of overlap / Area of union = Area of overlap / (Ground truth area + Predicted Box area - Area of overlap)
        #    https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

        det_to_gt, gt_to_det_dict = map_det_to_gt(val['bbox_det_raw'], val['bbox_gt'], min_IOU)

        # 2. For each ground-truth g, there is now a set of predictions S which map to it. Among these, find the one with the highest
        #    confidence score, and map the ground-truth back to it. This is the corresponding "correct" bounding-box for that ground truth.
        #    Note that some ground truths may not have a backward mapping (i.e. if S is empty) in which the backward mapping is to nothing.

        gt_to_det, chosen_dets = map_gt_to_det(val['bbox_gt'], gt_to_det_dict, val['score_det_raw'])
        chosen_dets_filled = fill_chosen_dets(chosen_dets, results[i]['bbox_det_raw'])

        # 3. Among the "correct" bounding-boxes (i.e. the ones which have a backwards mapping from some ground-truth), find the one with 
        #    the lowest confidence score. This is the "lowest confidence of correctness" - the lowest confidence value for a correctly 
        #    predicted bounding box which accurately maps to a ground-truth. 
        lc = lowest_confidence(chosen_dets_filled, val['score_det_raw'])
        if lc == -1:
            bad_list.append(i)

        # 4. Among the predictions which were in some set S, but didn't get chosen as a "correct" bounding-box, find the one with the
        #    highest confidence score. This is the "highest confidence of incorrectness" - the highest confidence value for a bounding box
        #    which was predicted but which was incorrect (i.e. it didn't accurately map to any ground-truth). 
        hic = highest_inconfidence(chosen_dets_filled, val['score_det_raw'])
        lowest_confidence_list.append(lc)
        lowest_confidence_neg.append(-1 * lc)
        highest_inconfidence_list.append(hic)
    
    # 5. We get a set of scores for both "lowest confidence of correctness" and "highest confidence of incorrectness" from the
    #    calibration data. This can be used with direct conformal prediction approaches to get a quantile value for each ((1) and (2)),  
    #    which can be used to select "uncertain" bounding boxes - those with a confidence value in the interval [(1), (2)] in test data
    # lc_quantile = get_conformal_quantile(lowest_confidence_list, 1 - delta)
    bound_delta = delta / 2
    lc_quantile = get_conformal_quantile(lowest_confidence_neg, bound_delta)
    hic_quantile = get_conformal_quantile(highest_inconfidence_list, bound_delta)
    return lc_quantile, hic_quantile

def get_end_interval_w(filename, min_IOU, delta):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    residuals = list()
    for i, val in enumerate(results):
        if val['bbox_gt'].nelement() == 0 and val['bbox_det_raw'].nelement() == 0:
            continue
        # No. of ground truth detections in image
        true_no = val['bbox_gt'].nelement() / 4
        # Predicted number of image detections in image
        predicted_no = val['bbox_det_raw'].nelement() / 4
        residuals.append(np.abs(true_no - predicted_no))
    w = get_c_width(delta, residuals)
    return w


# Returns the maximum difference between x-coordinate / y-coordinate of a detection and its chosen ground truth for a given image
# Used for generating prediction sets around the centers of detections 
def get_max_width_and_height(chosen_dets_filled, det_to_gt, det_list, gt_list):
    max_width = 0
    max_height = 0
    for i, val in enumerate(chosen_dets_filled):
        curr_det = det_list[i]
        det_center = get_center(curr_det)
        if val == 1: # This detection was chosen
            gt_index = det_to_gt[i]
            curr_gt = gt_list[gt_index]
            gt_center = get_center(curr_gt)
            max_width = max(max_width, np.abs(det_center[0] - gt_center[0]))
            max_height = max(max_height, np.abs(det_center[1] - gt_center[1]))
    
    return (max_width, max_height)

def get_bbox_center_bounds(filename, min_IOU, delta):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    max_width_list = list()
    max_height_list = list()
    for i, val in enumerate(results):
        if val['bbox_gt'].nelement() == 0 and val['bbox_det_raw'].nelement() == 0:
            continue
        det_to_gt, gt_to_det_dict = map_det_to_gt(val['bbox_det_raw'], val['bbox_gt'], min_IOU)
        gt_to_det, chosen_dets = map_gt_to_det(val['bbox_gt'], gt_to_det_dict, val['score_det_raw'])
        chosen_dets_filled = fill_chosen_dets(chosen_dets, results[i]['bbox_det_raw'])
        max_width, max_height = get_max_width_and_height(chosen_dets_filled, det_to_gt, val['bbox_det_raw'], val['bbox_gt'])
        max_width_list.append(max_width)
        max_height_list.append(max_height)
    
    bound_delta = delta / 2
    width_w = get_c_width(bound_delta, max_width_list)
    height_w = get_c_width(bound_delta, max_height_list)

    return (width_w, height_w)
            

def main():
    filename = '/home/sangdonp/data/tmp/results_coco_val.pk'
    min_IOU = 0.5
    small_area = 96**2
    flag_c = 0
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    lowest_confidence_list = list()
    highest_inconfidence_list = list()
    lowest_confidence_neg = list()

    max_width_list = list()
    max_height_list = list()
    residuals = list()
    bad_list = list()
    delta = 0.2
    unused_count = 0
    for i, val in enumerate(results):

        if val['bbox_gt'].nelement() == 0 and val['bbox_det_raw'].nelement() == 0:
            print('No ground truths: ' + str(i))
            unused_count += 1
            continue

        # 1. For each result, map every bounding box detection to a ground-truth bounding box. Do this on the basis of 
        #    IoU metric - Area of overlap / Area of union = Area of overlap / (Ground truth area + Predicted Box area - Area of overlap)
        #    https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

        det_to_gt, gt_to_det_dict = map_det_to_gt(val['bbox_det_raw'], val['bbox_gt'], min_IOU)

        # 2. For each ground-truth g, there is now a set of predictions S which map to it. Among these, find the one with the highest
        #    confidence score, and map the ground-truth back to it. This is the corresponding "correct" bounding-box for that ground truth.
        #    Note that some ground truths may not have a backward mapping (i.e. if S is empty) in which the backward mapping is to nothing.

        gt_to_det, chosen_dets = map_gt_to_det(val['bbox_gt'], gt_to_det_dict, val['score_det_raw'])
        chosen_dets_filled = fill_chosen_dets(chosen_dets, results[i]['bbox_det_raw'])

        # 3. Among the "correct" bounding-boxes (i.e. the ones which have a backwards mapping from some ground-truth), find the one with 
        #    the lowest confidence score. This is the "lowest confidence of correctness" - the lowest confidence value for a correctly 
        #    predicted bounding box which accurately maps to a ground-truth. 
        lc = lowest_confidence(chosen_dets_filled, val['score_det_raw'])
        if lc == -1:
            bad_list.append(i)

        # 4. Among the predictions which were in some set S, but didn't get chosen as a "correct" bounding-box, find the one with the
        #    highest confidence score. This is the "highest confidence of incorrectness" - the highest confidence value for a bounding box
        #    which was predicted but which was incorrect (i.e. it didn't accurately map to any ground-truth). 
        hic = highest_inconfidence(chosen_dets_filled, val['score_det_raw'])
        lowest_confidence_list.append(lc)
        lowest_confidence_neg.append(-1 * lc)
        highest_inconfidence_list.append(hic)

        max_width, max_height = get_max_width_and_height(chosen_dets_filled, det_to_gt, val['bbox_det_raw'], val['bbox_gt'])
        max_width_list.append(max_width)
        max_height_list.append(max_height)

        # 4b (special). Get residual for baseline (difference between GT and number of bounding boxes)
        true_no = get_top_images(val['bbox_gt'], 100)
        predicted_no = get_top_images(val['bbox_det_raw'], 100)
        residuals.append(np.abs(true_no - predicted_no))
    
    # 5. We get a set of scores for both "lowest confidence of correctness" and "highest confidence of incorrectness" from the
    #    calibration data. This can be used with direct conformal prediction approaches to get a quantile value for each ((1) and (2)),  
    #    which can be used to select "uncertain" bounding boxes - those with a confidence value in the interval [(1), (2)] in test data
    bound_delta = delta / 2
    lc_quantile = get_conformal_quantile(lowest_confidence_neg, bound_delta)
    hic_quantile = get_conformal_quantile(highest_inconfidence_list, bound_delta)
    w = get_c_width(delta, residuals)
    width_w = get_c_width(bound_delta, max_width_list)
    height_w = get_c_width(bound_delta, max_height_list)
    print("Lowest confidence: ", str(-1 * lc_quantile))
    print("Highest inconfidence: ", str(hic_quantile))
    print("Conformal Width: " + str(w))
    print("Max Width for Centers: " + str(width_w))
    print("Max Height for Centers: " + str(height_w))
    print('No. of empty images: ', str(unused_count))
    
    return

if __name__ == '__main__':
    main()
