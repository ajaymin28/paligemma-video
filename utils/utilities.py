import os
from PIL import Image
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Iterable
import re
import copy
import os
import pickle
import cv2
from tqdm import tqdm
import json
# from prompt_magic.TaskDescription import Task_description_v10_sam
import re
from collections import defaultdict

AG_OBJECTS_ALTERATIVES = {
    "cup": "cup/glass/bottle",
    "glass": "cup/glass/bottle",
    "bottle": "cup/glass/bottle",
    "closet": "closet/cabinet",
    "cabinet": "closet/cabinet",
    "phone": "phone/camera",
    "camera": "phone/camera",
    "notebook": "paper/notebook",
    "paper": "paper/notebook",
    "sofa": "sofa/couch",
    "couch": "sofa/couch"
}
class SEEDS:
    """
    Don't change values
    """
    RANDOM_ANNOTATIONS_SHUFFLE_SEED = 145
    AG_OBJECT_PREDICATE_PARTIAL_SELECTION_SEEDS = [978, 324]

def addTriplet(triplet, Objects, relations):
    """
    Add triplet if entities belongs to predefined selected list.
    """

def consolidate_results(jsons_list):
    cleaned_outputs = {}
    for rawData in jsons_list:
        # triplet_key = "cleaned_output"
        print(rawData)
        with open(rawData) as f:
            raw_data = json.loads(f.read())
            for raw_data_item in [raw_data]:
                for video_id, video_data in raw_data_item.items():
                    if video_id not in cleaned_outputs.keys():
                        cleaned_outputs[video_id] = {}
                    for block_id,block_data in video_data.items():
                        
                        if "cleaned_output" not in block_data.keys() and "triplets" not in block_data.keys():
                            continue

                        if "cleaned_output" not in block_data.keys():
                            triplet_key = "triplets"
                        else:
                            block_data["triplets"] = copy.deepcopy(block_data["cleaned_output"])
                            del block_data["cleaned_output"]
                            
                        # if "triplets" not in block_data.keys():
                        #     triplet_key = "cleaned_output"

                        if "triplets" in block_data.keys():
                            if block_id not in cleaned_outputs[video_id]:
                                cleaned_outputs[video_id][block_id] = {}

                            cleaned_outputs[video_id][block_id] = {
                                "frames": block_data["frames"],
                                "GT_triplets": block_data["GT_triplets"],
                                "triplets": block_data["triplets"],
                                # "triplets": block_data["cleaned_output"],
                            }
    return cleaned_outputs

def pre_clean_temporal_triplets(model_response, fileData=None, remove_entity_idx=False):
    ##[red panda-0:lie next to:red panda-1]_[Frame-0:Frame-7];[red panda-0:lie left:red panda-1]_[Frame-0:Frame-7];[red panda-1:lie right:red panda-0]_[Frame-0:Frame-7];[red panda-1:lie next to:red panda-0]_[Frame-0:Frame-7];[red panda-0:lie next to:red panda-2]_[Frame-0:Frame-7];[red panda-0:lie left:red panda-2]_[Frame-0:Frame-7];[red panda-2:sit right:red panda-0]_[Frame-0:Frame-7];[red panda-2:sit next to:red panda-0]_[Frame-0:Frame-7];[red panda-2:taller:red panda-0]_[Frame-0:Frame-7];[red panda-1:lie next to:red panda-2]_[Frame-0:Frame-7];[red panda-1:lie left:red panda-2]_[Frame-0:Frame-7];[red panda-2:sit right:red panda-1]_[Frame-0:Frame-7];[red panda-2:sit next to:red panda-1]_[Frame-0:Frame-7];[red panda-2:taller:red panda-1]_[Frame-0:Frame-7];
    block_triplets = {
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }

    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    try:
        triplets_list = prediction_data.split(";")
        for triplet_data in triplets_list:
            #[red panda-0:lie next to:red panda-1]_[Frame-0:Frame-7]
            triplet_data = triplet_data.replace(":", ",")
            splitTemporal = triplet_data.split("_")
            if len(splitTemporal)!=2:
                print(f"invalid entity length {splitTemporal}")
                continue
            
            triplet_data, temporal = splitTemporal

            triplet_data = triplet_data.replace("[","")
            triplet_data = triplet_data.replace("]","")
            triplet_data = triplet_data.split(",")
            if len(triplet_data)!=3:
                print(f"invalid triplet: {triplet_data}")
                continue

            subj, pred, obj = triplet_data
            triplet = [subj, pred, obj]

            
            if "[" in temporal and "]" in temporal:
                temporal_list = temporal_list.replace("Frame-", "")
                temporal_list = eval(temporal)

                if len(temporal_list)==1:
                    # only one frame 
                    temporal_entity_index = temporal_list[0]
                    if type(temporal_entity_index)!=int:
                        temporal_entity_index = int(temporal_entity_index)
                        block_triplets["triplets"][temporal_entity_index].append(triplet)
                elif len(temporal_list)==2:
                    temporal_entity_start_index,temporal_entity_end_index = temporal_list
                    if type(temporal_entity_start_index)!=int:
                        temporal_entity_start_index = int(temporal_entity_start_index)
                    if type(temporal_entity_end_index)!=int:
                        temporal_entity_end_index = int(temporal_entity_end_index)
                    
                    for i in range(temporal_entity_start_index,temporal_entity_end_index):
                        if i>len(block_triplets["triplets"]):
                            print(f"temporal entity index out of bound: {i}")
                            continue

                        block_triplets["triplets"][i].append(triplet)
                else:
                    print(f"invalid temporal entity: {temporal_list}")

            else:
                print(f"temporal entity is not surrounded by [] : {temporal}")
                continue

    except Exception as e:
        print(f"erro parsing triplet data: {e},{fileData}")

    
    return block_triplets


class SEEDS:
    """
    Don't change values
    """
    RANDOM_ANNOTATIONS_SHUFFLE_SEED = 145
    AG_OBJECT_PREDICATE_PARTIAL_SELECTION_SEEDS = [978, 324]

def addTriplet(triplet, Objects, relations):
    """
    Add triplet if entities belongs to predefined selected list.
    """
    subj, pred, obj = triplet
    if subj in Objects and obj in Objects and pred in relations:
        return True
    return False

def chunk_list(list_, chunk_n):
    chunk_n = max(1, chunk_n)
    return (list_[i:i+chunk_n] for i in range(0, len(list_), chunk_n))

def get_shuffled_list(input_list):
    random.shuffle(input_list)
    return input_list

def get_varying_list(current_block_list, full_list, fix_size=100):
	"""
	1. take current list (shuffle it)
	2. add elements to current list from full list without repeatation that sums to fix_size (shuffle it again)
	3. return the list
	"""
	current_block_list = set(copy.deepcopy(current_block_list))
	full_list = set(copy.deepcopy(full_list))

	newelements = full_list.difference(current_block_list)

	current_block_list = list(current_block_list)
	newelements =  list(newelements)
	newElementsNeeded = 0
	currentElementsSize = len(current_block_list) 
	if currentElementsSize>fix_size:
		## more items than predefined limit
		newElementsNeeded = 0
		pass
	else:
		newElementsNeeded = fix_size - len(current_block_list) 

	if len(newelements)<newElementsNeeded:
		current_block_list = current_block_list + random.sample(newelements,k=len(newelements))
	else:
		current_block_list = current_block_list + random.sample(newelements,k=newElementsNeeded)

	random.shuffle(current_block_list)
	return current_block_list


def check_bbox_overlap(xyxy1, xyxy2, iou_threshold=0.5):

    x_left = max(xyxy1[0], xyxy2[0])
    y_top = max(xyxy1[1], xyxy2[1])
    x_right = min(xyxy1[2], xyxy2[2])
    y_bottom = min(xyxy1[3], xyxy2[3])

    if x_right < x_left or y_bottom < y_top:
        return False  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    bbox2_area = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou >= iou_threshold


def unnormbb_vidvrd(bb_data, width, height, round_by=3):
    newbb_data = copy.deepcopy(bb_data)
    print(bb_data, width, height)
    newbb_data['xmin'] = int(round(newbb_data['xmin'],round_by)*width)
    newbb_data['ymin'] = int(round(newbb_data['ymin'],round_by)*height)
    newbb_data['xmax'] = int(round(newbb_data['xmax'],round_by)*width)
    newbb_data['ymax'] = int(round(newbb_data['ymax'],round_by)*height)
    print(newbb_data)
    return newbb_data

def get_substring_between(s, start_substring, end_substring):
    try:
        # Find the index of the start and end substrings
        start_index = s.find(start_substring)
        end_index = s.find(end_substring, start_index)

        # If start or end substring is not found, return None
        if start_index == -1 or end_index == -1:
            return None

        start_index = start_index + len(start_substring)
        # Extract the substring from the start to the end substring
        return s[start_index:end_index]
    
    except Exception as e:
        return str(e)

def remove_ids_V2(frames_tripletes, version="v2_1"):
    for idx, trip in enumerate(frames_tripletes):
        if version=="v2_1":
            subj, rel, obj = trip
        elif version=="v3_1":
            subj, obj, rel = trip
        
        subj = subj.split("-")[0]
        obj = obj.split("-")[0]

        # if version=="v2_1":
        frames_tripletes[idx] =  [subj, rel, obj]
        # elif version=="v3_1":
        # frames_tripletes[f_idx][idx] =  [subj, obj, rel]

    return frames_tripletes

def remove_idx(data):
    if "." in data:
        data = data.split(".")[-1]
    return data

def remove_ids(frames_tripletes, version="v2_1", remove_indexes=False):
    for f_idx, triplets in enumerate(frames_tripletes):
        for idx, trip in enumerate(triplets):
            if version=="v2_1":
                # predicate in middle
                subj, rel, obj = trip
            elif version=="v3_1":
                # predicate last
                subj, obj, rel = trip
            
            subj = subj.split("-")[0]
            obj = obj.split("-")[0]

            subj = subj.strip("[")
            obj = obj.strip("]")

            if remove_indexes:
                subj = remove_idx(subj)
                obj = remove_idx(obj)
                rel = remove_idx(rel)
            frames_tripletes[f_idx][idx] =  [subj, rel, obj]

    return frames_tripletes


def eval_tagging_scores2(gt_relations, pred_relations, min_pred_num=1, pred_pred_hits_at_k=None, pred_pred_fp_at_k=None):
    is_triplet = isinstance(pred_relations[0]['triplet'], List) or isinstance(pred_relations[0]['triplet'], Tuple)
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)

    gt_triplets = set(tuple(r['triplet']) if is_triplet else r['triplet'] for r in gt_relations)
    pred_triplets = []
    for pred_trip_cnt, r in enumerate(pred_relations):
        # if pred_trip_cnt>min_pred_num:
        #     break
        triplet = tuple(r['triplet']) if is_triplet else r['triplet']
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
    gt_hit_scores = []
    for r in gt_relations:
        gt_hit_scores.append(-np.inf)
    gt_hit_scores.extend([-np.inf]*(min_pred_num-len(gt_hit_scores)))
    gt_hit_scores = np.asarray(gt_hit_scores)

    fp_cnt, tp_cnt = 0,0 
    for i, t in enumerate(gt_triplets):
        if t in pred_triplets:
            gt_hit_scores[i] = 1
            tp_cnt +=1

            if is_triplet:
                for k in pred_pred_hits_at_k.keys():
                    predicate = t[1]
                    pred_pred_idx = pred_triplets.index(t)
                    if pred_pred_idx<=k:
                        pred_pred_hits_at_k[k][AG_relationsCombined.index(predicate)] += 1

    for i, t in enumerate(pred_triplets):
        if t not in gt_triplets:
            fp_cnt +=1
            if is_triplet:
                for k in pred_pred_fp_at_k.keys():
                    predicate = t[1]
                    pred_pred_idx = pred_triplets.index(t)
                    if pred_pred_idx<=k:
                        pred_pred_fp_at_k[k][AG_relationsCombined.index(predicate)] += 1

    rec = tp_cnt/np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = tp_cnt/np.maximum(tp_cnt+fp_cnt, np.finfo(np.float32).eps)

    return prec, rec, gt_hit_scores

def eval_tagging_scores(gt_relations, pred_relations, min_pred_num=1):
    is_triplet = isinstance(pred_relations[0]['triplet'], List) or isinstance(pred_relations[0]['triplet'], Tuple)
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)

    gt_triplets = set(tuple(r['triplet']) if is_triplet else r['triplet'] for r in gt_relations)
    pred_triplets = []
    for pred_trip_cnt, r in enumerate(pred_relations):
        if pred_trip_cnt>min_pred_num:
            break
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
    gt_hit_scores = []
    for r in gt_relations:
        gt_hit_scores.append(-np.inf)
    gt_hit_scores.extend([-np.inf]*(min_pred_num-len(gt_hit_scores)))
    gt_hit_scores = np.asarray(gt_hit_scores)

    fp_cnt, tp_cnt = 0,0 
    for i, t in enumerate(gt_triplets):
        if t in pred_triplets:
            gt_hit_scores[i] = 1
            tp_cnt +=1

    for i, t in enumerate(pred_triplets):
        if t not in gt_triplets:
            fp_cnt +=1

    rec = tp_cnt/np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = tp_cnt/np.maximum(tp_cnt+fp_cnt, np.finfo(np.float32).eps)

    return prec, rec, gt_hit_scores

def eval_tagging_scores_with_bb(gt_relations, pred_relations, min_pred_num=1, mode='triplet'):

    # here 'triplets' can be either triplet, subject, or object ...

    # gt_relations/pred_relations = [{'triplet': ((sub, bbox), rel, (obj, bbox)), 'score': 0.9}, ...] if mode=='triplet'
    #                             = [{'triplet': (obj, bbox), 'score': 0.9}, ...] if mode=='object'/'subject'
    #                             = [{'triplet': rel, 'score': 0.9}, ...] if mode=='predicate'

    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    if mode == 'predicate':
        gt_triplets = set(r['triplet'] for r in gt_relations)
    elif mode == 'subject' or mode == 'object':
        gt_triplets = set((r['triplet'][0], tuple(r['triplet'][1])) for r in gt_relations)
    else:
        gt_triplets = set(((r['triplet'][0][0], tuple(r['triplet'][0][1])), r['triplet'][1], (r['triplet'][2][0], tuple(r['triplet'][2][1]))) for r in gt_relations)

    if mode == 'predicate':
        gt_triplets_dict = {r['triplet']: (0,0,0,0) for r in gt_relations}
    elif mode == 'subject' or mode == 'object':
        gt_triplets_dict = {}
        for r in gt_relations:
            if r['triplet'][0] not in gt_triplets_dict:
                gt_triplets_dict[r['triplet'][0]] = []
            gt_triplets_dict[r['triplet'][0]].append(r['triplet'][1])
    else:
        gt_triplets_dict = {}
        for r in gt_relations:
            if (r['triplet'][0][0], r['triplet'][1], r['triplet'][2][0]) not in gt_triplets_dict:
                gt_triplets_dict[(r['triplet'][0][0], r['triplet'][1], r['triplet'][2][0])] = []
            gt_triplets_dict[(r['triplet'][0][0], r['triplet'][1], r['triplet'][2][0])].append((r['triplet'][0][1], r['triplet'][2][1]))

    pred_triplets = {} # {(sub, pred, obj): bbox}

    for r in pred_relations:
        if mode == 'triplet':
            triplet_no_bb = tuple([r['triplet'][0][0], r['triplet'][1], r['triplet'][2][0]])
        elif mode == 'subject' or mode == 'object':
            triplet_no_bb = r['triplet'][0]
        else: # relation
            triplet_no_bb = r['triplet']

        if not triplet_no_bb in pred_triplets:
            pred_triplets[triplet_no_bb] = []
        if mode == 'triplet':
            pred_triplets[triplet_no_bb].append((r['triplet'][0][1], r['triplet'][2][1]))
        elif mode == 'subject' or mode == 'object':
            pred_triplets[triplet_no_bb].append(r['triplet'][1])
        else:
            pred_triplets[triplet_no_bb] = True # set anything -- this is relation so we don't have bbox

    gt_hit_scores = []
    for r in gt_relations:
        gt_hit_scores.append(-np.inf)
    gt_hit_scores.extend([-np.inf]*(min_pred_num-len(gt_hit_scores)))
    gt_hit_scores_no_bb = np.asarray(gt_hit_scores)
    gt_hit_scores_with_bb = gt_hit_scores.copy()

    fp_cnt_no_bb, tp_cnt_no_bb = 0,0 
    fp_cnt_with_bb, tp_cnt_with_bb = 0,0 
    for i, t in enumerate(gt_triplets):
        if mode == 'triplet':
            gt_triplet_no_bb = tuple([t[0][0], t[1], t[2][0]])
        elif mode == 'subject' or mode == 'object':
            gt_triplet_no_bb = t[0]
        else:
            gt_triplet_no_bb = t

        if gt_triplet_no_bb in pred_triplets:
            gt_hit_scores_no_bb[i] = 1
            tp_cnt_no_bb +=1
            
            if mode == 'predicate':
                gt_hit_scores_with_bb[i] = 1
                tp_cnt_with_bb += 1
                continue

            for bbox in pred_triplets[gt_triplet_no_bb]:
                if mode == 'triplet':
                    is_match = check_bbox_overlap(t[0][1], bbox[0]) and check_bbox_overlap(t[2][1], bbox[1])
                else:
                    is_match = check_bbox_overlap(t[1], bbox)
                if is_match:
                    gt_hit_scores_with_bb[i] = 1
                    tp_cnt_with_bb += 1
                    break
        
    for i, t in enumerate(pred_relations):

        if mode == 'predicate':
            pred_no_bb = t['triplet']
        elif mode == 'subject' or mode == 'object':
            pred_no_bb = t['triplet'][0]
        else:
            pred_no_bb = (t['triplet'][0][0], t['triplet'][2][0])

        if pred_no_bb not in gt_triplets_dict:
            fp_cnt_no_bb += 1
            fp_cnt_with_bb += 1

        else:
            is_overlap = False
            if mode == 'predicate':
                continue
            for bbox in pred_triplets[pred_no_bb]:
                if mode == 'triplet':
                    ismatch_subject = any([check_bbox_overlap(t['triplet'][0][1], gt[0]) for gt in gt_triplets_dict[(t['triplet'][0][0], t['triplet'][1], t['triplet'][2][0])]])
                    ismatch_object = any([check_bbox_overlap(t['triplet'][0][1], gt[1]) for gt in gt_triplets_dict[(t['triplet'][0][0], t['triplet'][1], t['triplet'][2][0])]])
                    # is_match = check_bbox_overlap(t['triplet'][0][1], gt_triplets_dict[(t['triplet'][0][0], t['triplet'][1], t['triplet'][2][0])][0]) \
                    #     and check_bbox_overlap(t['triplet'][2][1], gt_triplets_dict[(t['triplet'][0][0], t['triplet'][1], t['triplet'][2][0])][1])
                    is_match = ismatch_subject and ismatch_object
                elif mode == 'subject' or mode == 'object':
                    # is_match = check_bbox_overlap(t['triplet'][1], gt_triplets_dict[t['triplet'][0]])
                    is_match = any([check_bbox_overlap(t['triplet'][1], gt) for gt in gt_triplets_dict[t['triplet'][0]]])
                if is_match: 
                    is_overlap = True
                    break
            if not is_overlap:
                fp_cnt_with_bb += 1

    rec_sgcls = tp_cnt_no_bb/np.maximum(len(gt_triplets_dict), np.finfo(np.float32).eps)
    prec_sgcls = tp_cnt_no_bb/np.maximum(tp_cnt_no_bb+fp_cnt_no_bb, np.finfo(np.float32).eps)

    rec_sgdet = tp_cnt_with_bb/np.maximum(len(gt_triplets_dict), np.finfo(np.float32).eps)
    prec_sgdet = tp_cnt_with_bb/np.maximum(tp_cnt_with_bb+fp_cnt_with_bb, np.finfo(np.float32).eps)

    return (prec_sgcls, rec_sgcls, gt_hit_scores_no_bb), (prec_sgdet, rec_sgdet, gt_hit_scores_with_bb)

# def eval_tagging_scores_vrdformer(gt_relations, pred_relations, min_pred_num=0):
#     pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
#     # ignore trajectories
#     gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
#     pred_triplets = []
#     hit_scores = []
#     for r in pred_relations:
#         triplet = tuple(r['triplet'])
#         if not triplet in pred_triplets:
#             pred_triplets.append(triplet)
#             hit_scores.append(r['score'])
 
#     hit_scores.extend([-np.inf]*(min_pred_num-len(hit_scores)))
#     hit_scores = np.asarray(hit_scores)
#     for i, t in enumerate(pred_triplets):
#         if not t in gt_triplets:
#             hit_scores[i] = -np.inf
#     tp = np.isfinite(hit_scores)
#     fp = ~tp
#     cum_tp = np.cumsum(tp).astype(np.float32)
#     cum_fp = np.cumsum(fp).astype(np.float32)
#     rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
#     prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
#     return prec, rec, hit_scores

def calculate_accuracy_varying_lengths(gt_triplets, pred_triplets, remove_duplicates=True):
    """
    Calculate accuracy for scene graph triplets and their individual components 
    when the counts of ground truth and predicted triplets are not the same.

    :param gt_triplets: List of ground truth triplets [(subject, predicate, object), ...]
    :param pred_triplets: List of predicted triplets [(subject, predicate, object), ...]
    :return: A dictionary containing the accuracies for triplets, subjects, predicates, and objects
    """

    if remove_duplicates:
        gt_set = set(gt_triplets)
        pred_set = set(pred_triplets)
        correct_triplets = gt_set & pred_set  # Intersection of both sets gives correct triplets
    else:
        correct_triplets = 0
        for predt in pred_triplets:
            if predt in gt_triplets:
                correct_triplets +=1

    
    total_triplets = len(gt_triplets)
    total_predicted_triplets = len(pred_triplets)
    
    correct_subjects = sum(1 for gt in gt_triplets if any(gt[0] == pred[0] for pred in pred_triplets))
    correct_predicates = sum(1 for gt in gt_triplets if any(gt[1] == pred[1] for pred in pred_triplets))
    correct_objects = sum(1 for gt in gt_triplets if any(gt[2] == pred[2] for pred in pred_triplets))

    unique_subjects = list(set([gt[0] for gt in gt_triplets]))
    unique_predicates = list(set([gt[1] for gt in gt_triplets]))
    unique_objects = list(set([gt[2] for gt in gt_triplets]))
    total_pred_predicates = list(set([pred[1] for pred in pred_triplets]))

    # triplet_accuracy = len(correct_triplets) / total_triplets if total_triplets > 0 else 0
    # subject_accuracy = correct_subjects / total_triplets if total_triplets > 0 else 0
    # predicate_accuracy = correct_predicates / total_triplets if total_triplets > 0 else 0
    # object_accuracy = correct_objects / total_triplets if total_triplets > 0 else 0

    
    if type(correct_triplets)==list or type(correct_triplets)==set:
        correct_triplets = len(correct_triplets)

    return {
        'correct_triplet_cnt': correct_triplets,
        'correct_subject_cnt': correct_subjects,
        'correct_predicate_cnt': correct_predicates,
        'correct_object_cnt': correct_objects,
        'total_triplets': total_triplets,
        'total_subjects': len(unique_subjects),
        'total_objects': len(unique_objects),
        'total_predicates': len(unique_predicates),
        'total_pred_predicates': len(total_pred_predicates),
        'total_predicted_triplets': total_predicted_triplets
    }

class SGSpecialTokens:
    VIDEO_FRAME_ID = "#frameid"
    SG_END = "#sgend"
    SG_SUBJECT = "#subject"
    SG_SUBJECT_ID = "#subid"
    SG_OBJECT = "#object"
    SG_OBJECT_ID = "#objid"
    SG_PREDICATE = "#sgpred"
    SG_BB_START = "#sgbb"
    SG_BB_END = "#sgbbend"
    SG_BB_X1Y1 = "#bbx1y1"
    SG_BB_X2Y2 = "#bbx2y2"
    # SG_START = "#sg"
    # SG_BB_X1 = "#sgx1"
    # SG_BB_X2 = "#sgx2"
    # SG_BB_Y1 = "#sgy1"
    # SG_BB_Y2 = "#sgy2"

    @staticmethod
    def get_tokens():
        members = [attr for attr in dir(SGSpecialTokens) if not callable(getattr(SGSpecialTokens, attr)) and not attr.startswith("__")]
        tokens = []
        for mem in members:
            tokens.append(SGSpecialTokens.__getattribute__(SGSpecialTokens,mem))
        return tokens

def get_block_number_for_frame(frame_idx, frame_blocks):
    frame_block = None
    for block, b_frames in enumerate(frame_blocks):
        if frame_idx in b_frames:
            frame_block = block
            break
    return frame_block

def get_bb_subj_obj(data_root,vid_id,frame_idx,subject_id,object_id):
  sub_bb, obj_bb, mask_size = [], [], None
  try:
    sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=subject_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass
  
  try:
    obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=object_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass

  return sub_bb, obj_bb, mask_size

def get_frame_range_for_annotations(vid_objects, vid_data):
  min_frame_idx, max_frame_idx = -1, 0
  frames_for_obj = {}
  for vid_obj_idx, vobj in enumerate(vid_objects):
    category = vobj["category"]
    object_id = vobj["object_id"]
    frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
    if frames_=="None":
        continue
    
    for frame_range in frames_:
      frame_start, frame_end = frame_range

      if f"{category}{object_id}" not in frames_for_obj:
        frames_for_obj[f"{category}{object_id}"] = {
          "frames": []
        }

      frames_for_obj[f"{category}{object_id}"]["frames"].append(frame_range)

      if min_frame_idx ==-1:
          min_frame_idx = frame_start
      if frame_start<=min_frame_idx:
        min_frame_idx = frame_start
      if frame_end>=max_frame_idx:
        max_frame_idx = frame_end

  return min_frame_idx, max_frame_idx, frames_for_obj

def create_batch_frames(vid_data, totalFrames, frame_batch=8):
    ## out of total frames send frames in batch of 8.
    # total_frame_batch = int(totalFrames/8)
    remaining_frames = totalFrames%frame_batch
    total_frame_indices = [i for i in range(totalFrames)]

    vid_rels = vid_data["relations"]
    objects = vid_data["objects"]
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in objects}

    # rels_for_the_block = []
    frames_to_consider = []
    rel_by_frames = []
    batch_of_frames = []
    # batch_of_frames_rels = []
    batch_rels = []

    for frame_idx in range(0, totalFrames):
        rel_for_frame = []
        for idx, vid_r in enumerate(vid_rels):
            sub = vid_r[0]
            obj = vid_r[1]
            rel = vid_r[2]
            frames = vid_r[3].copy()
            # frame_start, frame_end = frames[0][0], frames[0][1]
            for frame_range in frames:
                frame_start, frame_end = frame_range
                
                if frame_start>totalFrames:
                    continue
                if frame_end>totalFrames:
                    continue

                # if frame_start>=frame_idx and frame_idx<=frame_end: # FIXED CONDITION
                if frame_idx>=frame_start and frame_idx<=frame_end:
                    subn = vid_objects_by_id[sub]["category"]
                    objn = vid_objects_by_id[obj]["category"]
                    rel_for_frame.append([subn,rel,objn])

        frames_to_consider.append(frame_idx)
        rel_by_frames.append(rel_for_frame)
        
        if len(frames_to_consider)>=8:
            batch_of_frames.append(frames_to_consider)
            batch_rels.append(rel_by_frames)
            frames_to_consider = []
            rel_by_frames = [] 


    if len(frames_to_consider)>0 and len(frames_to_consider)<8:
        # num_frames_to_add = frame_batch - len(frames_to_consider)
        # batch_first_frame_idx = frames_to_consider[0]
        # 13,14,15,16
        while len(frames_to_consider)<8:
            batch_first_frame_idx = frames_to_consider[0]
            frames_to_consider.insert(0,batch_first_frame_idx-1)

            if len(frames_to_consider)>=8:
                break

        batch_of_frames.append(frames_to_consider)
        batch_rels.append(rel_by_frames)
            # while len(frames_to_consider)<8:
            #     random_idx = random.choice(total_frame_indices)
            #     if random_idx not in frames_to_consider:
            #         frames_to_consider.append(random_idx)
            #     if len(frames_to_consider)>=8:
            #         break
            # batch_of_frames.append(frames_to_consider)


    # frames_to_consider = []
    # batch_of_frames = []
    # batch_of_frames_rels = []
    # batch_rels = []
    # for idx, vid_r in enumerate(vid_rels):
    #     sub = vid_r[0]
    #     obj = vid_r[1]
    #     rel = vid_r[2]
    #     frames = vid_r[3].copy()
    #     frame_start, frame_end = frames[0][0], frames[0][1]
    #     if frame_start>totalFrames:
    #        continue
    #     if frame_end>totalFrames:
    #        continue

    #     if frame_start not in frames_to_consider:
    #         frames_to_consider.append(frame_start)
    #         subn = vid_objects_by_id[sub]["category"]
    #         objn = vid_objects_by_id[obj]["category"]
    #         rels_for_the_block.append([subn,rel,objn])

    #     if len(frames_to_consider)>=8:
    #         batch_of_frames.append(frames_to_consider)
    #         batch_rels.append(rels_for_the_block)
    #         frames_to_consider = []
    #         rels_for_the_block = []


    # if len(rels_for_the_block)>0:
    #     batch_rels.append(rels_for_the_block)


    # # print("frames to consider ", len(frames_to_consider))
    # if len(frames_to_consider)>0 and len(frames_to_consider)<8:
    #     while len(frames_to_consider)<8:

    #         random_idx = random.choice(total_frame_indices)
    #         if random_idx not in frames_to_consider:
    #             frames_to_consider.append(random_idx)

    #         if len(frames_to_consider)>=8:
    #             break
    #     batch_of_frames.append(frames_to_consider)
        
    # total_frame_indices = [i for i in range(totalFrames)]
    # current_frame_batch_idx = 0
    # while current_frame_batch_idx<=total_frame_batch:
    #     start_idx = current_frame_batch_idx * frame_batch
    #     frames_to_infer = total_frame_indices[start_idx:start_idx+frame_batch]
    #     # print(f"T {start_idx}:{start_idx+8} => {frames_to_infer}")
    #     current_frame_batch_idx +=1
    #     if len(frames_to_infer)<frame_batch:
    #         print("less frames batch")
    #         continue
    #     batch_of_frames.append(frames_to_infer)
    # last_batch_remaining = total_frame_indices[-remaining_frames-(frame_batch-remaining_frames):] # add previous batch frames to accomodate n batch
    # batch_of_frames.append(last_batch_remaining)
    # print("batch of frames", batch_of_frames)

    return batch_of_frames, remaining_frames, batch_rels

# def create_batch_frames(totalFrames, frame_batch=8):
#     ## out of total frames send frames in batch of 8.
#     total_frame_batch = int(totalFrames/8)
#     remaining_frames = totalFrames%8

#     batch_of_frames = []

#     total_frame_indices = [i for i in range(totalFrames)]
#     current_frame_batch_idx = 0
#     while current_frame_batch_idx<=total_frame_batch:
#         start_idx = current_frame_batch_idx * frame_batch
#         frames_to_infer = total_frame_indices[start_idx:start_idx+frame_batch]
#         # print(f"T {start_idx}:{start_idx+8} => {frames_to_infer}")
#         current_frame_batch_idx +=1
#         if len(frames_to_infer)<frame_batch:
#             # print("less frames batch")
#             continue
#         batch_of_frames.append(frames_to_infer)

    
#     last_batch_remaining = total_frame_indices[-remaining_frames-(frame_batch-remaining_frames):] # add previous batch frames to accomodate n batch
#     batch_of_frames.append(last_batch_remaining)

#     return batch_of_frames, remaining_frames

def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True, dataset="vidor"):
    mask_name = os.path.join(data_root, dataset, 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        if normlize_bb:
           x_min = round(x_min/mask_w,3)
           x_max = round(x_max/mask_w,3)
           y_min = round(y_min/mask_h,3)
           y_max = round(y_max/mask_h,3)

        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]

def getFramesForObject(vid_data, Subject_id):
    vid_rels = vid_data["relations"]
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        # rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if Subject_id==sub or Subject_id==obj:
            return frames_
    return "None"


def unnormbb(pred_box, mask, normlize=False, decimal=2, img_h=None, img_w=None):
    if normlize:
        pred_box[0] = round(pred_box[0]/img_w, decimal)
        pred_box[2] = round(pred_box[2]/img_w, decimal)
        pred_box[1] = round(pred_box[1]/img_h, decimal)
        pred_box[3] = round(pred_box[3]/img_h, decimal)
        return pred_box

    pred_box[0] = int(pred_box[0]*mask.shape[1])
    pred_box[2] = int(pred_box[2]*mask.shape[1])
    pred_box[1] = int(pred_box[1]*mask.shape[0])
    pred_box[3] = int(pred_box[3]*mask.shape[0])
    return pred_box


def parse_bb_from_string(str_data):
    # "[0.048, 0.0, 0.517, 0.997]"
    bb = []
    str_data = str_data.strip("</s>").strip("[").strip("]").split(",")
    if len(str_data)<4:
        return []
    for bb_coord in str_data:
        try:
            bb.append(round(float(bb_coord),3))
        except ValueError:
            return []
    return bb

def parse_sg_data(pred_sg_str_data):
    pred_sgs = []
    predictions = pred_sg_str_data.strip("</s>").split(";")

    for pred in predictions:
        if len(pred)<30:
            continue

        # subPredObj, pred_Frame = pred.split("_")
        # pred_Frame = pred_Frame.strip("[").strip("]")
        pred_Frame = 0

        triplates = pred.split(":")
        if len(triplates)==3:
            subj, predi, obj = triplates

            subj_data = subj.split("-")
            if len(subj_data)<3:
                continue
            subj_id = f"{subj_data[0].strip('[').strip(']')}-{subj_data[1]}"
            subj_bb = parse_bb_from_string(subj_data[2])

            obj_data = obj.split("-")
            
            if len(obj_data)<3:
                continue

            obj_id = f"{obj_data[0]}{obj_data[1]}"
            
            obj_bb = parse_bb_from_string(obj_data[2])

            sg = {
                "subject": {
                    "id": subj_id,
                    "bbox": subj_bb
                },
                "predicate": predi,
                "object":{
                    "id": obj_id,
                    "bbox": obj_bb
                },
                "uni_frame_idx": pred_Frame
            }

            pred_sgs.append(sg)

    return pred_sgs 


def validate_model_response(model_response):
    validation_flags = []

    if "{" not in model_response:
        validation_flags.append(1)

    if "}" not in model_response:
        validation_flags.append(1)
    
    for i in range(8):
        if not f"Frame {i}" in model_response:
            validation_flags.append(1)

    # print("validation flags ", validation_flags)
    if sum(validation_flags)>0:
        return False
    
    return True


# def pre_clean_prediction_data_v3vidvrd(model_response):
#     frame_triplets = []
#     prediction_data = model_response
#     prediction_data = prediction_data.strip("</s>")
#     framewiseTriplets = prediction_data.split(f"{SGSpecialTokens.VIDEO_FRAME_ID}")[1:]

#     special_tokens = SGSpecialTokens.get_tokens()
#     for cnt_idx, ftriplets in enumerate(framewiseTriplets):

#         for spetok in special_tokens:
#             ftriplets = ftriplets.replace(f"{spetok}", "")

#         ftriplets = ftriplets.replace(f":", ",")
#         ftriplets = ftriplets.split(";")

#         current_frame_triplets = []

#         for ftr in ftriplets:
#             ftr_temp = ftr.split(",")
#             if len(ftr_temp)==3:
#                 # print("conveting to list",ftr)
#                 ftr_temp[0] = str(ftr_temp[0]).strip("[").strip("]")
#                 ftr_temp[1] = str(ftr_temp[1]).strip("[").strip("]")
#                 ftr_temp[1] = ftr_temp[1].strip().replace(" ","_") # predicates are trained by removing _ and appended again for evaluation
#                 ftr_temp[2] = str(ftr_temp[2]).strip("[").strip("]")  
#                 current_frame_triplets.append(ftr_temp)

#         frame_triplets.append(current_frame_triplets)
    
#     return frame_triplets

def remove_triplet_indexes(triplet):
    return [remove_idx(element) for element in triplet]


def pre_clean_prediction_data_onevision_v14_AG(model_response, fileData=None, remove_entity_idx=False, contains_temporal_entity=False):
    """
    Quadruplets for spatial + action predicates
    """
    block_triplets = {
        "quadruplets": [[] for i in range(8)],
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    if "#sg_start" in prediction_data and "#sg_end" in prediction_data:

        # print(cleanString)
        try:
            cleanString = get_substring_between(s=prediction_data,start_substring="#sg_start",end_substring="#sg_end").strip()
            # comment_str = "// This triplet is not necessary as it does not provide additional information.\n"
            # if comment_str in cleanString:
            #     cleanString = cleanString.replace(comment_str, "")
        except Exception as e:
            print("error getting sgblock data")
    else:
        cleanString = prediction_data

    # cleanString = re.sub(r"(frame-\d+)", r"'\1'", cleanString)

    # print(cleanString)
    try:
        # print("evaluating")
        evaluated_string_json = eval(cleanString)
        for key,frame_data in evaluated_string_json.items():
            # print(key)
            if key=="scene":
                block_triplets["scene"].append(frame_data)
            elif key=="description":
                block_triplets["description"].append(frame_data)
            elif key=="objects":
                for obj in frame_data:
                    block_triplets["objects"].append(obj)
            elif key=="triplets":

                def append_to_block(triplData, block_triplets):
                    for j in range(8):
                        for trp in triplData:
                            block_triplets["triplets"][j].append(trp)
                    return block_triplets

                # import pdb
                # pdb.set_trace()

                try:
                    attention = frame_data["attention"]
                    block_triplets = append_to_block(attention,block_triplets)

                except Exception as e:
                    print(f"error parsing: triplets: {frame_data}")

                try:
                    spatial = frame_data["spatial"]
                    block_triplets = append_to_block(spatial,block_triplets)
                except Exception as e:
                    print(f"error parsing: triplets: {frame_data}")

                try:
                    contacting = frame_data["contacting"]
                    block_triplets = append_to_block(contacting,block_triplets)
                except Exception as e:
                    print(f"error parsing: triplets: {frame_data}")

                    
    except Exception as e:
        print(e, fileData)
        # print("model response", model_response)
        pass


    return block_triplets


def pre_clean_prediction_data_onevision_v5_3_FT_Quad_vrd(model_response, fileData=None, remove_entity_idx=False, contains_temporal_entity=False):
    """
    Quadruplets for spatial + action predicates
    """
    block_triplets = {
        "quadruplets": [[] for i in range(8)],
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()


    frame_blocks = re.findall(r'#frameid(.*?)(?=#frameid|#sgend)', prediction_data)

    for idx, block in enumerate(frame_blocks):
        quadruplets = re.findall(r'\[(.*?)\]', block)
        for quad in quadruplets:
            quad_list = quad.split(':')

            predicate_joined = " ".join(quad_list[2:])
            if quad_list[-1].lower()=="na":
                predicate_joined = quad_list[2]
            

            triplet = [quad_list[0], predicate_joined, quad_list[1]]
            if idx < len(block_triplets["quadruplets"]):
                block_triplets["quadruplets"][idx].append(quad_list)
                block_triplets["triplets"][idx].append(triplet)

    # print(block_triplets)




    return block_triplets


def pre_clean_prediction_data_onevision_v14_vrd(model_response, fileData=None, remove_entity_idx=False, contains_temporal_entity=False):
    """
    Quadruplets for spatial + action predicates
    """
    block_triplets = {
        "quadruplets": [[] for i in range(8)],
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    if "#sg_start" in prediction_data and "#sg_end" in prediction_data:

        # print(cleanString)
        try:
            cleanString = get_substring_between(s=prediction_data,start_substring="#sg_start",end_substring="#sg_end")
            comment_str = "// This triplet is not necessary as it does not provide additional information.\n"
            if comment_str in cleanString:
                cleanString = cleanString.replace(comment_str, "")
        except Exception as e:
            print("error getting sgblock data")
    else:
        cleanString = prediction_data

    try:
        # print("evaluating")
        evaluated_string_json = eval(cleanString.replace("#sg_end", ""))
        for key,frame_data in evaluated_string_json.items():
            # print(key)
            if key=="scene":
                block_triplets["scene"].append(frame_data)
            elif key=="description":
                block_triplets["description"].append(frame_data)
            elif key=="objects":
                block_triplets["objects"].append(frame_data)
            elif key=="quadruplets":
                # strkey = str(key)
                # strkey_f_index = strkey.strip("F")  # F1 ==> 1
                # current_frame_triplets = []
                for frame_Quadruplets in frame_data:
                    if contains_temporal_entity:
                        frame_Quadruplets, Quadruplets_time = frame_Quadruplets

                    if len(frame_Quadruplets)==4:
                        if remove_entity_idx:
                            frame_Quadruplets = remove_triplet_indexes(triplet=frame_Quadruplets)

                        # convert to triplets
                        frame_triplet = [frame_Quadruplets[0],f"{frame_Quadruplets[1]} {frame_Quadruplets[2]}",frame_Quadruplets[3]]

                        if contains_temporal_entity:
                            if len(Quadruplets_time)==1:
                                # single frame
                                triplet_time_fidx = int(Quadruplets_time[0].split("-")[-1])
                                if triplet_time_fidx>7:
                                    continue
                                block_triplets["triplets"][triplet_time_fidx].append(frame_triplet)
                                block_triplets["quadruplets"][triplet_time_fidx].append(frame_Quadruplets)
                            else:
                                triplet_time_fidx_start = int(Quadruplets_time[0].split("-")[-1])
                                triplet_time_fidx_end = int(Quadruplets_time[0].split("-")[-1])
                                for j in range(triplet_time_fidx_start,triplet_time_fidx_end):
                                    if j>8:
                                        continue
                                    block_triplets["triplets"][j].append(frame_triplet)
                                    block_triplets["quadruplets"][j].append(frame_Quadruplets)
                        else:
                            for j in range(8):
                                block_triplets["triplets"][j].append(frame_triplet)
                                block_triplets["quadruplets"][j].append(frame_Quadruplets)
                            # multi frame
                    else:
                        print("invalid length for Quadruplets",frame_Quadruplets)
    except Exception as e:
        print(e, fileData)
        # print("model response", model_response)
        pass


    return block_triplets

def pre_clean_temporal_triplets(model_response, fileData=None, remove_entity_idx=False):
    ##[red panda-0:lie next to:red panda-1]_[Frame-0:Frame-7];[red panda-0:lie left:red panda-1]_[Frame-0:Frame-7];[red panda-1:lie right:red panda-0]_[Frame-0:Frame-7];[red panda-1:lie next to:red panda-0]_[Frame-0:Frame-7];[red panda-0:lie next to:red panda-2]_[Frame-0:Frame-7];[red panda-0:lie left:red panda-2]_[Frame-0:Frame-7];[red panda-2:sit right:red panda-0]_[Frame-0:Frame-7];[red panda-2:sit next to:red panda-0]_[Frame-0:Frame-7];[red panda-2:taller:red panda-0]_[Frame-0:Frame-7];[red panda-1:lie next to:red panda-2]_[Frame-0:Frame-7];[red panda-1:lie left:red panda-2]_[Frame-0:Frame-7];[red panda-2:sit right:red panda-1]_[Frame-0:Frame-7];[red panda-2:sit next to:red panda-1]_[Frame-0:Frame-7];[red panda-2:taller:red panda-1]_[Frame-0:Frame-7];
    block_triplets = {
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }

    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    try:
        triplets_list = prediction_data.split(";")
        for triplet_data in triplets_list:
            #[red panda-0:lie next to:red panda-1]_[Frame-0:Frame-7]
            triplet_data = triplet_data.replace(":", ",")
            splitTemporal = triplet_data.split("_")
            if len(splitTemporal)!=2:
                print(f"invalid entity length {splitTemporal}")
                continue
            
            triplet_data, temporal = splitTemporal

            triplet_data = triplet_data.replace("[","")
            triplet_data = triplet_data.replace("]","")
            triplet_data = triplet_data.split(",")
            if len(triplet_data)!=3:
                print(f"invalid triplet: {triplet_data}")
                continue

            subj, pred, obj = triplet_data
            triplet = [subj, pred, obj]

            
            if "[" in temporal and "]" in temporal:
                temporal_list = temporal_list.replace("Frame-", "")
                temporal_list = eval(temporal)

                if len(temporal_list)==1:
                    # only one frame 
                    temporal_entity_index = temporal_list[0]
                    if type(temporal_entity_index)!=int:
                        temporal_entity_index = int(temporal_entity_index)
                        block_triplets["triplets"][temporal_entity_index].append(triplet)
                elif len(temporal_list)==2:
                    temporal_entity_start_index,temporal_entity_end_index = temporal_list
                    if type(temporal_entity_start_index)!=int:
                        temporal_entity_start_index = int(temporal_entity_start_index)
                    if type(temporal_entity_end_index)!=int:
                        temporal_entity_end_index = int(temporal_entity_end_index)
                    
                    for i in range(temporal_entity_start_index,temporal_entity_end_index):
                        if i>len(block_triplets["triplets"]):
                            print(f"temporal entity index out of bound: {i}")
                            continue

                        block_triplets["triplets"][i].append(triplet)
                else:
                    print(f"invalid temporal entity: {temporal_list}")

            else:
                print(f"temporal entity is not surrounded by [] : {temporal}")
                continue

    except Exception as e:
        print(f"erro parsing triplet data: {e},{fileData}")

    
    return block_triplets

def pre_clean_prediction_data_onevision_FT_v7(model_response, fileData=None, remove_entity_idx=False, contains_temporal_entity=False):
    block_triplets = {
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    try:
        evaluated_string = prediction_data.replace("#sg_end", "")

        pattern = re.compile(r'\[(.*?)\]_\[frame-(\d+):frame-(\d+)\]')
        matches = pattern.findall(evaluated_string)

        for triplet, start_frame, end_frame in matches:
            triplet_list = triplet.split(':')
            if len(triplet_list)!=3:
                continue
            start_frame, end_frame = int(start_frame), int(end_frame)
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(block_triplets["triplets"]):
                    block_triplets["triplets"][frame_idx].append(triplet_list)
    except Exception as e:
        print(e, fileData)
        # print("model response", model_response)
        pass


    return block_triplets

def pre_clean_prediction_data_onevision_v7(model_response, fileData=None, remove_entity_idx=False, contains_temporal_entity=False):
    block_triplets = {
        "triplets": [[] for i in range(8)],
        "scene": [],
        "description": [],
        "objects": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    if "#sg_start" in prediction_data and "#sg_end" in prediction_data:

        # print(cleanString)
        try:
            cleanString = get_substring_between(s=prediction_data,start_substring="#sg_start",end_substring="#sg_end")
            comment_str = "// This triplet is not necessary as it does not provide additional information.\n"
            if comment_str in cleanString:
                cleanString = cleanString.replace(comment_str, "")
        except Exception as e:
            print("error getting sgblock data")
    else:
        cleanString = prediction_data

    # cleanString = re.sub(r"(frame-\d+)", r"'\1'", cleanString)

    # print(cleanString)
    try:
        # print("evaluating")
        evaluated_string_json = eval(cleanString.replace("#sg_end", ""))
        for key,frame_data in evaluated_string_json.items():
            # print(key)
            if key=="scene":
                block_triplets["scene"].append(frame_data)
            elif key=="description":
                block_triplets["description"].append(frame_data)
            elif key=="objects":
                block_triplets["objects"].append(frame_data)
            elif key=="triplets":
                # strkey = str(key)
                # strkey_f_index = strkey.strip("F")  # F1 ==> 1
                current_frame_triplets = []
                for frame_triplet in frame_data:
                    if contains_temporal_entity:
                        frame_triplet, triplet_time = frame_triplet

                    if len(frame_triplet)==3:
                        if remove_entity_idx:
                            frame_triplet = remove_triplet_indexes(triplet=frame_triplet)

                        if contains_temporal_entity:
                            if len(triplet_time)==1:
                                # single frame
                                triplet_time_fidx = int(triplet_time[0].split("-")[-1])
                                if triplet_time_fidx>7:
                                    continue
                                block_triplets["triplets"][triplet_time_fidx].append(frame_triplet)
                            else:
                                triplet_time_fidx_start = int(triplet_time[0].split("-")[-1])
                                triplet_time_fidx_end = int(triplet_time[0].split("-")[-1])
                                for j in range(triplet_time_fidx_start,triplet_time_fidx_end):
                                    if j>8:
                                        continue
                                    block_triplets["triplets"][j].append(frame_triplet)
                        else:
                            for j in range(8):
                                block_triplets["triplets"][j].append(frame_triplet)
                            # multi frame
                    else:
                        print("invalid length for triplet",frame_triplet)
    except Exception as e:
        print(e, fileData)
        # print("model response", model_response)
        pass


    return block_triplets

def pre_clean_prediction_data_onevision_v6(model_response, fileData=None):
    frame_triplets = {
        "triplets": [],
        "scene": [],
        "st_progression": []
    }
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>").lower()

    if "#sg_start" in prediction_data and "#sg_end" in prediction_data:

        # print(cleanString)
        try:
            cleanString = get_substring_between(s=prediction_data,start_substring="#sg_start",end_substring="#sg_end")
            comment_str = "// This triplet is not necessary as it does not provide additional information.\n"
            if comment_str in cleanString:
                cleanString = cleanString.replace(comment_str, "")
        except Exception as e:
            print("error getting sgblock data")
    
    else:
        cleanString = prediction_data

    # print(cleanString)
    try:
        evaluated_string_json = eval(cleanString)
        for key,frame_data in evaluated_string_json.items():
            if key=="scene":
                frame_triplets["scene"].append(frame_data)
            elif key=="st progression":
                frame_triplets["st_progression"].append(frame_data)
            else:
                # strkey = str(key)
                # strkey_f_index = strkey.strip("F")  # F1 ==> 1
                current_frame_triplets = []
                for frame_triplet in frame_data["triplets"]:
                    if len(frame_triplet)==3:
                        current_frame_triplets.append(frame_triplet)
                    else:
                        print("invalid length for triplet",frame_triplet)
                frame_triplets["triplets"].append(current_frame_triplets)

    except Exception as e:
        print(e, fileData)
        print("model response", model_response)
        pass



    return frame_triplets


def pre_clean_prediction_data_v7_with_time(model_response):
    frame_triplets = []
    frame_triplets_time_windows = []
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>")
    try:
        Triplets = prediction_data.split(";")
        for cnt_idx, triplets_data in enumerate(Triplets):
            if len(triplets_data)<2:
                continue

            # [red panda-0:lie next to:red panda-1]_[Frame-0:Frame-7]
            triplets_data = triplets_data.replace(f":", ",")

            triplets_data = triplets_data.split("_")
            triplet = triplets_data[0]
            triplet_time = triplets_data[1]

            triplet_time = triplet_time.strip("[").strip("]")
            triplet_time = triplet_time.split(",")
            triplet_start = int(triplet_time[0].split("-")[-1])
            triplet_end = int(triplet_time[1].split("-")[-1])

            ftr_temp = triplet.split(",")
            # print(ftr_temp)
            ftr_temp[0] = str(ftr_temp[0]).strip("[").strip("]")
            ftr_temp[1] = str(ftr_temp[1]).strip("[").strip("]")
            ftr_temp[2] = str(ftr_temp[2]).strip("[").strip("]")  

            frame_triplets.append(ftr_temp)
            frame_triplets_time_windows.append([triplet_start,triplet_end])
    
    except Exception as e:
        print("Exception ", e)
        pass
    
    return frame_triplets, frame_triplets_time_windows

def pre_clean_prediction_data_v18_withbb(model_response):
    """
    0. #frameid[person_[0.28, 0.96, 0.72, 0.96]:not looking at:mirror_[0.0, 0.96, 0.08, 0.96]];
    1. [person_[0.28, 0.96, 0.72, 0.96]:not looking at:mirror_[0.0, 0.96, 0.08, 0.96]]
    2. [person_[0.28, 0.96, 0.72, 0.96],  not looking at,  mirror_[0.0, 0.96, 0.08, 0.96]
    3. person, [0.28, 0.96, 0.72, 0.96]  
    3. not looking at, 
    3. mirror,[0.0, 0.96, 0.08, 0.96]
    """

    parsed_data = {
        "triplets": [],
        "triplets_bb": []
    }

    frame_triplets = []
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>")

    bbSubString = get_substring_between(s=prediction_data,start_substring="#bb_start",end_substring="#bbend").strip()
    bbSubString = bbSubString
    
    bbox_Data = {}
    try:
        bbox_Data = eval(bbSubString)
    except Exception as e:
        print(f"error parsing bbox data from str: {bbSubString}")

    parsed_data["triplets_bb"] = bbox_Data

    tripletSubstring = get_substring_between(s=prediction_data,start_substring="#sg_start",end_substring="#sg_end")
    if tripletSubstring is not None:
        tripletSubstring = tripletSubstring.split(f"{SGSpecialTokens.VIDEO_FRAME_ID}")[1:]

        special_tokens = SGSpecialTokens.get_tokens()
        for cnt_idx, ftriplets in enumerate(tripletSubstring):
            if cnt_idx>7:
                break

            # for spetok in special_tokens:
            #     ftriplets = ftriplets.replace(f"{spetok}", "")

            # ftriplets = ftriplets.replace(f":", ",")
            ftriplets = ftriplets.strip().split(";")

            current_frame_triplets = []
            current_frame_triplets_bb = []
            for ftr in ftriplets:
                ftr_temp = ftr.split(":")

                if ftr_temp=="":
                    continue

                if len(ftr_temp)!=3:
                    print(f"invalid triplet length : {ftr_temp}")
                    continue
                
                subject_, predicate_, object_ = ftr_temp # [person_[0.28, 0.96, 0.72, 0.96],  not looking at,  mirror_[0.0, 0.96, 0.08, 0.96]
                # subject_ = subject_.split("_")
                # if len(subject_)!=2:
                #     print(f"invalid subject token: {subject_}")
                #     continue
                
                # subject_name, subject_bb = subject_[0],subject_[1]
                # subject_name = subject_name.strip("[").strip("]")

                # subject_bb = subject_bb.strip("[").strip("]")
                # subject_bb = eval(subject_bb)

                # object_ = object_.split("_")
                # if len(object_)!=2:
                #     print(f"invalid object_ token: {object_}")
                #     continue

                # object_name, object_bb = object_[0],object_[1]
                # object_name = object_name.strip("[").strip("]")
                # object_bb = object_bb.strip("[").strip("]")
                # try:
                #     object_bb = eval(object_bb)
                # except Exception as e:
                #     print(f"error parsing bounding box: {object_bb}")

                current_frame_triplets.append([subject_,predicate_,object_])
                # current_frame_triplets_bb.append([subject_bb,object_bb])
            # frame_triplets.append([current_frame_triplets,current_frame_triplets_bb])

            parsed_data["triplets"].append(current_frame_triplets)
            # parsed_data["triplets_bb"].append(current_frame_triplets_bb)
    
    return parsed_data

def pre_clean_newsvideo_data_v1(model_response):
    """
    Output response format:
    {
        'video_title': 'Flood Rescue Operations in Coastal City',
        'video_description': {
            'foreground_view': 'Emergency responders are seen navigating floodwaters in a small inflatable boat, helping stranded residents evacuate their homes. Some residents are carrying personal belongings while others assist elderly individuals onto the boat.',
            'background_view': 'Flooded streets with partially submerged vehicles and damaged infrastructure. Trees are swaying in the wind, suggesting ongoing harsh weather conditions. Power lines appear downed, adding to the chaotic scene.'
        },
        'video_subtitle': 'Rescue teams are working tirelessly to evacuate residents affected by the severe flooding caused by torrential rains in the area.',
        'video_headlines': 'Breaking News: Massive Flooding Hits Coastal City'
    }
    """
    video_data = {}
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>")
    prediction_data = prediction_data.strip()

    try: 
        parsed_data = eval(prediction_data)
        for key,val in parsed_data.items():
            video_data[key] = val
    except Exception as e:
        print(f"error parsing response : {e}")

    
    return video_data

def pre_clean_prediction_data_v18_paligemma(model_response):
    """
    #frameid [subject, predicate,object];.. #frameid
    """

    # Remove anything after #sg_end
    cleaned_text = re.split(r'#sg_end', model_response)[0]

    # Step 2: Extract the first 8 `#frameid[...]` blocks
    frame_blocks = re.findall(r'#frameid((?:\[[^\]]+\];?)*)', cleaned_text)[:8]

    # Step 3: Extract individual triplets from each block
    frame_triplets = []
    for block in frame_blocks:

        current_frame_triplets = []

        triplets = re.findall(r'\[([^\]]+)\]', block)  # capture inside [...], excluding brackets

        for triplet in triplets:
            sub, pred, obj = triplet.split(":")
            current_frame_triplets.append([sub,pred,obj])

        frame_triplets.append(current_frame_triplets)

    # # Output: list of list of triplet strings
    # for idx, frame in enumerate(triplets_per_frame):
    #     print(f"Frame {idx+1}:")
    #     for triplet in frame:
    #         print(f"  {triplet}")

    return frame_triplets

def pre_clean_prediction_data_v18(model_response):
    """
    #frameid [subject, predicate,object];.. #frameid
    """
    frame_triplets = []
    prediction_data = model_response
    prediction_data = prediction_data.strip("</s>")
    framewiseTriplets = prediction_data.split(f"{SGSpecialTokens.VIDEO_FRAME_ID}")[1:]

    special_tokens = SGSpecialTokens.get_tokens()
    for cnt_idx, ftriplets in enumerate(framewiseTriplets):
        if cnt_idx>7:
            break

        for spetok in special_tokens:
            ftriplets = ftriplets.replace(f"{spetok}", "")

        ftriplets = ftriplets.replace(f":", ",")
        ftriplets = ftriplets.split(";")

        current_frame_triplets = []

        for ftr in ftriplets:
            ftr_temp = ftr.split(",")
            if len(ftr_temp)==3:
                # print("conveting to list",ftr)
                ftr_temp[0] = str(ftr_temp[0]).strip("[").strip("]")
                ftr_temp[1] = str(ftr_temp[1]).strip("[").strip("]")
                ftr_temp[2] = str(ftr_temp[2]).strip("[").strip("]")  
                current_frame_triplets.append(ftr_temp)

        frame_triplets.append(current_frame_triplets)
    
    return frame_triplets

def clean_prediction_data(model_response, val_id, block_id):
    prediction_data = model_response[val_id][f"{block_id}"].copy()
    prediction_data = prediction_data["triplets"].strip("</s>")

    # print("#"*5)
    # print(prediction_data)
    # print("#"*5)

    if validate_model_response(model_response=prediction_data):

        # print("First token ###==>", prediction_data[0])
        # if prediction_data[0].strip()!="{":
        #     if "Frame 0" in prediction_data:
        token_cnt = prediction_data.count("Frame 0")  # if more than 1 reponse repeated
        # print("token count ", token_cnt)
        if token_cnt>1:
            token_idx = prediction_data.index("}")
            prediction_data = prediction_data[0:token_idx+1]
            # print("NEW STRING########> ", prediction_data)

        model_res_len = len(prediction_data)
        end_idx = prediction_data.index("}")
        if model_res_len!=end_idx:
            prediction_data = prediction_data[0:end_idx+1]
        
        if prediction_data[-1]!="}":
            prediction_data_split = prediction_data.split(";")
            last_element = prediction_data_split[-1]
            last_element_spilit = last_element.split(":")
            if len(last_element_spilit)<3:
                del prediction_data_split[-1]
            prediction_data = "".join(prediction_data_split)
            prediction_data += "'}"

        for i in range(8):
            prediction_data = prediction_data.replace(f"Frame {i}", f"{i}")
        
        FrameLevelPredictions = eval(prediction_data)

        return FrameLevelPredictions
    
    return None

def remove_duplicates(frame_level_prediction_for_block):

    # print("#######RM DP#######", frame_level_prediction_for_block)
    all_over_triplates = []
    for frame, data in frame_level_prediction_for_block.items():
        data_split = data.split("[")
        triplates = []
        for i_, data_ in enumerate(data_split):
            data_ = data_.strip("]").strip(";")

            sub_pred_obj = data_.split(":")
            if len(sub_pred_obj)!=3:
                continue
            if sub_pred_obj not in triplates:
                for spoidx, subpreobj in enumerate(sub_pred_obj):
                    sub_pred_obj[spoidx] = subpreobj.strip("]").strip(";")
                triplates.append(sub_pred_obj)

            if sub_pred_obj not in all_over_triplates:
                all_over_triplates.append(sub_pred_obj)
        
        frame_level_prediction_for_block[frame] = triplates
    return frame_level_prediction_for_block, all_over_triplates


prompts_list = {
    
    "summary": ["Describe the video in detail",
                "What is happening in the video?",
                "What is the central narrative or story in the video?",
                "What is the purpose or goal of the video?",
                "What are the key takeaways or lessons from the video?"
                ],

    "identify_subject_objects": [
                        "List the objects present in the video",
                        "What objects, items, or elements appear prominently?", 
                        "Identify any significant objects in the video.",
                        "What objects are visible in the video?",
                        "List the main objects featured in the video.",
                        "what are the main objects featured in the video?"
                        ],
    "identify_predicates": [
                            "List the actions, movements or placements of the objects in the scene.",
                            "Describe any interactions between people or objects in the video.",
                            "Describe any significant gestures or interactions between objects in the scene",
                            "How subjects and objects relates to each other in the video?",
                            "How do the objects interact with their environment in the video?",
                            "Describe any notable physical interactions between objects in the video.",
                            "Describe any interactions that highlight the relationships between objects.",
                            "What actions or events take place in the video?",
                          ],
    "SGG": [
       "Generate frame-by-frame scene graph for the provided video",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id:Object-id:Predicate]",
       "Generate scene graph for the provided video",
       "Provide scene graph for the provided video",
       "Identify subjects, predicates and objects frame-by-frame in the provided video"
    ],

    "SGG_image": [
       "Generate scene graph for the provided image",
       "Provide Scene graph triplets in the form of [Subject-id:Predicate:Object-id] for the provided image",
       "Generate scene graph for the provided image",
       "Provide scene graph for the provided image",
       "Identify subjects, predicates and objects in the provided image"
    ],

    "SGG_with_bb": [
       "Generate frame-by-frame scene graph for the provided video along with bounding box of each objects",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id-[min_x,min_y,max_x,max_y]:Predicate:Object-id-[min_x,min_y,max_x,max_y]]",
       "Generate scene graph for the provided video along with bounding box of each objects",
       "Provide scene graph for the provided video with bounding box location of each objects",
       "Identify Subjects, Predicates and Objects frame-by-frame in the provided video, also provide bounding box location of each subject and object"
    ],

    "sg_localization": [
      "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
      #"Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ],

    "sg_localization_image": [
      "Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ],

    "predict_predicate": [
      "What is the relationship between [{sub}:{obj}] in the video. Use only the provided lists for predicates. Predicates: {predicates}" # {} to be replaced by actual value
    ],

    "AG_Prompt_Temporal" : [
      """
      You are given a list of predefined objects={objects_list} and three different types of predicates list. 
      1.Attention={attention_relations},
      2.Spatial={spatial_relations} and 
      3.Contacting={contacting_relations}

      Attention relationship indicates whether the person is visually focused on the object in the scene.
      Spatial relationship describes the object's position relative to the person within the scene.
      Contacting relationship specifies the physical interaction or contact between the person and the object.

      Your task is to identify relationships between person and predefined objects visible in the provided video in the format [Subject:Predicate:Object]_[Frame-start:Frame-end].
      """
    ],

    "AG_Prompt" : [
      """
      You are given a list of predefined objects={objects_list} and three different types of predicates list. 
      1.Attention={attention_relations},
      2.Spatial={spatial_relations} and 
      3.Contacting={contacting_relations}

      Attention relationship indicates whether the person is visually focused on the object in the scene.
      Spatial relationship describes the object's position relative to the person within the scene.
      Contacting relationship specifies the physical interaction or contact between the person and the object.

      Your task is to identify relationships between person and predefined objects visible in the provided video.
      """
    ],

    "AG_Prompt_sg_tagging" : [
      """
      You are given three different types of predefined predicates list. 
      1.Attention={attention_relations},
      2.Spatial={spatial_relations} and 
      3.Contacting={contacting_relations}

      Attention relationship indicates whether the person is visually focused on the object in the scene.
      Spatial relationship describes the object's position relative to the p+
      erson within the scene.
      Contacting relationship specifies the physical interaction or contact between the person and the object.

      When given object pairs with bounding box location [xmin,ymin,xmax,ymax] in the video, the task is to identify relationships between them in the provided video from the predefined list.
      For each pair, all three relationships(i.e. attention,spatial and contacting) should be given.
      """
    ],

    "AG_Prompt_sg_with_bb" : [
      """
      You are given a list of predefined objects={objects_list} and three different types of predicates list. 
      1.Attention={attention_relations},
      2.Spatial={spatial_relations} and 
      3.Contacting={contacting_relations}

      Attention relationship indicates whether the person is visually focused on the object in the scene.
      Spatial relationship describes the object's position relative to the person within the scene.
      Contacting relationship specifies the physical interaction or contact between the person and the object.

      Your task is to identify relationships between person and objects visible in the provided video frame-by-frame from the predefined list along with their bounding box locations in format [xmin,ymin,xmax,ymax].
      """
    ],

    "AG_Prompt_sg_with_center" : [
      """
      You are given a list of predefined objects={objects_list} and three different types of predicates list. 
      1.Attention={attention_relations},
      2.Spatial={spatial_relations} and 
      3.Contacting={contacting_relations}

      Attention relationship indicates whether the person is visually focused on the object in the scene.
      Spatial relationship describes the object's position relative to the person within the scene.
      Contacting relationship specifies the physical interaction or contact between the person and the object.

      Your task is to identify relationships between person and objects visible in the provided video frame-by-frame from the predefined list along with their object center locations in format [x,y].
      """
    ],

    

    "AG_Prompt_bbonly" : [
      """
      Generate detailed frame-by-frame bounding box coordinates in the format [xmin, ymin, xmax, ymax] for each visible object in the provided video.
      """
    ],

    # """
    # Generate frame-by-frame scene graph for the provided video
    #    Use the following list to select the object {}, 
    #    the following list to select the subject {}, 
    #    and the follwing list to select the predicate {}.
    # """

    # "triplet_prompt": [
    #   """You are given a list of predefined subjects, objects, and predicates. Your task is to predict scene graph triplets in the format [Subject:Object:Predicate] based on the given scene description in the video. Use only the provided lists for objects, and predicates.
    #   Subjects: {subjects}
    #   Objects: {objects}
    #   Predicates: {predicates}
    #   """
    # ]
    "triplet_prompt": [
      """You are given a list of predefined subjects, objects, and predicates. Your task is to predict scene graph triplets in the format [Subject-id:Object-id:Predicate] based on the given scene in the video. Use only the provided lists for subjects, objects, and predicates.
      Subjects: {subjects}
      Objects: {objects}
      Predicates: {predicates}
      """
    ],
    
    # "v10_sam": [Task_description_v10_sam],

    "quad_vrd_prompt": [
      """You are given a list of predefined subjects, objects, action predicates and spatial predicates. Your task is to predict scene graph quadruplets in the format [Subject-id:Object-id:action predicate:spatial predicate] based on the given scene in the video. Use only the provided lists for subject, objects,action predicates and spatial predicates. \n\
      Subjects: {subjects} \n\
      Action Predicates: {action_predicates} \n\
      Spatial Predicates: {spatial_predicates} \n\
      Objects: {objects} \n\
      """
    ]


}


def getConvBlock(value,conv_type="human", media_type="<image>", add_media_token=False):
   assert conv_type=="human" or conv_type=="gpt"
   assert media_type=="<image>" or media_type=="<video>"
   conv = {"from": conv_type, "value": f"{value}"}
   if add_media_token:
      conv["value"] = f"{media_type}\n{value}"
   else:
      conv["value"] = f"{value}" 

   return conv

def getPromptTemplate(media_path, media_type="image"):
  assert media_type=="image" or media_type=="video"
  Prompt = {
          "id": "TobeUpdated",
          f"{media_type}": f"{media_path}",
          "conversations": [],
          "frame_indices": [],  # selected indices will be passed to model for train and test
          "total_frames": "",
  }
  return Prompt


def getRandomPrompt(key="summary", static=False):
    if static:
       return prompts_list[key][0]
    return random.choice(prompts_list[key])

def getFramesForObject(vid_data, Subject_id):
    vid_rels = vid_data["relations"]
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        # rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if Subject_id==sub or Subject_id==obj:
            return frames_
    return "None"

def normlize_boundingbox(bbox, height, width,decimal=3, is_width_hight_bb=False, norm_bb=True):
    x1,y1,x2,y2 = bbox

    if is_width_hight_bb:
        # convert x1,y1,w,h to x1y1x2y2
        x1,y1,w,h = bbox
        x2 = x1 + w
        y2 = y1 + h

    if norm_bb:
        x1 = round((x1/width),decimal)
        y1 = round((y1/height),decimal)
        x2 = round((x2/width),decimal)
        y2 = round((y2/height),decimal)
    
    return [x1,y1,x2,y2]

def getbbcenter(bb,height,width,norm_center=False,decimal_points=2, is_width_hight_bb=False):
   if len(bb)<4:
    return []
   
   if is_width_hight_bb:
       # conver xywh to x1y1x2y2
       bb = normlize_boundingbox(bbox=bb,height=height,
                    width=width,decimal=decimal_points,
                    is_width_hight_bb=is_width_hight_bb,
                    norm_bb=False)
   x1,y1,x2,y2 = bb
   bb_w = (x2 - x1)/2
   bb_h = (y2 - y1)/2
   xcenter = x1 + bb_w
   ycenter = y1 + bb_h
   if norm_center:
        xcenter = round((xcenter/width),decimal_points)
        ycenter = round((ycenter/height),decimal_points)

   return [round(xcenter,decimal_points), round(ycenter,decimal_points)]

def getListofCategoryString(data_root, vid_objects, vid_data, addObjectId=False, addFrames=False, addBB=False , uniform_sampling_idx=8):
    
    AnswerString = ""
    frame_indices = []
    total_frames = vid_data["meta"]["num_frames"]
    """V11 implementation
    [X] Select frames which covers all objects, avoid repetations
    """

    frames_where_obj_is_present = {}
    min_frame_idx, max_frame_idx, frames_for_obj = get_frame_range_for_annotations(vid_objects, vid_data)

    for frame_idx  in range(min_frame_idx, max_frame_idx+1):
      if frame_idx>total_frames:
         continue

      if frame_idx not in frames_where_obj_is_present.keys():
        frames_where_obj_is_present[frame_idx] ={
          "objects_present": [],
          "object_bb": [],
          "object_cnt": 0
        }

      for vid_obj_idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        object_id = vobj["object_id"]
        
        try:
          sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_idx, object_id=object_id)
        except FileNotFoundError:
          pass

        if sum(sub_bb)>0:
          frames_where_obj_is_present[frame_idx]["objects_present"].append(vobj)
          frames_where_obj_is_present[frame_idx]["object_bb"].append(sub_bb)
          frames_where_obj_is_present[frame_idx]["object_cnt"] +=1

    # Take frames with more objects count first
    frames_with_obj_cnt = [(frames_where_obj_is_present[f_idx]["object_cnt"], f_idx) for f_idx in frames_where_obj_is_present]
    frames_with_obj_cnt = sorted(frames_with_obj_cnt,reverse=True)

    objects_added = []

    """
    Frame wise
    AnswerString = {
      0: "floor-1, wall-1, pillow-4",
      1: "floor-1, wall-1, shelf-4"
      .
      .
      7: "obj1,obj2"
    }
    """

    AnswerString += "{"

    for f_obj_idx, f_obj_cnt in enumerate(frames_with_obj_cnt):
      cnt_,f_idx = f_obj_cnt
      data = frames_where_obj_is_present[f_idx]

      AnswerString += f"{f_obj_idx}:"
      AnswerString +="'"  # start the list of objects string by "'"

      objects_present = data["objects_present"]
      objects_bb = data["object_bb"]

      frame_indices.append(f_idx) # use frame indices where object annotations are present

      for oidx, obj in enumerate(objects_present):
        category = obj["category"]
        object_id = obj["object_id"]

        # object_name_id = f"{category}-{object_id}"
        # if object_name_id not in objects_added:
        #   """This ensures unique objects in the list"""
        #   objects_added.append(object_name_id)

        AnswerString += f"{category}"

        if addObjectId:
          AnswerString += f"-{object_id}"

        if addBB:
          AnswerString += f"-{objects_bb[oidx]}"
        if addFrames:
          AnswerString += f"_[{f_idx}]"

        if oidx!=len(objects_present)-1:
          AnswerString +=","
        else:
          AnswerString +="'"  # finish the list of objects string by "'"
        
        if f_obj_idx>6:
           # TODO: some objects which appears in low count, will not be taken due to object density
           # In order to resolve this issue, need to accomodate all frames in 8 frames
           break
        
        if f_obj_idx!=len(frames_with_obj_cnt)-1:
           AnswerString += f"," # end of current key in dict


    AnswerString += "}"

    return AnswerString, frame_indices


def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True, dataset="vidor"):
    mask_name = os.path.join(data_root, dataset, 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        if normlize_bb:
           x_min = round(x_min/mask_w,3)
           x_max = round(x_max/mask_w,3)
           y_min = round(y_min/mask_h,3)
           y_max = round(y_max/mask_h,3)

        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]



def get_frame_range_for_annotations(vid_objects, vid_data,):
  min_frame_idx, max_frame_idx = -1, 0
  frames_for_obj = {}
  for vid_obj_idx, vobj in enumerate(vid_objects):
    category = vobj["category"]
    object_id = vobj["object_id"]
    frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
    if frames_=="None":
        continue
    
    for frame_range in frames_:
      frame_start, frame_end = frame_range

      if f"{category}{object_id}" not in frames_for_obj:
        frames_for_obj[f"{category}{object_id}"] = {
          "frames": []
        }

      frames_for_obj[f"{category}{object_id}"]["frames"].append(frame_range)

      if min_frame_idx ==-1:
        min_frame_idx = frame_start
      if frame_start<=min_frame_idx:
        min_frame_idx = frame_start
      if frame_end>=max_frame_idx:
        max_frame_idx = frame_end

  return min_frame_idx, max_frame_idx, frames_for_obj

def get_frame_samples(total_frames,every_nth=4,frame_window_size=32,shift_step=3, total_shifts=100):
	frames_selected = []
	assert shift_step!=every_nth
	for i in range(0,total_shifts,shift_step):
		frames =[]
		for j in range(i, i+frame_window_size,every_nth):
			if j>total_frames:
				break
			frames.append(j)
		if len(frames)>=int(frame_window_size/every_nth):
			frames.sort()
			frames_selected.append(frames)
	return frames_selected

def get_default_indices(video_path, frames_to_add=8):
    total_frames = getVideoFrameCount(video_path=video_path)
    if total_frames is not None:
        return np.linspace(0, total_frames-1, frames_to_add, dtype=int)
    else:
        return np.array([i for i in range(frames_to_add)])
    
def getVideoFrameCount(video_path):
    cv2_vr = cv2.VideoCapture(video_path)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2_vr.release()
    return total_frames



 
### Action Gnome helpers

AG_relations = {
"attention": ['unsure', 'not looking at', 'looking at'],
"spatial": ['in front of', 'beneath', 'behind', 'on the side of', 'in', 'above'],
"contacting": ['not contacting', 'sitting on', 'leaning on', 'other relationship', 'holding', 'touching', 'twisting', 'eating', 
                'drinking from', 'standing on','wearing','lying on','carrying','wiping','covered by','writing on','have it on the back']
}

AG_relationsCombined = AG_relations["attention"]+AG_relations["spatial"]+AG_relations["contacting"]

AG_Objects = ['table','chair','bag','doorway','medicine','cup/glass/bottle','food','floor','broom','shoe','clothes','door','doorknob','groceries',
'closet/cabinet','laptop','bed','shelf','blanket','sandwich','refrigerator','vacuum','box','light','phone/camera','dish','paper/notebook',
'mirror','book','sofa/couch','television','pillow','towel','picture','window']



def load_AG_annotations(annotation_dir):
    """
    Taken from https://github.com/JingweiJ/ActionGenome/blob/master/tools/load_annotations.py#L5
    """
    with open(os.path.join(annotation_dir, 'object_bbox_and_relationship.pkl'), 'rb') as f:
        object_anno = pickle.load(f)

    with open(os.path.join(annotation_dir, 'person_bbox.pkl'), 'rb') as f:
        person_anno = pickle.load(f)

    frame_list = []
    with open(os.path.join(annotation_dir, 'frame_list.txt'), 'r') as f:
        for frame in f:
            frame_list.append(frame.rstrip('\n'))

    return object_anno, person_anno, frame_list

def addIf(triplet, Objects, relations):
    """
    Add triplet if entities belongs to predefined selected list.
    """
    subj, pred, obj = triplet
    if subj in Objects and obj in Objects and pred in relations:
        return True
    return False

def get_AG_annotations_framewise(AG_ANNOTATIONS_DIR,subset="train", EvalKeepData=None):
    """
    Custom helper function for AG annotation

    returns list of annotatation in list type [video_id, AG_annotations]
    can be looped like: \n
    
        for video_id, video_annotations in get_AG_annotation(args): \n
            for frame_id, triplets in video_annotations: \n

    """

    if EvalKeepData is not None:
        AllRels = EvalKeepData["selected_relations"]["spatial"] + EvalKeepData["selected_relations"]["contacting"] + EvalKeepData["selected_relations"]["attention"]
        AllObjects = EvalKeepData["selected_objects"]

    object_anno, person_anno, frame_list = load_AG_annotations(annotation_dir=AG_ANNOTATIONS_DIR)

    assert set(object_anno.keys()) == set(person_anno.keys())
    assert len(object_anno) == len(frame_list)

    set_count = {"train": 0,"test": 0}
    video_ids_by_set = { "train": [], "test": [] }

    dataset_meta = {
        "objects": [],
        "relationships": {
            "attention": [],
            "spatial": [],
            "contacting": []
        }
    }

    # video2frames = {}
    video2frames_full = {}
    for path in frame_list:
        video, frame = path.split('/')
        if video not in video2frames_full:
            video2frames_full[video] =[]
        video2frames_full[video].append(path)
    
    # person data and object data by video frameid
    video_frame_data = {}
    # For each video, dump frames.
    for v in tqdm(video2frames_full):
        # curr_frame_dir = os.path.join(frame_dir, v)
        if v not in video_frame_data.keys():
            video_frame_data[v] = []
        framesToKeep = video2frames_full[v]
        for frameid in framesToKeep:
            objects_annot = object_anno[frameid]
            person_data = person_anno[frameid]
            frameid = frameid.split("/")[-1]
            video_frame_data[v].append([frameid,person_data,objects_annot])



    # get dataset metadata, train/test split
    for videoid, video_data in video_frame_data.items():
        for video_annotation in video_data:
            frameid,person_data,objects_annot = video_annotation

            for objAnnot in objects_annot:
                obj_class = objAnnot["class"]
                obj_bb =  objAnnot["bbox"]    # NOT USED

                if obj_class not in dataset_meta["objects"]:
                    dataset_meta["objects"].append(obj_class)

                attention_relationship = objAnnot["attention_relationship"]
                spatial_relationship = objAnnot["spatial_relationship"]
                contacting_relationship = objAnnot["contacting_relationship"]

                if attention_relationship!=None:
                    for attn_rel in attention_relationship:
                        if attn_rel not in dataset_meta["relationships"]["attention"]:
                            dataset_meta["relationships"]["attention"].append(attn_rel)

                if spatial_relationship!=None:
                    for spa_rel in spatial_relationship:
                        if spa_rel not in dataset_meta["relationships"]["spatial"]:
                            dataset_meta["relationships"]["spatial"].append(spa_rel)

                if contacting_relationship!=None:
                    for cont_rel in contacting_relationship:
                        if cont_rel not in dataset_meta["relationships"]["contacting"]:
                            dataset_meta["relationships"]["contacting"].append(cont_rel)

                metadata = objAnnot["metadata"]
                data_split = metadata["set"]
                if data_split=="train":
                    set_count["train"] +=1
                    if videoid not in video_ids_by_set["train"]:
                        video_ids_by_set["train"].append(videoid)
                else:
                    set_count["test"] +=1
                    if videoid not in video_ids_by_set["test"]:
                        video_ids_by_set["test"].append(videoid)

    assert len(video_ids_by_set["train"])==len(list(set(video_ids_by_set["train"])))
    assert len(video_ids_by_set["test"])==len(list(set(video_ids_by_set["test"])))

    # prepare annotations videoid->frames->triplets
    overall_annotations = []
    for video_id in tqdm(video_ids_by_set[subset]):
        video_data = video_frame_data[video_id]

        frame_block_triplets = []
        for video_annotation in video_data:

            frameid, person_data,objects_annot = video_annotation

            frame_triplets = []
            frame_triplets_bb = []
            for objAnnot in objects_annot:
                obj_class = objAnnot["class"]
                metadata = objAnnot["metadata"]
                if objAnnot["visible"]:

                    obj_bb =  list(objAnnot["bbox"])
                    # print("converted list", obj_bb)

                    frame_h,frame_w = person_data['bbox_size']
                    unnorm_person_bb = person_data["bbox"]
                    if len(unnorm_person_bb)>0:
                        unnorm_person_bb = list(unnorm_person_bb[0])
                    else:
                        unnorm_person_bb = []
                    if len(unnorm_person_bb)==0 or obj_bb==None:
                        continue
                    
                    x, y, w, h = obj_bb
                    obj_bb = [x, y, x+w, y+h]

                    attention_relationship = objAnnot["attention_relationship"]
                    spatial_relationship = objAnnot["spatial_relationship"]
                    contacting_relationship = objAnnot["contacting_relationship"]

                    for attn_rel in attention_relationship:
                        if "_" in attn_rel: attn_rel = attn_rel.replace("_", " ")
                        trip = ["person", attn_rel, obj_class]
                        if EvalKeepData is not None:
                            if addIf(triplet=trip, Objects=AllObjects,relations=AllRels):
                                frame_triplets.append(trip)
                                frame_triplets_bb.append([unnorm_person_bb,obj_bb,(frame_h,frame_w)])
                        else:
                            frame_triplets.append(trip)
                            frame_triplets_bb.append([unnorm_person_bb,obj_bb,(frame_h,frame_w)])

                    for spa_rel in spatial_relationship:
                        if "_" in spa_rel: spa_rel = spa_rel.replace("_", " ")
                        trip = [obj_class, spa_rel, "person"]
                        if EvalKeepData is not None:
                            if addIf(triplet=trip, Objects=AllObjects,relations=AllRels):
                                frame_triplets.append(trip)
                                frame_triplets_bb.append([obj_bb,unnorm_person_bb,(frame_h,frame_w)])
                        else:
                            frame_triplets.append(trip)
                            frame_triplets_bb.append([obj_bb,unnorm_person_bb,(frame_h,frame_w)])
                        

                    for cont_rel in contacting_relationship:
                        if "_" in cont_rel: cont_rel = cont_rel.replace("_", " ")
                        trip = ["person", cont_rel, obj_class]

                        if EvalKeepData is not None:
                            if addIf(triplet=trip, Objects=AllObjects,relations=AllRels):
                                frame_triplets.append(trip)
                                frame_triplets_bb.append([unnorm_person_bb,obj_bb,(frame_h,frame_w)])
                        else:
                            frame_triplets.append(trip)
                            frame_triplets_bb.append([unnorm_person_bb,obj_bb,(frame_h,frame_w)])
            
            
            if len(frame_triplets)>0:
                frame_block_triplets.append([frameid,frame_triplets,frame_triplets_bb])

        overall_annotations.append([video_id, frame_block_triplets])

    return overall_annotations,dataset_meta,video_frame_data

def print_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_mem = trainable_params * 2 / (1024 ** 2)  # 2 bytes for fp16
    total_mem = all_params * 2 / (1024 ** 2)

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable memory (approx): {trainable_mem:.2f} MB")
    print(f"Total model memory (approx): {total_mem:.2f} MB")

# ## add new tokens
# tokenizer = processor.tokenizer

# # Get original sizes
# original_vocab_size = tokenizer.vocab_size
# original_total_size = len(tokenizer)

# print(f"Original vocab size (pretrained): {original_vocab_size}")
# print(f"Original total tokenizer size (includes added tokens): {original_total_size}")

# added_tokens_count = tokenizer.add_tokens(["#frameid", "#sgend"], special_tokens=True)

# # Get updated sizes
# new_total_size = len(tokenizer)

# print(f"Number of new tokens added: {added_tokens_count}")
# print(f"New total tokenizer size: {new_total_size}")

# # Attach updated tokenizer to processor if needed
# processor.tokenizer = tokenizer

# model.resize_token_embeddings(len(processor.tokenizer))
# print(f"Model's token embeddings resized to: {len(processor.tokenizer)}")