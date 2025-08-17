import os
import json
import random
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18_paligemma
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import AG_Objects,AG_relations, AG_OBJECTS_ALTERATIVES

import argparse
import os
import torch

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
# from utils.dataset_utils import LazySupervisedDataset, collate_fn_video
from utils.config import Configuration
import os
import time

from tqdm import tqdm
from PIL import Image

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
from utils.video_processing import process_video_with_decord
from utils.dataset_utils import collate_fn_video_gemma3_test

def init_main(args):

    global model, processor
    global config, stop_token_ids

    config = Configuration()
    config.PROJECT_NAME =  "gemma3-it-video_dsgg"
    config.attn_implementation = ["sdpa", "eager", "flash_attention_2"][2] 
    config.MAX_FRAMES_TO_TRAIN = 4
    config.MODEL_ID = "google/gemma-3-4b-it" # "google/gemma-3n-e4b-it"
    # get the processor
    # config.MODEL_ID= "google/paligemma-3b-ft-nlvr2-448"
    print(f"[INFO] loading {config.MODEL_ID} processor from hub...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # or bfloat16 if supported
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=None
    )

    # load the pre trained model
    print(f"[INFO] loading {config.MODEL_ID} model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.MODEL_DTYPE,
        device_map=config.DEVICE,
        # revision=config.MODEL_REVISION,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation
    )
    # model.config.attn_implementation = "flash_attention_2"

    # Load adapter
    print(f"loading lora model from: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)

    # Send to device
    model.to("cuda").eval()


def get_model_output(batch_data):

    inputs = collate_fn_video_gemma3_test(examples=batch_data, processor=processor, device=config.DEVICE)
    input_len = inputs["input_ids"].shape[-1] ## padded input length for the batch

    batch_outputs = []
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=4000,
                                    do_sample=False,  
                                    eos_token_id=stop_token_ids,
                                    cache_implementation="dynamic")

        batch_size = generation.shape[0]

        for i in range(batch_size):
            decoded = processor.decode(generation[i][input_len:], skip_special_tokens=True)
            batch_outputs.append({
                "triplets": decoded.strip()
            })

    return batch_outputs


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=False)
    parser.add_argument("--output_dir", help="Directory to save the model results.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results", required=False)
    parser.add_argument("--model-path", type=str, default="/home/ja882177/dso/gits/paligemma-video/checkpoints/gemma-3-4b-it-lora_r16_a16_lr0.0002_justlora_imgmask")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--load_4bit",  type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--start_index", type=int, default=0,required=False)
    parser.add_argument("--samples_to_process", type=int, default=100,required=False)
    # parser.add_argument("--prev_eval_data", type=str, default="", required=False)
    return parser.parse_args()

def getAGAnnotationsBatch(AG_Annotations, frames_per_block=8):    
    all_video_samples = []

    for val_id_idx,AG_Annotation in enumerate(AG_Annotations):

        if val_id_idx<args.start_index:
            ## To Continue unfinished job
            pbar.n = val_id_idx
            pbar.last_print_n = pbar.n
            pbar.refresh()
            continue

        video_id, video_annotations = AG_Annotation
        video_path = os.path.join(VIDEO_ROOT_PATH,video_id)
        if not os.path.exists(video_path):
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()
        
        
        block_wise_GT_data = []
        frame_indices = []
        added_GT_triplets_frames = []
        added_GT_triplets_bb = []

        for frame_id, frame_triplets,frame_triplets_bb in video_annotations:
            frame_int_idx = int(frame_id.split(".")[0])
            # print(frame_id, frame_int_idx)
            added_GT_triplets_frames.append(frame_triplets)
            added_GT_triplets_bb.append(list(frame_triplets_bb))
            frame_indices.append(frame_int_idx)

            if len(frame_indices)>=frames_per_block:
                block_wise_GT_data.append({
                    "frame_idxes": frame_indices,
                    "triplets": added_GT_triplets_frames,
                    "triplets_bb": added_GT_triplets_bb
                })

                frame_indices = []
                added_GT_triplets_frames = []
                added_GT_triplets_bb = []

        
        # print(f"remaining frames: {len(frame_indices)}")
        if len(frame_indices)>0:
            ## add remaning frames
            block_wise_GT_data.append({
                "frame_idxes": frame_indices,
                "triplets": added_GT_triplets_frames,
                "triplets_bb": added_GT_triplets_bb
            })


        for frame_block_index, block_data in enumerate(block_wise_GT_data):

            all_video_samples.append({
                "prompt": AG_Prompt,
                "video_path": video_path,
                "block_data": block_data,
                "frame_block_index": frame_block_index,
                "video_id": video_id
            })

    return all_video_samples

if __name__=="__main__":

    args = parse_args()
    print(args)

    BATCH_SIZE  = 16
    BATCH_CNT = 0
    TOTAL_SAMPLES_TO_PROCESS = args.samples_to_process
    BATCH_PROCESS_TIME = ""

    # model_save_path = "./checkpoints/paligemma2-lora_r256_a512_lr1e-5"

    init_main(args)

    # exit()

    sg_eval_counts = {
        "total_obj_cnt" : 0,
        "total_pred_cnt" : 0,
        "total_sub_cnt" : 0,
        "correct_obj_pred_cnt" : 0,
        "correct_subj_pred_cnt" : 0,
        "correct_predicate_cnt" : 0,
        "gt_triplets_cnt": 0,
        "pred_triplets_cnt": 0,
        "correct_pred_triplets_cnt": 0,
        "total_predicted_triplets": 0
    }

    GtData = {
        "subjects": [],
        "objects": [],
        "predicates": []
    }

    PredData = {
        "subjects": [],
        "predicates": [],
        "objects": []
    }

    dataset_name = "ActionGnome"
    dataset_name_to_save = dataset_name
    version = args.output_dir

    splits = ["test"]
    VIDEO_ROOT_PATH = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    AG_ANNOTATIONS_DIR = "/groups/sernam/datasets/ActionGenome/ActionGenome/annotations"
    CHUNK_N = 1000 # Q&A will be chunked into CHUNK_N parts
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])

    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    AG_Prompt = getRandomPrompt(key='AG_Prompt', static=True)
    AG_Prompt = AG_Prompt.replace("{objects_list}",  ",".join(get_shuffled_list(AG_Objects)) )
    AG_Prompt = AG_Prompt.replace("{spatial_relations}", ",".join(get_shuffled_list(AG_relations["spatial"])))
    AG_Prompt = AG_Prompt.replace("{contacting_relations}", ",".join(get_shuffled_list(AG_relations["contacting"])))
    AG_Prompt = AG_Prompt.replace("{attention_relations}", ",".join(get_shuffled_list(AG_relations["attention"])))

    AG_relationsCombined = AG_relations["attention"]+AG_relations["spatial"]+AG_relations["contacting"]
    
    llava_response_json = {}
    llava_raw_response_json = {}
    frame_block = 0

    overall_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }
    

    block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
    last_processed_time = None
    
    ALL_VIDEO_SAMPLES = getAGAnnotationsBatch(AG_Annotations=AG_Annotations, frames_per_block=4)
    TOTAL_BATCH = len(ALL_VIDEO_SAMPLES)//BATCH_SIZE
    TOTAL_BATCH_TO_PROCESS = args.samples_to_process//BATCH_SIZE

    random.seed(123)
    random.shuffle(ALL_VIDEO_SAMPLES)

    pbar = tqdm(total=TOTAL_BATCH_TO_PROCESS)
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()

    batch_data = []
    infer_times = []
    batch_process_time = []
    
    for sample in ALL_VIDEO_SAMPLES:

        # create batch of samples with size BATCH_SIZE
        if len(batch_data)<BATCH_SIZE:
            batch_data.append(sample)
            continue

        # check batch processing time
        if last_processed_time is None:
            last_processed_time = time.perf_counter()
        else:
            nowTime = time.perf_counter()
            BATCH_PROCESS_TIME = nowTime-last_processed_time
            batch_process_time.append(BATCH_PROCESS_TIME)
            last_processed_time = nowTime

        # Get batch inference
        start = time.perf_counter()
        outputs_unclean = get_model_output(batch_data)
        end = time.perf_counter()
        INFERENCE_TIME = round((end-start),4)
        # print(f"inference done in : {end-start:.4f}s")
        infer_times.append(INFERENCE_TIME)

        for bidx, batch in enumerate(batch_data):
            
            block_data= batch["block_data"]
            video_id = batch["video_id"]
            frame_block_index = batch["frame_block_index"]

            Block_frame_ids = block_data["frame_idxes"]
            Block_GT_Triplets = block_data["triplets"]

            ## parse model outputs
            try:
                outputs = pre_clean_prediction_data_v18_paligemma(outputs_unclean[bidx]["triplets"])
                # print(outputs)
            except IndexError as ie:
                print(f"error parsing outputs: {ie}")
                continue
            except Exception as e:
                print(f"error parsing outputs: {e}")
                continue


            if video_id not in llava_response_json:
                llava_response_json[video_id] = {}
                llava_raw_response_json[video_id] = {}

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}


            llava_response_json[video_id][frame_block_index] = {
                "triplets": outputs,
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets
            }

            llava_raw_response_json[video_id][frame_block_index] = {
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                "raw": outputs_unclean[bidx]["triplets"],
                "Prompt": AG_Prompt,
                "cleaned_output": outputs
            }

            try:
                Block_GT_triplets_woids = remove_ids(Block_GT_Triplets,version="v2_1",remove_indexes=True)
                Block_predicated_triplets_woids = remove_ids(outputs,version="v2_1",remove_indexes=True)
            except Exception as e:
                print(f"error remove_ids() {e}")
                continue

            frame_metric = {
                "subject": {"precision": [], "recall": []},
                "object": {"precision": [], "recall": []},
                "predicate": {"precision": [], "recall": []},
                "triplet": {"precision": [], "recall": []}
            }

            for fidx, GT_tripdata in enumerate(Block_GT_triplets_woids):

                results = None
                frame_GT_triplets = GT_tripdata
                frame_pred_triplets = []

                try:frame_pred_triplets = Block_predicated_triplets_woids[fidx]
                except Exception as e:
                    print(f"error getting pred triplets: {e} for idx: {fidx}")
                    continue


                gt_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                pred_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},

                gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
                pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

                for fgt in frame_GT_triplets:
                    fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
                    gt_all["triplet"].append({"triplet": fgt, "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

                for fpred in frame_pred_triplets:
                    fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes

                    if fpred_s not in AG_Objects:
                        if fpred_s not in PredData["subjects"]:
                            PredData["subjects"].append(fpred_s)
                    if fpred_p not in AG_relationsCombined:
                        if fpred_p not in PredData["predicates"]:
                            PredData["predicates"].append(fpred_p)
                    if fpred_o not in AG_Objects:
                        if fpred_o not in PredData["objects"]:
                            PredData["objects"].append(fpred_o)

                    fpred_s = AG_OBJECTS_ALTERATIVES.get(fpred_s,fpred_s)
                    fpred_o = AG_OBJECTS_ALTERATIVES.get(fpred_o,fpred_o)
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                for fm_key, fmdata in frame_metric.items():
                    """
                    Eval score for each frame
                    """
                    if len(gt_all[fm_key])>0 and len(pred_all[fm_key])>0:
                        prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
                        frame_metric[fm_key]["precision"].append(prec)
                        frame_metric[fm_key]["recall"].append(rec)

                
                if len(GT_tripdata)>0 and len(frame_pred_triplets)>0:
                    try:
                        results = calculate_accuracy_varying_lengths(gt_triplets=GT_tripdata,pred_triplets=frame_pred_triplets, remove_duplicates=False)
                    except Exception as e:
                        print(f"error calculating score for vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx: {Block_frame_ids[fidx]}")

                    if results is not None:
                        sg_eval_counts["correct_pred_triplets_cnt"] +=  results["correct_triplet_cnt"]
                        sg_eval_counts["correct_obj_pred_cnt"] += results["correct_object_cnt"]
                        sg_eval_counts["correct_subj_pred_cnt"] +=  results["correct_subject_cnt"]
                        sg_eval_counts["correct_predicate_cnt"] +=  results["correct_predicate_cnt"]
                        sg_eval_counts["gt_triplets_cnt"] +=  results["total_triplets"]
                        sg_eval_counts["total_predicted_triplets"] += results["total_predicted_triplets"]
                        sg_eval_counts["total_obj_cnt"] +=  results["total_objects"]
                        sg_eval_counts["total_sub_cnt"] +=  results["total_subjects"]
                        sg_eval_counts["total_pred_cnt"] +=  results["total_predicates"] 
                else:
                    print(f"[ERROR] len of gt data: {len(GT_tripdata)} len of ped data: {frame_pred_triplets}")
                    continue
                    # print(f"vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")


            for bm_key, bmdata in block_metric.items():
                """
                    average eval score for each frame and appned it to block
                """
                if len(frame_metric[bm_key]["precision"])>0 and len(frame_metric[bm_key]["recall"])>0:
                    block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]['precision'])))
                    block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]['recall'])))


        # overall
        for oam_key, oamdata in overall_metric.items():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric[oam_key]["precision"])>0 and len(block_metric[oam_key]["recall"])>0:
                overall_metric[oam_key]["precision"].append(round(float(np.average(np.array(block_metric[oam_key]['precision']))), 4))
                overall_metric[oam_key]["recall"].append(round(float(np.average(np.array(block_metric[oam_key]['recall']))), 4))

        # reset batch data
        batch_data = []

        try:
            with open(f"{inference_prog_output_dir}/{BATCH_CNT}_{TOTAL_BATCH-1}.txt", "w") as f:
                f.write(json.dumps(overall_metric, indent=4))
        except Exception as e:
            print(f"error saving file: {inference_prog_output_dir}/{BATCH_CNT}_{TOTAL_BATCH-1}.txt")
        
        BATCH_CNT +=1
        pbar.n += 1
        pbar.last_print_n = pbar.n
        pbar.refresh()

        sg_eval_counts["VRDFormer_Logic"] = {}
        total_vid_ids = len(overall_metric["triplet"]["precision"])
        for metric_key, metric_values in overall_metric.items():
            if metric_key not in sg_eval_counts["VRDFormer_Logic"].keys():
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {}
            
            if len(overall_metric[metric_key]["precision"])>0 and len(overall_metric[metric_key]["recall"])>0:
                overall_precision = np.average(np.array(overall_metric[metric_key]["precision"]))
                overall_recall = np.average(np.array(overall_metric[metric_key]["recall"]))
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {
                    "Precision@1": round(float(overall_precision), 4),
                    "Recall@1": round(float(overall_recall), 4),
                }
        
        # sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = BATCH_CNT*BATCH_SIZE
        sg_eval_counts["VRDFormer_Logic"]["TotalSamples"] = BATCH_CNT*BATCH_SIZE

        try:
            sg_eval_counts["dataset_meta"] ={
                "dataset_triplets_existing": GtData,
                "dataset_triplets_new": PredData
            }
        except Exception as e:
            pass

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val.json"
            # outputfile = f"{inference_output_dir}/results.json"
            with open(outputfile, "w") as f:
                json.dump(llava_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val_raw_response.json"
            # outputfile = f"{inference_output_dir}/results_raw_response.json"
            with open(outputfile, "w") as f:
                json.dump(llava_raw_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_results_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(sg_eval_counts,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        print(f"Processed batch {BATCH_CNT}/{TOTAL_BATCH_TO_PROCESS-1} in:{BATCH_PROCESS_TIME} Avg Inf time: {np.array(infer_times).mean()} Avg batch proc time: {np.array(batch_process_time).mean()}")

        if BATCH_CNT>TOTAL_BATCH_TO_PROCESS:
            print(f"Processed: {BATCH_CNT}/{TOTAL_BATCH_TO_PROCESS-1} random video samples out of total: {len(ALL_VIDEO_SAMPLES)} batch :{TOTAL_BATCH}")
            break