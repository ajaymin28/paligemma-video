import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig

# from configs import object_detection_config
# from paligemma_ft.data_utis import collate_fn
# from paligemma_ft.model_utils import freeze_layers

from functools import partial
# from matplotlib import pyplot as plt, patches
from utils.dataset_utils import LazySupervisedDataset, collate_fn_video
from utils.config import Configuration
from utils.utilities import print_trainable_params
import os


# def infer_on_model(model, test_batch, before_pt=True):
#     # hardcoding the index to get same before and after results
#     index = 0

#     # help from : https://discuss.huggingface.co/t/vitimageprocessor-output-visualization/76335/6
#     mean = processor.image_processor.image_mean
#     std = processor.image_processor.image_std

#     pixel_value = test_batch["pixel_values"][index].cpu().to(torch.float32)

#     unnormalized_image = (
#         pixel_value.numpy() * np.array(std)[:, None, None]
#     ) + np.array(mean)[:, None, None]
#     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
#     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

#     with torch.inference_mode():
#         generated_outputs = model.generate(
#             **test_batch, max_new_tokens=100, do_sample=False
#         )
#         generated_outputs = processor.batch_decode(
#             generated_outputs, skip_special_tokens=True
#         )

#     if before_pt:
#         # generation of the pre trained model
#         for element in generated_outputs:
#             location = element.split("\n")[1]
#             if location == "":
#                 print("No bbox found")
#             else:
#                 print(location)
#     else:
#         # generation of the fine tuned model
#         element = generated_outputs[index]
#         detection_string = element.split("\n")[1]
#         objects = extract_objects(detection_string, 224, 224, unique_labels=False)
#         draw_bbox(unnormalized_image, objects)


if __name__ == "__main__":
    # get the device
    config = Configuration()

    # save_path = "./checkpoints/paligemma-lora"
    save_path = "./checkpoints/paligemma2-lora_r256_a512_lr1e-5_withemb"
    os.makedirs(save_path,exist_ok=True)

    # get the processor
    # config.MODEL_ID= "google/paligemma-3b-ft-nlvr2-448"
    # print(f"[INFO] loading {config.MODEL_ID} processor from hub...")
    processor = AutoProcessor.from_pretrained(save_path)

    ##
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',  # or 'fp4'
        bnb_4bit_compute_dtype=torch.bfloat16,  # or bfloat16 if supported
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )

    # load the pre trained model
    print(f"[INFO] loading {config.MODEL_ID} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.MODEL_DTYPE,
        device_map=config.DEVICE,
        # revision=config.MODEL_REVISION,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    # model.config.attn_implementation = "flash_attention_2"

    # Load adapter
    model = PeftModel.from_pretrained(model, save_path)


    video_data_root = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    test_dataset = LazySupervisedDataset(data_path="/home/ja882177/dso/gits/paligemma-video/data/ag_dataset_test.yaml",
                          video_data_root="/groups/sernam/datasets/ActionGenome/ActionGenome/videos",
                          processor=processor,
                          config=config)

    # Send to device
    model.to("cuda").eval()

    sample = test_dataset[1]

    inputs = collate_fn_video([sample], processor=processor, video_data_root=video_data_root,device=config.DEVICE)

    output = model.generate(**inputs, max_new_tokens=2000, do_sample=False,  cache_implementation="dynamic")
    print(processor.decode(output[0], skip_special_tokens=True))