import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

    save_path = "./checkpoints/paligemma2-lora_r256_a512_lr1e-5_withemb"
    os.makedirs(save_path,exist_ok=True)

    # get the processor
    print(f"[INFO] loading {config.MODEL_ID} processor from hub...")
    processor = PaliGemmaProcessor.from_pretrained(config.MODEL_ID)

    ## add new tokens
    tokenizer = processor.tokenizer

    # Get original sizes
    original_vocab_size = tokenizer.vocab_size
    original_total_size = len(tokenizer)

    print(f"Original vocab size (pretrained): {original_vocab_size}")
    print(f"Original total tokenizer size (includes added tokens): {original_total_size}")

    added_tokens_count = tokenizer.add_tokens(["#frame", "#sgend"], special_tokens=False)

    # Get updated sizes
    new_total_size = len(tokenizer)

    print(f"Number of new tokens added: {added_tokens_count}")
    print(f"New total tokenizer size: {new_total_size}")

    # Attach updated tokenizer to processor if needed
    processor.tokenizer = tokenizer

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

    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Model's token embeddings resized to: {len(processor.tokenizer)}")

    video_data_root = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    train_dataset = LazySupervisedDataset(data_path="/home/ja882177/dso/gits/paligemma-video/data/ag_dataset.yaml",
                          video_data_root="/groups/sernam/datasets/ActionGenome/ActionGenome/videos",
                          processor=processor,
                          config=config)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=lambda examples: collate_fn_video(examples, processor, video_data_root, config.DEVICE),
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True)


    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=256,
        target_modules="all-linear",
        lora_alpha=512,
        lora_dropout=0.05,
        bias="none",
        # use_rslora=False,
        # use_dora=False,
        # modules_to_save=None
        init_lora_weights="olora"
    )

    """Additional Layers to Train"""
    additional_base_layers_to_train = ["emb"]
    for name, param in model.named_parameters():
        # print(name)
        for add_layers in additional_base_layers_to_train:
            if add_layers.lower() in name.lower()  and "lora" not in name.lower():
                if torch.is_floating_point(param):
                    print(name)
                    param.requires_grad = True

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model=model, peft_config=peft_config)
    model.to("cuda")
    print_trainable_params(model)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    
    # exit()

    # fine tune the model
    print("[INFO] fine tuning the model...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE
    )

    print(torch.cuda.memory_summary())

    gradient_accumulation_steps = 5
    accumulated_loss = 0

    for epoch in range(config.EPOCHS):
        for idx, batch in enumerate(train_dataloader):
            # print(list(batch.keys()))
            # print(f"Batch size: {len(batch['pixel_values'])}, Image shape: {batch['pixel_values'][0].shape}")
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0

            if idx % 100 == 0:
                print(f"Epoch: {epoch} Iter: {idx}/{len(train_dataloader)} Loss: {loss.item():.4f}")

            # print(torch.cuda.max_memory_allocated() / 1024**2, "MB")

            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            # Optional: clear GPU cache to reduce OOM risk
            torch.cuda.empty_cache()

    # If your model is already a PeftModel (after get_peft_model)
    model.save_pretrained(save_path)
    print(f"[INFO] LoRA adapter saved to {save_path}")
    processor.save_pretrained(save_path)
    print(f"[INFO] Processor saved to {save_path}")

    # # run model generation after fine tuning
    # infer_on_model(model, test_batch, before_pt=False)