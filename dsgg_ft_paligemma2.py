import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.dataset_utils import LazySupervisedDataset, collate_fn_video
from utils.config import Configuration
from utils.utilities import print_trainable_params
import os


if __name__ == "__main__":
    # get the device
    config = Configuration()

    config.MODEL_ID = "google/paligemma2-3b-pt-224"

    save_path = "./checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_mm_proj"
    os.makedirs(save_path,exist_ok=True)

    # get the processor
    print(f"[INFO] loading {config.MODEL_ID} processor from hub...")
    processor = PaliGemmaProcessor.from_pretrained(config.MODEL_ID)

    ##
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',  # or 'fp4'
        bnb_4bit_compute_dtype=torch.bfloat16,  # or bfloat16 if supported
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=None
    )

    # load the pre trained model
    print(f"[INFO] loading {config.MODEL_ID} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.MODEL_DTYPE,
        device_map=config.DEVICE,
        # revision=config.MODEL_REVISION,
        # quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )

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

    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model=model, peft_config=peft_config)
    model.to("cuda")

    """Additional Layers to Train"""
    additional_base_layers_to_train = [
                                        # "embed_tokens",
                                        # "vision", 
                                        "multi_modal_projector"
                                    ]
    
    for name, param in model.named_parameters():
        # print(name)
        for add_layers in additional_base_layers_to_train:
            if add_layers.lower() in name.lower()  and "lora" not in name.lower():
                # if torch.is_floating_point(param):
                print(name)
                param.requires_grad = True

    print("#"*10)
    for name, param in model.named_parameters():
        for add_layers in additional_base_layers_to_train:
            if add_layers.lower() in name.lower()  and "lora" not in name.lower():
                print(name, param.requires_grad)
    print("#"*10)

    print_trainable_params(model)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    

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

            # Optional: clear GPU cache to reduce OOM risk
            torch.cuda.empty_cache()

    # If your model is already a PeftModel (after get_peft_model)
    model.save_pretrained(save_path)
    print(f"[INFO] LoRA adapter saved to {save_path}")
    processor.save_pretrained(save_path)
    print(f"[INFO] Processor saved to {save_path}")