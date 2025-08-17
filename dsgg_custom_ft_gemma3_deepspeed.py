import os
import json
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import deepspeed
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.dataset_utils import LazySupervisedDataset, collate_fn_video_gemma3
from utils.config import Configuration
from utils.utilities import print_trainable_params
from utils.gpu_utils import memory_stats
from transformers.feature_extraction_utils import BatchFeature

import wandb

# Better matmul perf on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True

# ---- helpers ----
def is_main_process() -> bool:
    try:
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True

# def move_to_device(batch, device):
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device, non_blocking=True)
#     if isinstance(batch, dict):
#         return {k: move_to_device(v, device) for k, v in batch.items()}
#     if isinstance(batch, (list, tuple)):
#         return type(batch)(move_to_device(v, device) for v in batch)
#     return batch

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        # print(f"Moving tensor to {device}")
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        # print(f"Processing dictionary with keys: {list(batch.keys())}")
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (dict, BatchFeature)):
        # print(f"Processing {type(batch).__name__} with keys: {list(batch.keys())}")
        return type(batch)({k: move_to_device(v, device) for k, v in batch.items()})
    elif isinstance(batch, (list, tuple)):
        # print(f"Processing {type(batch).__name__} with {len(batch)} elements")
        return type(batch)(move_to_device(v, device) for v in batch)
    else:
        print(f"Skipping non-tensor type: {type(batch)}")
        return batch

if __name__ == "__main__":
    # ---------------------------
    # Config / env
    # ---------------------------
    config = Configuration()
    config.LORA_RANK = 16
    config.LORA_ALPHA = 16
    config.LEARNING_RATE = 2e-4
    config.PROJECT_NAME = "gemma3-it-video_dsgg"
    config.attn_implementation = ["sdpa", "eager", "flash_attention_2"][2]
    config.EPOCHS = 1
    config.MAX_FRAMES_TO_TRAIN = 8
    config.MODEL_ID = "google/gemma-3-4b-it"  # or "google/gemma-3n-e4b-it"
    config.GRADIENT_ACCU_STEPS = 5

    RUN_NAME = f"gemma-3-4b-it-lora_r{config.LORA_RANK}_a{config.LORA_ALPHA}_lr{config.LEARNING_RATE}_justlora_8frames_deepspeed_customft_loop"
    save_path = f"./checkpoints/{RUN_NAME}"
    os.makedirs(save_path, exist_ok=True)

    os.environ["WANDB_PROJECT"] = config.PROJECT_NAME
    os.environ["WANDB_RUN_NAME"] = RUN_NAME
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ---------------------------
    # DeepSpeed init (dist)
    # ---------------------------
    deepspeed.init_distributed(timeout=timedelta(minutes=30))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Load DS config (same JSON you used with Trainer)
    ds_config_path = "/home/ja882177/dso/gits/paligemma-video/config/deepspeed.json"
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    # ---------------------------
    # Processor
    # ---------------------------
    if is_main_process():
        print(f"[INFO] loading {config.MODEL_ID} processor from hub...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)

    # ---------------------------
    # 4-bit quantization (QLoRA)
    # ---------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=None,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # ---------------------------
    # Model (no device_map, no .to("cuda"))
    # ---------------------------
    if is_main_process():
        print(f"[INFO] loading {config.MODEL_ID} model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.MODEL_DTYPE,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation,
    )

    # Prepare for k-bit training & LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.LORA_RANK,
        target_modules="all-linear",
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model=model, peft_config=peft_config)

    # Optionally unfreeze more base layers
    for _name, _p in model.named_parameters():
        pass  # keep LoRA-only unless you add patterns like "vision", "proj", etc.

    if is_main_process():
        print_trainable_params(model)

    # ---------------------------
    # Dataset / Sampler / DataLoader
    # ---------------------------
    video_data_root = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    train_dataset = LazySupervisedDataset(
        data_path="/home/ja882177/dso/gits/paligemma-video/data/ag_dataset.yaml",
        video_data_root=video_data_root,
        processor=processor,
        config=config,
    )

    # DistributedSampler for proper sharding
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True,
        drop_last=False,
    )

    # IMPORTANT: keep tensors on CPU in collator; DS moves after we .to(engine.device)
    def data_collator(examples):
        return collate_fn_video_gemma3(
            examples,
            processor,
            video_data_root,
            train=True,
            device="cuda",                    # <<< don't move to CUDA here
            max_frames=config.MAX_FRAMES_TO_TRAIN,
            isDeepSpeed=True,                # <<< collator should avoid .to(device)
        )

    # Keep micro-batch-size-per-GPU == deepspeed.json (1). Accumulate for global batch.
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=data_collator,
        batch_size=ds_config.get("train_micro_batch_size_per_gpu", 1),
        num_workers=4,
        pin_memory=True,
    )

    # ---------------------------
    # Optimizer (8-bit if available)
    # ---------------------------
    # try:
    #     from bitsandbytes.optim import AdamW8bit as AdamW
    # except Exception:
    
    from torch.optim import AdamW  # fallback

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=config.LEARNING_RATE)

    # (Optional) scheduler – keep constant LR like your manual loop.
    lr_scheduler = None

    # ---------------------------
    # Build DeepSpeed engine
    # ---------------------------
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optim_params,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )

    print(f"Deepspeed model engine device: {model_engine.device}")

    # ---------------------------
    # W&B
    # ---------------------------
    if is_main_process():
        try:
            wandb.init(project=config.PROJECT_NAME, name=RUN_NAME, config=vars(config))
            memory_stats(custom_label="Before training")
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")

    # ---------------------------
    # Train (custom loop, manual grad-accum)
    # ---------------------------
    global_step = 0
    model_engine.train()

    for epoch in range(config.EPOCHS):
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # Move batch to the engine device
            batch = move_to_device(batch, model_engine.device)

            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"Tensor {k} is on device: {v.device}")

            # Forward + loss
            outputs = model_engine(**batch)
            loss = outputs.loss / config.GRADIENT_ACCU_STEPS

            # Backward with DS
            model_engine.backward(loss)

            # Step every grad_accum steps
            if (step + 1) % config.GRADIENT_ACCU_STEPS == 0:
                model_engine.step()      # includes optimizer.step + zeroing grads
            # If you want to zero more aggressively, uncomment:
            # else:
            #     model_engine.optimizer.zero_grad(set_to_none=True)

            # Logging (rank 0)
            if is_main_process() and (global_step % 100 == 0):
                try:
                    log_data = {
                        "train/loss": loss.item() * config.GRADIENT_ACCU_STEPS,  # unscaled
                        "train/epoch": epoch,
                    }
                    log_data.update(memory_stats(get_dict=True))
                    wandb.log(log_data, step=global_step)
                except Exception as e:
                    print(f"[WARN] wandb.log failed: {e}")

            global_step += 1

            # Optional: clear cache occasionally
            if (global_step % 200) == 0:
                torch.cuda.empty_cache()

        if is_main_process():
            print(f"[INFO] finished epoch {epoch+1}/{config.EPOCHS}")

    # ---------------------------
    # Save (rank 0 only)
    # ---------------------------
    if is_main_process():
        # engine.module is the PEFT-wrapped model
        model_engine.module.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"[INFO] LoRA adapter & processor saved to {save_path}")