import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.dataset_utils import LazySupervisedDataset, collate_fn_video_gemma3
from utils.config import Configuration
from utils.utilities import print_trainable_params
from utils.gpu_utils import memory_stats

import wandb

# Optional: better matmul perf on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True

class WandbMemCallback:
    """Tiny callback to log memory stats every Trainer log step."""
    def __init__(self): pass
    def on_log(self, args, state, control, **kwargs):
        try:
            if int(os.environ.get("RANK", "0")) == 0:
                mem = memory_stats(get_dict=True)
                wandb.log({f"mem/{k}": v for k, v in mem.items()}, step=state.global_step)
        except Exception:
            pass

if __name__ == "__main__":
    # ---------------------------
    # Config
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
    RUN_NAME = f"gemma-3-4b-it-lora_r{config.LORA_RANK}_a{config.LORA_ALPHA}_lr{config.LEARNING_RATE}_vision"
    save_path = f"./checkpoints/{RUN_NAME}"
    os.makedirs(save_path, exist_ok=True)

    # Make W&B use your project/run names (Trainer will pick these up)
    os.environ["WANDB_PROJECT"] = config.PROJECT_NAME
    os.environ["WANDB_RUN_NAME"] = RUN_NAME

    # ---------------------------
    # Processor
    # ---------------------------
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
    # Model (NO device_map here; DeepSpeed will shard it)
    # ---------------------------
    print(f"[INFO] loading {config.MODEL_ID} model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.MODEL_DTYPE,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation,
    )

    # Prepare for k-bit training & LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.LORA_RANK,
        target_modules="all-linear",
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        # modules_to_save=["lm_head", "embed_tokens"],  # optionally keep extra heads trainable
    )
    model = get_peft_model(model=model, peft_config=peft_config)

    # Optional: unfreeze extra base layers (kept from your code)
    additional_base_layers_to_train = [
        # "embed_tokens",
        # "vision",
        # "multi_modal_projector",
        # "attn",
        # "proj",
    ]
    if len(additional_base_layers_to_train) > 0:
        for name, param in model.named_parameters():
            for add_layers in additional_base_layers_to_train:
                if add_layers.lower() in name.lower() and "lora" not in name.lower():
                    param.requires_grad = True
        print("#" * 10)
        for name, param in model.named_parameters():
            for add_layers in additional_base_layers_to_train:
                if add_layers.lower() in name.lower() and "lora" not in name.lower():
                    print(name, param.requires_grad)
        print("#" * 10)

    print_trainable_params(model)

    # ---------------------------
    # Dataset (reuse yours)
    # ---------------------------
    video_data_root = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    train_dataset = LazySupervisedDataset(
        data_path="/home/ja882177/dso/gits/paligemma-video/data/ag_dataset.yaml",
        video_data_root=video_data_root,
        processor=processor,
        config=config,
    )

    # Collator must keep tensors on CPU; DeepSpeed will move them
    def data_collator(examples):
        return collate_fn_video_gemma3(
            examples, processor, video_data_root, device="cpu",  # <<< important
            max_frames=config.MAX_FRAMES_TO_TRAIN,
            isDeepSpeed=True  # doesnt move inputs to device
        )

    # ---------------------------
    # TrainingArguments (+ DeepSpeed)
    # ---------------------------
    # Tip: start with per_device_train_batch_size=1 and scale via grad_accum
    per_device_bs = 1

    training_args = TrainingArguments(
        output_dir=save_path,
        run_name=RUN_NAME,

        deepspeed="/home/ja882177/dso/gits/paligemma-video/config/deepspeed.json",     # <<< enable ZeRO-3 sharding
        bf16=True,
        tf32=True,

        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=5,  # you had 5; adjust effective BS via this
        learning_rate=config.LEARNING_RATE,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",

        num_train_epochs=config.EPOCHS,
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"],           # <<< W&B integration
        remove_unused_columns=False,   # required for custom collator
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        ddp_find_unused_parameters=True,  # safer with PEFT branches
        optim="adamw_bnb_8bit",           # memory-efficient optimizer
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        # callbacks=[WandbMemCallback()],  # tiny memory logger
    )

    # Init W&B only on rank 0 (optional; Trainer already reports if env vars set)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        try:
            wandb.init(project=config.PROJECT_NAME, name=RUN_NAME, config=vars(config))
            memory_stats(custom_label="Before training")
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")

    # ---------------------------
    # Train & Save
    # ---------------------------
    trainer.train()
    model.save_pretrained(save_path)            # saves PEFT adapters (rank 0)
    processor.save_pretrained(save_path)
    print(f"[INFO] LoRA adapter & processor saved to {save_path}")