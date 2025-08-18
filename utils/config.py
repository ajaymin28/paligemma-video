from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    MODEL_ID = "google/paligemma2-3b-pt-224" #"google/paligemma-3b-ft-nlvr2-448"  # google/paligemma2-3b-pt-224
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    MODEL_DTYPE = torch.bfloat16
    MODEL_REVISION = "bfloat16"
    NUM_DL_WORKERS = 4
    EPOCHS = 1

    PROJECT_NAME: str = "video-paligemma-3b-pt-224"
    CHECKPOINT_ID: str = ""

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation = "flash_attention_2" # eager or flash_attention_2

    GRADIENT_ACCU_STEPS = 5
    WANDB_PROJECT_NAME = ""

    LORA_RANK = 128
    LORA_ALPHA = 256
    MAX_FRAMES_TO_TRAIN = 6

    HF_HUB_MODEL_SAVE_ID = "ajaymin28/Gemma3-video-qlora-AG-8-frames-rank32"