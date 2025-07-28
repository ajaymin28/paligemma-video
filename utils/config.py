from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    MODEL_ID = "google/paligemma2-3b-pt-224" #"google/paligemma-3b-ft-nlvr2-448"  # google/paligemma2-3b-pt-224
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    MODEL_DTYPE = torch.bfloat16
    MODEL_REVISION = "bfloat16"
    EPOCHS = 1

    PROJECT_NAME: str = "video-paligemma-3b-pt-224"
    CHECKPOINT_ID: str = ""

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation = "flash_attention_2" # eager or flash_attention_2