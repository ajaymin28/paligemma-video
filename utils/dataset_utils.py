from torch.utils.data import Dataset
from transformers import AutoProcessor
import yaml
import re
import json
import math
import random
from PIL import Image
from typing import Dict
import torch
import time
import os
from utils.video_processing import process_video_with_decord
from utils.config import Configuration

def rank0_print(*args):
    # if dist.is_initialized():
    #     if dist.get_rank() == 0:
    #         print(f"Rank {dist.get_rank()}: ", *args)
    # else:
    #     print(*args)
    print(*args)


def collate_fn_video_gemma3(examples,processor,video_data_root, device, train=True):
    batch_messages = []

    video_folder = video_data_root
    for example in examples:

        video_file = example["video"]
        video_file = os.path.join(video_folder, video_file)
        # suffix = video_file.split(".")[-1]
        if not os.path.exists(video_file):
            print("File {} not exist!".format(video_file))

        frame_indices = example["frame_indices"]
        video, num_frames_to_sample = process_video_with_decord(video_file,frame_indices_custom=frame_indices)
        # print("decord video sample: ", video.shape)

        # convert video(np.array) to PIL.Image
        frames = [Image.fromarray(video[i]).convert("RGB") for i in range(video.shape[0])]
        frame_tokens = [f"<image>" for _ in range(video.shape[0])]
        frame_tokens = "".join(frame_tokens)
        # print("len of frames: ", len(frames))

        # batch_images.append(frames)
        del video

        # extract user prompt
        user_prompt = example["conversations"][0]
        assert user_prompt["from"] == "human"

        assitant_respose = example["conversations"][1]
        assert assitant_respose["from"] == "gpt"

        message = [
            {
                "role": "user",
                "content": [{"type": "image", "image": frame} for frame in frames]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": assitant_respose["value"]}
                ]
            }
        ]
        
        ## add user prompt
        message[0]["content"].append({"type": "text", "text": user_prompt["value"]})

        batch_messages.append(message)

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(device)

    labels = inputs["input_ids"].clone()
    special_token_ids = processor.tokenizer.all_special_ids

    special_token_ids_tensor = torch.tensor(special_token_ids, device=labels.device)
    mask = torch.isin(labels, special_token_ids_tensor)
    labels[mask] = -100

    inputs["labels"] = labels

    return inputs

def collate_fn_video(examples,processor,video_data_root, device, train=True):
    batch_images = []
    batch_prompt = []
    batch_suffix = []

    video_folder = video_data_root
    for example in examples:

        video_file = example["video"]
        video_file = os.path.join(video_folder, video_file)
        # suffix = video_file.split(".")[-1]
        if not os.path.exists(video_file):
            print("File {} not exist!".format(video_file))

        frame_indices = example["frame_indices"]
        video, num_frames_to_sample = process_video_with_decord(video_file,frame_indices_custom=frame_indices)
        # print("decord video sample: ", video.shape)

        # convert video(np.array) to PIL.Image
        frames = [Image.fromarray(video[i]).convert("RGB") for i in range(video.shape[0])]
        frame_tokens = [f"<image>" for _ in range(video.shape[0])]
        frame_tokens = "".join(frame_tokens)
        # print("len of frames: ", len(frames))

        batch_images.append(frames)
        del video

        # extract user prompt
        user_prompt = example["conversations"][0]
        assert user_prompt["from"] == "human"

        user_prompt["value"] = user_prompt["value"].replace("<video>", frame_tokens)
        batch_prompt.append(user_prompt["value"])

        assitant_respose = example["conversations"][1]
        assert assitant_respose["from"] == "gpt"

        batch_suffix.append(assitant_respose["value"])

    if not train:
        print("Not using suffix")
        inputs = processor(
            images=batch_images,
            text=batch_prompt,
            return_tensors="pt",
            padding="longest",
        )
    else:
        inputs = processor(
            images=batch_images,
            text=batch_prompt,
            suffix=batch_suffix,
            return_tensors="pt",
            padding="longest",
        )

    inputs = inputs.to(torch.bfloat16).to(device)

    return inputs

class LazySupervisedDataset(Dataset):
    
    def __init__(self, data_path: str, video_data_root:str, processor: AutoProcessor, config: Configuration):
        super(LazySupervisedDataset, self).__init__()
        # self.tokenizer = tokenizer
        self.list_data_dict = []
        self.processor = processor
        dataset_paths = []
        self.video_data_root = video_data_root
        self.config = config

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            dataset_paths = []
            for file_name in file_names:
                dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = tokenizer
        # self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # if "image" in sources[0]:
        #     image_file = self.list_data_dict[i]["image"]
        #     if type(image_file) is list:
        #         image = [self.process_image(f) for f in image_file]
        #         # Handling multi images
        #         # overwrite to process with simple pad 
        #         if len(image_file) > 1:
        #             image = [self.process_image(f, "pad") for f in image_file]
        #             image = [[im[0], im[1], "image"] for im in image]
        #     else:
        #         image = [self.process_image(image_file)]
        #     sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.video_data_root
            video_file = os.path.join(video_folder, video_file)
            # suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            return self.list_data_dict[i]
        
        else:
            raise Exception("Only video data is supported for now.")
            # sources = copy.deepcopy([e["conversations"] for e in sources])