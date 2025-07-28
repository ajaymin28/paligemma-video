from decord import VideoReader, cpu
import numpy as np

def process_video_with_decord(video_file, frame_indices_custom):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    frame_idx = frame_indices_custom

    video = vr.get_batch(frame_idx).asnumpy()

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, num_frames_to_sample