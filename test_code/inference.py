'''
    This file is to inference arbitrary video files for Shot Cut
'''
import os, sys, shutil
import argparse
import numpy as np
import math
import subprocess
import cv2
import copy
from decord import VideoReader, cpu as decord_cpu
import json
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from config.argument_setting import get_args_parser
from architecture.backbone import build_backbone
from architecture.transformer import build_transformer
from architecture.model import OmniShotCut
from datasets.transforms import Video_Augmentation_Transform
from util.visualization import visualize_concated_frames
from config.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping, intra_int2string, inter_int2string


# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")




def load_model(checkpoint_path: str):


    # Check the checkpoint
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in state_dict or "model" not in state_dict:
        raise ValueError("Checkpoint must contain keys: 'args' and 'model'.")


    # Load the model
    model_args = state_dict["args"]
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = model_args.num_intra_relation_classes,
                            num_inter_relation_classes = model_args.num_inter_relation_classes,
                            num_frames = model_args.max_process_window_length,
                            num_queries = model_args.num_queries,
                            aux_loss = model_args.aux_loss,
                        )
    model.load_state_dict(state_dict["model"], strict=True)
    model.to("cuda")
    model.eval()


    return model, model_args



def get_video_fps_safe(video_path: str, default_fps: float = 24.0) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 1e-6 or math.isnan(fps):
            return default_fps
        return float(fps)
    except Exception:
        return default_fps



def split_videos(video, chunk_size, overlap_size):

    assert video.ndim == 4, "video must be (T, H, W, C)"
    assert overlap_size >= 0 and overlap_size < chunk_size

    T, H, W, C = video.shape
    stride = chunk_size - overlap_size

    # Form the return list
    return_list = []
    window_start_idx = 0

    while window_start_idx < T:

        window_end_idx = window_start_idx + chunk_size
        valid_len = min(chunk_size, T - window_start_idx)

        # Fetch current window
        chunk = video[window_start_idx:min(window_end_idx, T)]

        # Padding
        num_pad_frames = chunk_size - valid_len
        if num_pad_frames > 0:
            black = np.zeros((num_pad_frames, H, W, C), dtype=video.dtype)
            chunk = np.concatenate([chunk, black], axis=0)

        # Valid region for this window. We split the overlap region by half.
        left_overlap = overlap_size // 2
        right_overlap = overlap_size - left_overlap

        if window_start_idx == 0:
            valid_start_idx = 0
        else:
            valid_start_idx = window_start_idx + left_overlap

        if window_end_idx >= T:
            valid_end_idx = T
        else:
            valid_end_idx = window_end_idx - right_overlap

        return_list.append(
            [
                chunk,
                num_pad_frames,
                window_start_idx,
                valid_start_idx,
                valid_end_idx,
                valid_len,
            ]
        )

        # End
        if window_end_idx >= T:
            break

        window_start_idx += stride

    return return_list



def merge_predictions(pred_boundary_full, pred_boundary, duplicate_tolerance=2):

    # Sort
    pred_boundary = sorted(pred_boundary, key=lambda x: x["end_frame_idx"])

    # Merge
    for item in pred_boundary:

        # Check duplicate
        if len(pred_boundary_full) != 0:
            last_end_frame_idx = pred_boundary_full[-1]["end_frame_idx"]
            if abs(item["end_frame_idx"] - last_end_frame_idx) <= duplicate_tolerance:
                continue

        pred_boundary_full.append(item)

    return pred_boundary_full



def single_video_inference(video_path, model, model_args, overlap_window_length):


    # Init the parameter
    max_process_window_length = model_args.max_process_window_length
    process_height, process_width = model_args.process_height, model_args.process_width


    # Read the Video
    vr = VideoReader(video_path, ctx=decord_cpu(0), width=process_width, height=process_height)
    fps = vr.get_avg_fps()
    video_np_full = vr[:].asnumpy()  # (T, H, W, 3), RGB
    

    # Iterate all the clips
    pred_boundary_full = []

    for clip_idx, (video_np, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np_full, max_process_window_length, overlap_window_length)):

        # Transform
        video_tensor = video_transform(video_np).unsqueeze(0).to("cuda")


        # Inference
        with torch.inference_mode():
            outputs = model(video_tensor)
        

        # Choose the label with max value
        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1] 
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]  
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]  
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)


        pred_boundary = []
        start_frame_idx_local = 0

        for keep_idx in range(len(query_intra_idx)):

            # Fetch Label
            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            # Convert ranges from local window scale to video duration scale
            end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
            end_frame_idx_local = min(end_frame_idx_local, valid_len)

            pred_range = [start_frame_idx_local, end_frame_idx_local]
            pred_range_global = [
                window_start_idx + start_frame_idx_local,
                window_start_idx + end_frame_idx_local,
            ]

            # Sometimes model outputs the same start/end. Skip to avoid invalid range.
            if start_frame_idx_local >= end_frame_idx_local:
                continue

            # Append only the boundary inside the valid region
            end_frame_idx_global = window_start_idx + end_frame_idx_local

            if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_frame_idx_global),
                        "intra_label": int(pred_intra_label),
                        "inter_label": int(pred_inter_label),
                    }
                )

            start_frame_idx_local = end_frame_idx_local

            # End
            if end_frame_idx_local >= valid_len:
                break

        # Merge predicted results; here pred_boundary are already valid
        pred_boundary_full = merge_predictions(
            pred_boundary_full,
            pred_boundary,
        )


    # Convert boundary to range
    pred_ranges_full = []
    pred_intra_labels_full = []
    pred_inter_labels_full = []

    start_frame_idx_local = 0

    for item in pred_boundary_full:

        end_frame_idx = int(item["end_frame_idx"])

        if end_frame_idx <= start_frame_idx_local:
            continue

        pred_ranges_full.append(
            [
                int(start_frame_idx_local),
                int(end_frame_idx),
            ]
        )
        pred_intra_labels_full.append(int(item["intra_label"]))
        pred_inter_labels_full.append(int(item["inter_label"]))

        start_frame_idx_local = end_frame_idx


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps




def dump_list_of_dict(data, save_path, indent=4):
    """
    Save list[dict] as JSON.
    Convert pred_intra_labels / pred_inter_labels from int to string.
    """

    def convert_item(item):
        item = copy.deepcopy(item)

        if "pred_intra_labels" in item:
            item["pred_intra_labels"] = [
                intra_int2string.get(x, f"Unknown_{x}")
                for x in item["pred_intra_labels"]
            ]

        if "pred_inter_labels" in item:
            item["pred_inter_labels"] = [
                inter_int2string.get(x, f"Unknown_{x}")
                for x in item["pred_inter_labels"]
            ]

        return item

    def format_dict(d, level):
        indent_str = " " * (indent * level)
        inner_indent = " " * (indent * (level + 1))

        lines = [indent_str + "{"]
        items = list(d.items())

        for i, (k, v) in enumerate(items):
            value_str = json.dumps(v, ensure_ascii=False)
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'{inner_indent}"{k}": {value_str}{comma}')

        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    data = [convert_item(item) for item in data]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        for i, item in enumerate(data):
            dict_str = format_dict(item, level=1)
            comma = "," if i < len(data) - 1 else ""
            f.write(dict_str + comma + "\n")

        f.write("]\n")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                            "--checkpoint_path",
                            type = str,
                            default = "checkpoints/OmniShotCut_ckpt.pth",
                            help = "Path to checkpoint file."
                        )
    parser.add_argument(
                            "--input_video_path",
                            type = str,
                            default = "__assets__/demo_video1.mp4",
                            help = "Path to the input video path."
                        )
    parser.add_argument(
                            "--result_store_path",
                            type = str,
                            default = "results.json",
                            help="Path to save result json."
                        )
    parser.add_argument(
                            "--overlap_window_length",
                            type = int,
                            default = 20,
                            help = "Number of overlapped frames between adjacent inference windows."
                        )
    parser.add_argument(
                            "--visual_store_folder_path",
                            type = str,
                            default = "demo_video_results",
                            help = "Path to save the visualization results. Set to None to disable."
                        )
    parser.add_argument(
                            "--mode",
                            type = str,
                            default = "default",
                            choices = ["default", "clean_shot"],
                            help = "Output Mode. 'default' means all Intra and Inter label. 'clean_shot' means only General Shot Cut without transitions. "
                        )

    return parser.parse_args()




if __name__ == '__main__':

    # Setting
    inference_args = parse_args()
    checkpoint_path = inference_args.checkpoint_path
    input_video_path = inference_args.input_video_path
    assert(os.path.exists(input_video_path))
    result_store_path = inference_args.result_store_path
    visual_store_folder_path = inference_args.visual_store_folder_path
    mode = inference_args.mode
    


    # Prepare the folder
    if visual_store_folder_path is not None:
        if os.path.exists(visual_store_folder_path):
            shutil.rmtree(visual_store_folder_path)
        os.makedirs(visual_store_folder_path)


    # Load Checkpoint & Model Config
    assert(os.path.exists(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_args = state_dict['args']
    print("Checkpoint stored args are", model_args)


    # Init the Model
    print("Loading OmniShotCut Model!")
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = model_args.num_intra_relation_classes,
                            num_inter_relation_classes = model_args.num_inter_relation_classes,
                            num_frames = model_args.max_process_window_length, 
                            num_queries = model_args.num_queries,
                            aux_loss = model_args.aux_loss,
                        )
    model.load_state_dict(state_dict['model'], strict=True)
    model.to("cuda")
    model.eval()




    # Do the inference
    print("Do the inference!")
    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps = single_video_inference(
                                                                                                                    input_video_path,
                                                                                                                    model,
                                                                                                                    model_args,
                                                                                                                    inference_args.overlap_window_length,
                                                                                                                )



    # Visualize
    if visual_store_folder_path is not None:
        print("Visualize the results!")
        pred_saved_paths = visualize_concated_frames(
                                                        video_np_full,
                                                        visual_store_folder_path,
                                                        pred_ranges_full,
                                                        max_frames_per_img=264,
                                                        end_range_exclusive=True,
                                                        fps=fps,
                                                        start_index = 0,
                                                    )



     # For Clean Shot mode, we only leave general types
    if mode == "clean_shot":     
        
        # Clean shot mode (No Transitions in the middle)
        general_type_idx = unique_intra_label_mapping["general"]
        effective_indices = []
        for idx, intra_label in enumerate(pred_intra_labels_full):
            if intra_label == general_type_idx:
                effective_indices.append(idx)
        
        # Reassign the shots
        pred_ranges_full = np.array(pred_ranges_full)[effective_indices].tolist()
        pred_intra_labels_full = np.array(pred_intra_labels_full)[effective_indices].tolist()
        pred_inter_labels_full = np.array(pred_inter_labels_full)[effective_indices].tolist()

    

    # Collect prediction results
    pred_result = {}
    pred_result["video_path"] = input_video_path
    pred_result["pred_ranges"] = pred_ranges_full
    pred_result["pred_intra_labels"] = pred_intra_labels_full
    pred_result["pred_inter_labels"] = pred_inter_labels_full
    
    # Dump to json
    dump_list_of_dict([pred_result], result_store_path)


    print("Finished!")