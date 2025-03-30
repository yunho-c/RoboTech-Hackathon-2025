import os
import copy
import cv2
import json
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from gsam_utils.common_utils import CommonUtils
from gsam_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from gsam_utils.video_utils import create_video_from_images

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# setup the input text prompt for SAM 2 and Grounding DINO
text = "box."

video_path = "/home/yunhocho/GitHub/test_images/videos/Moving_Boxes_360p.mp4"
output_dir = "./grounded_sam_reverse_outputs"
# output_dir = "./grounded_sam_reverse_outputs"
output_video_path = f"{output_dir}/{os.path.basename(video_path).split('.')[0]}.mp4"

# create the output directories
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
raw_image_dir = os.path.join(output_dir, "raw_images")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
CommonUtils.creat_dirs(raw_image_dir)

# init video capture and get total frames
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_path, offload_video_to_cpu=True, async_loading_frames=True)
step = 20  # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
objects_count = 0
frame_object_count = {}

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", total_frames)
for start_frame_idx in range(0, total_frames, step):
    print("start_frame_idx", start_frame_idx)
    
    # Read the frame directly from video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read frame {start_frame_idx}")
        continue
        
    # Save raw frame
    raw_frame_path = os.path.join(raw_image_dir, f"frame_{start_frame_idx:05d}.jpg")
    cv2.imwrite(raw_frame_path, frame)
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_base_name = f"frame_{start_frame_idx:05d}"
    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"]
    OBJECTS = results[0]["labels"]
    if input_boxes.shape[0] != 0:
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor
        """
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    else:
        print(f"No object detected in frame {start_frame_idx}, skip merge")
        mask_dict = sam2_masks

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    frame_object_count[start_frame_idx] = objects_count
    print("objects_count", objects_count)
    
    if len(mask_dict.labels) == 0:
        # Generate empty masks for the step frames
        for i in range(start_frame_idx, min(start_frame_idx + step, total_frames)):
            image_base_name = f"frame_{i:05d}"
            empty_mask = MaskDictionaryModel()
            empty_mask.mask_name = f"mask_{image_base_name}.npy"
            empty_mask.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=[f"frame_{i:05d}"])
        print(f"No object detected in frame {start_frame_idx}, skip")
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
        
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = f"frame_{out_frame_idx:05d}"
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}
for frame_idx, current_object_count in frame_object_count.items():
    print("reverse tracking frame", frame_idx, f"frame_{frame_idx:05d}")
    if frame_idx != 0:
        video_predictor.reset_state(inference_state)
        image_base_name = f"frame_{frame_idx:05d}"
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        for object_id in range(start_object_id+1, current_object_count+1):
            print("reverse tracking object", object_id)
            object_info_dict[object_id] = json_data.labels[object_id]
            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
    start_object_id = current_object_count
        
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2, start_frame_idx=frame_idx, reverse=True):
        image_base_name = f"frame_{out_frame_idx:05d}"
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        # merge the reverse tracking masks with the original masks
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu()
            if out_mask.sum() == 0:
                print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                continue
            object_info = object_info_dict[out_obj_id]
            object_info.mask = out_mask[0]
            object_info.update_box()
            json_data.labels[out_obj_id] = object_info
            mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
            mask_array[object_info.mask] = out_obj_id
        
        np.save(mask_data_path, mask_array)
        json_data.to_json(json_data_path)

"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(raw_image_dir, mask_data_dir, json_data_dir, result_dir+"_reverse")

create_video_from_images(result_dir+"_reverse", output_video_path, frame_rate=15)