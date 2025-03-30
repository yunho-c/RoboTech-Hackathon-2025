# source: https://github.com/yunho-c/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id.py
import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_id = "IDEA-Research/grounding-dino-tiny"
grounding_processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    DEVICE
)

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

# video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
# sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
# image_predictor = SAM2ImagePredictor(sam2_image_model)
sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
