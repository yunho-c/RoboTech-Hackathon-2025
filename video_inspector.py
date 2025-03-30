import cv2
import dearpygui.dearpygui as dpg
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import time
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import pyvista as pv
import threading

# local
import DAM
import GSAM2

# Constants
SCALE_FACTOR = 0.25  # Scale factor for the video display
WINDOW_OFFSET = 20  # Offset between windows in pixels

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Video path from file
with open("video_path_2.txt", "r") as f:
    VIDEO_PATH = f.readline().strip()


class VideoInspector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.depth_min = 0.0
        self.depth_max = 1.0
        self.pv_mesh = None
        self.pv_plotter = None
        self.pv_thread = None
        self.current_frame_rgb = None
        self.current_depth = None

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Load computer vision models
        self.depth_model = DAM.model
        self.grounding_processor = GSAM2.grounding_processor
        self.grounding_model = GSAM2.grounding_model
        self.track_target_query = "box."  # VERY important: text queries need to be lowercased + end with a dot
        self.sam_predictor = GSAM2.sam2_predictor

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set display dimensions
        self.display_width = int(self.width * SCALE_FACTOR)
        self.display_height = int(self.height * SCALE_FACTOR)

        # Current frame and playback state
        self.current_frame_idx = 0
        self.is_playing = False
        self.last_play_time = 0

        # Initialize 3D visualization
        self.setup_3d_visualization()

        # Initialize DearPyGUI
        self.setup_dpg()

    def setup_3d_visualization(self):
        """Initialize PyVista plotter in a separate thread"""
        self.pv_plotter = pv.Plotter()
        self.pv_plotter.set_background('black')
        self.pv_thread = threading.Thread(target=self.pv_plotter.show, daemon=True)
        self.pv_thread.start()

    def create_depth_mesh(self, frame_rgb, depth):
        """Create a 3D mesh from depth data and frame RGB"""
        # Normalize depth
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # Create grid coordinates
        rows, cols = depth.shape
        x = np.linspace(0, cols, cols)
        y = np.linspace(0, rows, rows)
        xx, yy = np.meshgrid(x, y)

        # Scale depth for better visualization
        zz = depth_normalized * 100

        # Create structured grid
        grid = pv.StructuredGrid(xx, yy, zz)

        # Add texture coordinates
        tex_coords = np.column_stack([
            xx.ravel() / xx.max(),
            yy.ravel() / yy.max()
        ])

        # Convert frame to texture
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        texture = pv.numpy_to_texture(frame_rgb)

        return grid, texture

    def update_3d_visualization(self, frame_rgb, depth):
        """Update the 3D visualization with new frame and depth data"""
        if self.pv_plotter is None:
            return

        # Create new mesh
        grid, texture = self.create_depth_mesh(frame_rgb, depth)

        # Clear previous actors
        self.pv_plotter.clear()

        # Add new mesh
        self.pv_plotter.add_mesh(
            grid,
            texture=texture,
            smooth_shading=True,
            show_edges=False
        )

        # Update camera to follow the mesh
        self.pv_plotter.camera_position = 'xy'
        self.pv_plotter.camera.azimuth = 30
        self.pv_plotter.camera.elevation = 30

        # Force render update
        self.pv_plotter.render()

    def setup_dpg(self):
        dpg.create_context()

        # Create texture registry
        with dpg.texture_registry(show=False):
            # Main video texture
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
                tag="texture_original",
            )

            # Effect textures
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
                tag="texture_effect1",
            )

            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
                tag="texture_effect2",
            )

            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
                tag="texture_effect3",
            )

        # Create main window for original video
        with dpg.window(
            label="Original Video",
            width=self.display_width,
            height=self.display_height,
            pos=[0, 0],
            no_resize=True,
        ):
            # Original video
            dpg.add_image(
                "texture_original",
                width=self.display_width,
                height=self.display_height,
            )

        # Effect 1 - Bottom left
        with dpg.window(
            label="Effect 1 - Depth Estimation",
            pos=[0, self.display_height + WINDOW_OFFSET],
            width=self.display_width,
            height=self.display_height,
            no_resize=True,
        ):
            dpg.add_image(
                "texture_effect1",
                width=self.display_width,
                height=self.display_height,
            )

        # Effect 2 - Bottom right
        with dpg.window(
            label="Effect 2 - Edge Detection",
            pos=[
                self.display_width + WINDOW_OFFSET,
                self.display_height + WINDOW_OFFSET,
            ],
            width=self.display_width,
            height=self.display_height,
            no_resize=True,
        ):
            dpg.add_image(
                "texture_effect2",
                width=self.display_width,
                height=self.display_height,
            )

        # Effect 3 - Object Detection
        with dpg.window(
            label="Effect 3 - Blur",
            pos=[self.display_width + WINDOW_OFFSET, 0],
            width=self.display_width,
            height=self.display_height,
            no_resize=True,
        ):
            dpg.add_image(
                "texture_effect3",
                width=self.display_width,
                height=self.display_height,
            )

        # Controls window - Below the second row
        with dpg.window(
            label="Controls",
            pos=[0, (self.display_height + WINDOW_OFFSET) * 2],
            width=self.display_width * 2 + WINDOW_OFFSET,
            height=150,  # Increased height for slider
            no_resize=True,
        ):
            # Frame information
            with dpg.group():
                dpg.add_text("", tag="frame_info")

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Previous Frame",
                    callback=self.prev_frame,
                )
                dpg.add_button(label="Next Frame", callback=self.next_frame)
                dpg.add_button(label="Play/Pause", callback=self.toggle_play)

            # Add frame slider
            with dpg.group(width=self.display_width * 2):
                dpg.add_slider_int(
                    label="Frame",
                    default_value=0,
                    min_value=0,
                    max_value=max(0, self.frame_count - 1),
                    width=self.display_width * 2 - 20,
                    callback=self.slider_frame_change,
                    tag="frame_slider",
                )

        # Normalization controls window - Below controls window
        with dpg.window(
            label="Depth Normalization",
            pos=[0, (self.display_height + WINDOW_OFFSET) * 2 + 150],
            width=self.display_width * 2 + WINDOW_OFFSET,
            height=150,
            no_resize=True,
        ):
            dpg.add_slider_float(
                label="Min Depth",
                default_value=0.0,
                min_value=0.0,
                max_value=255.0,
                width=self.display_width * 2 - 20,
                callback=self.update_depth_min,
                tag="depth_min_slider",
            )
            dpg.add_slider_float(
                label="Max Depth",
                default_value=255.0,
                min_value=0.0,
                max_value=255.0,
                width=self.display_width * 2 - 20,
                callback=self.update_depth_max,
                tag="depth_max_slider",
            )

        # Create viewport - Adjust height to include controls window
        dpg.create_viewport(
            title="Video Inspector",
            width=self.display_width * 2 + WINDOW_OFFSET * 2,
            height=self.display_height * 2
            + WINDOW_OFFSET * 3
            + 250,  # Add height for controls and normalization windows
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set the first frame
        self.update_frame(0)

    def slider_frame_change(self, sender, app_data):
        """Callback for when the slider value changes"""
        if app_data != self.current_frame_idx:
            self.is_playing = False
            self.update_frame(app_data)

    def update_depth_min(self, sender, app_data):
        """Callback for when the min depth slider changes"""
        self.depth_min = app_data
        self.update_frame(self.current_frame_idx)

    def update_depth_max(self, sender, app_data):
        """Callback for when the max depth slider changes"""
        self.depth_max = app_data
        self.update_frame(self.current_frame_idx)

    def update_frame(self, frame_idx):
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self.current_frame_idx = frame_idx

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            print(f"Failed to read frame {frame_idx}")
            return

        dpg.set_value(
            "frame_info",
            f"Frame: {frame_idx + 1}/{self.frame_count} | Time: {frame_idx / self.fps:.2f}s",
        )

        if dpg.does_item_exist("frame_slider"):
            current_callback = dpg.get_item_callback("frame_slider")
            dpg.set_item_callback("frame_slider", None)
            dpg.set_value("frame_slider", frame_idx)
            dpg.set_item_callback("frame_slider", current_callback)

        self.process_and_display_frame(frame)

    def apply_depth_estimation(self, frame):
        """Applies depth estimation and converts to RGB display format."""
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame)
            self.current_depth = depth  # Store for 3D visualization

        depth_normalized = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)
        depth_normalized = np.clip(depth_normalized, 0, 1) * 255
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_rgb = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_JET)
        return depth_rgb

    def apply_object_tracking(self, frame):
        """Applies object tracking using Grounding DINO and SAM."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.grounding_processor(
            images=image, text=self.track_target_query, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )

        if len(results[0]["boxes"]) == 0:
            return frame_rgb

        self.sam_predictor.set_image(frame_rgb)
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=results[0]["boxes"],
            multimask_output=False,
        )

        for mask in masks:
            color = np.array([0, 255, 0], dtype=np.uint8)
            frame_rgb = np.where(
                mask[..., None], 0.5 * color + 0.5 * frame_rgb, frame_rgb
            )

        return frame_rgb.astype(np.uint8)

    def apply_object_detection(self, frame):
        """Applies YOLO object detection and draws bounding boxes."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model = YOLO("yolov8n.pt")
        results = model(frame_rgb)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result.names[cls_id]} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )

                if y1 - text_height - 15 > 0:
                    text_y = y1 - 10
                    bg_y1 = y1 - text_height - 15
                    bg_y2 = y1
                else:
                    text_y = y2 + text_height + 5
                    bg_y1 = y2
                    bg_y2 = y2 + text_height + 10

                bg_x2 = min(x1 + text_width, frame_rgb.shape[1] - 1)
                cv2.rectangle(
                    frame_rgb,
                    (x1, bg_y1),
                    (bg_x2, bg_y2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame_rgb,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame_rgb

    def prepare_for_display(self, frame_rgb):
        """Converts an RGB frame to RGBA float32 format suitable for DPG texture."""
        frame_rgb_f32 = frame_rgb.astype(np.float32) / 255.0
        frame_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_rgba[:, :, 0:3] = frame_rgb_f32
        frame_rgba_flat = frame_rgba.flatten()
        return frame_rgba_flat

    def process_and_display_frame(self, frame):
        """Processes the frame using CV algorithms and updates DPG textures."""
        self.current_frame_rgb = frame  # Store for 3D visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_depth_rgb = self.apply_depth_estimation(frame)
        frame_tracking_rgb = self.apply_object_tracking(frame)
        frame_detection_rgb = self.apply_object_detection(frame)

        frame_rgba_flat = self.prepare_for_display(frame_rgb)
        frame_depth_rgba_flat = self.prepare_for_display(frame_depth_rgb)
        frame_tracking_rgba_flat = self.prepare_for_display(frame_tracking_rgb)
        frame_detection_rgba_flat = self.prepare_for_display(frame_detection_rgb)

        dpg.set_value("texture_original", frame_rgba_flat)
        dpg.set_value("texture_effect1", frame_depth_rgba_flat)
        dpg.set_value("texture_effect2", frame_tracking_rgba_flat)
        dpg.set_value("texture_effect3", frame_detection_rgba_flat)

        # Update 3D visualization
        if self.current_depth is not None and self.current_frame_rgb is not None:
            self.update_3d_visualization(self.current_frame_rgb, self.current_depth)

    def prev_frame(self):
        self.is_playing = False
        self.update_frame(self.current_frame_idx - 1)

    def next_frame(self):
        self.is_playing = False
        self.update_frame(self.current_frame_idx + 1)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.last_play_time = time.time()

    def run(self):
        while dpg.is_dearpygui_running():
            current_time = time.time()

            if self.is_playing:
                elapsed = current_time - self.last_play_time
                if elapsed >= 1.0 / self.fps:
                    self.last_play_time = current_time
                    next_frame = self.current_frame_idx + 1

                    if next_frame >= self.frame_count:
                        next_frame = 0

                    self.update_frame(next_frame)

            dpg.render_dearpygui_frame()

        self.cap.release()
        dpg.destroy_context()


if __name__ == "__main__":
    try:
        inspector = VideoInspector(VIDEO_PATH)
        inspector.run()
    except Exception as e:
        print(f"Error: {e}")
