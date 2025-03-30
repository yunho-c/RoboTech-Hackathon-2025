import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import time
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import DAM

# Constants
# SCALE_FACTOR = 1.0  # Scale factor for the video display
SCALE_FACTOR = 0.25  # Scale factor for the video display
WINDOW_OFFSET = 20  # Offset between windows in pixels

# Video path from file
# with open("video_path_1.txt", "r") as f:
with open("video_path_2.txt", "r") as f:
    VIDEO_PATH = f.readline().strip()


class VideoInspector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # self.depth_model = DepthAnythingV2(**model_configs[encoder])
        # self.depth_model.load_state_dict(
        #     torch.load(
        #         f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"
        #     )
        # )
        # self.depth_model = self.depth_model.to(device).eval()
        self.depth_model = DAM.model

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

        # Initialize DearPyGUI
        self.setup_dpg()

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
            # Frame information
            with dpg.group():
                dpg.add_text("", tag="frame_info")

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

        # Effect 3 - Top right
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
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Previous Frame",
                    callback=self.prev_frame,
                    width=150,
                    height=50,
                )
                dpg.add_button(
                    label="Next Frame", callback=self.next_frame, width=150, height=50
                )
                dpg.add_button(
                    label="Play/Pause", callback=self.toggle_play, width=150, height=50
                )

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

        # Create viewport - Adjust height to include controls window
        dpg.create_viewport(
            title="Video Inspector",
            width=self.display_width * 2 + WINDOW_OFFSET * 2,
            height=self.display_height * 2
            + WINDOW_OFFSET * 3
            + 100,  # Add height for controls window
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set the first frame
        self.update_frame(0)

    def slider_frame_change(self, sender, app_data):
        """Callback for when the slider value changes"""
        # Only update if the value actually changed to avoid infinite loops
        if app_data != self.current_frame_idx:
            self.is_playing = False  # Stop playback when manually changing frames
            self.update_frame(app_data)

    def update_frame(self, frame_idx):
        # Ensure frame index is within bounds
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self.current_frame_idx = frame_idx

        # Set the video position and read the frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            print(f"Failed to read frame {frame_idx}")
            return

        # Update frame info text
        dpg.set_value(
            "frame_info",
            f"Frame: {frame_idx + 1}/{self.frame_count} | Time: {frame_idx / self.fps:.2f}s",
        )

        # Update slider value (without triggering callback)
        if dpg.does_item_exist("frame_slider"):
            # Get current callback
            current_callback = dpg.get_item_callback("frame_slider")
            # Temporarily remove callback
            dpg.set_item_callback("frame_slider", None)
            # Update value
            dpg.set_value("frame_slider", frame_idx)
            # Restore callback
            dpg.set_item_callback("frame_slider", current_callback)

        # Process the frame for display
        self.process_and_display_frame(frame)

    def apply_depth_estimation(self, frame):
        """Applies depth estimation and converts to RGB display format."""
        # # Convert to RGB and normalize
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0

        # # Convert to tensor and add batch dimension
        # frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
        # frame_tensor = frame_tensor.to(next(self.depth_model.parameters()).device)

        # Get depth map
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame)

        # Normalize depth to 0-255 and convert to uint8
        # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        # depth_normalized = depth * 255
        depth_normalized = depth * 1
        depth_uint8 = depth_normalized.astype(np.uint8)

        # Convert to RGB (grayscale colormap)
        # depth_rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
        # depth_rgb = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        depth_rgb = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_JET)
        return depth_rgb

    def apply_edge_detection(self, frame):
        """Applies Canny edge detection and converts back to RGB."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_edges = cv2.Canny(frame_gray, 100, 200)
        frame_edges_rgb = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
        return frame_edges_rgb

    def apply_blur(self, frame):
        """Applies Gaussian blur and converts back to RGB."""
        frame_blur = cv2.GaussianBlur(frame, (15, 15), 0)
        frame_blur_rgb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)
        return frame_blur_rgb

    def prepare_for_display(self, frame_rgb):
        """Converts an RGB frame to RGBA float32 format suitable for DPG texture."""
        # Convert to float32 and normalize to 0-1 range
        frame_rgb_f32 = frame_rgb.astype(np.float32) / 255.0

        # Add alpha channel (all opaque)
        frame_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_rgba[:, :, 0:3] = frame_rgb_f32

        # Flatten for DearPyGUI
        frame_rgba_flat = frame_rgba.flatten()
        return frame_rgba_flat

    def process_and_display_frame(self, frame):
        """Processes the frame using CV algorithms and updates DPG textures."""
        # Convert original BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply effects
        frame_depth_rgb = self.apply_depth_estimation(frame)
        frame_edges_rgb = self.apply_edge_detection(frame)
        frame_blur_rgb = self.apply_blur(frame)

        # Prepare frames for display
        frame_rgba_flat = self.prepare_for_display(frame_rgb)
        frame_depth_rgba_flat = self.prepare_for_display(frame_depth_rgb)
        frame_edges_rgba_flat = self.prepare_for_display(frame_edges_rgb)
        frame_blur_rgba_flat = self.prepare_for_display(frame_blur_rgb)

        # Update textures
        dpg.set_value("texture_original", frame_rgba_flat)
        dpg.set_value("texture_effect1", frame_depth_rgba_flat)
        dpg.set_value("texture_effect2", frame_edges_rgba_flat)
        dpg.set_value("texture_effect3", frame_blur_rgba_flat)

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
        # Main loop
        while dpg.is_dearpygui_running():
            current_time = time.time()

            # Handle playback
            if self.is_playing:
                elapsed = current_time - self.last_play_time
                if elapsed >= 1.0 / self.fps:
                    self.last_play_time = current_time
                    next_frame = self.current_frame_idx + 1

                    # Loop back to the beginning if we reach the end
                    if next_frame >= self.frame_count:
                        next_frame = 0

                    self.update_frame(next_frame)

            dpg.render_dearpygui_frame()

        # Cleanup
        self.cap.release()
        dpg.destroy_context()


if __name__ == "__main__":
    try:
        inspector = VideoInspector(VIDEO_PATH)
        inspector.run()
    except Exception as e:
        print(f"Error: {e}")
