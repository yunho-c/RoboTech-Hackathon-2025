import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import time

# Constants
# SCALE_FACTOR = 1.0  # Scale factor for the video display
SCALE_FACTOR = 0.25  # Scale factor for the video display
WINDOW_OFFSET = 20  # Offset between windows in pixels

# Video path from file
with open("video_path.txt", "r") as f:
    VIDEO_PATH = f.readline().strip()


class VideoInspector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

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

        # Create main window
        with dpg.window(
            label="Video Inspector",
            width=self.display_width + 200,
            height=self.display_height * 2 + 150,
        ):
            # Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Previous Frame", callback=self.prev_frame)
                dpg.add_button(label="Next Frame", callback=self.next_frame)
                dpg.add_button(label="Play/Pause", callback=self.toggle_play)

            # Frame information
            with dpg.group():
                dpg.add_text("", tag="frame_info")

            # Original video window
            with dpg.child_window(
                width=self.display_width,
                height=self.display_height,
                tag="original_window",
            ):
                dpg.add_image(
                    "texture_original",
                    width=self.display_width,
                    height=self.display_height,
                )

            # Effect windows in a grid layout
            x_pos = self.display_width + WINDOW_OFFSET
            y_pos = 0

            with dpg.window(
                label="Effect 1 - Grayscale",
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

        # Create viewport
        dpg.create_viewport(
            title="Video Inspector",
            width=self.display_width * 2 + WINDOW_OFFSET * 2,
            height=self.display_height * 2 + WINDOW_OFFSET * 2,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set the first frame
        self.update_frame(0)

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

        # Process the frame for display
        self.process_and_display_frame(frame)

    def process_and_display_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create effect frames
        # Effect 1: Grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

        # Effect 2: Edge Detection
        frame_edges = cv2.Canny(frame_gray, 100, 200)
        frame_edges_rgb = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)

        # Effect 3: Blur
        frame_blur = cv2.GaussianBlur(frame, (15, 15), 0)
        frame_blur_rgb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to 0-1 range
        frame_rgb_f32 = frame_rgb.astype(np.float32) / 255.0
        frame_gray_rgb_f32 = frame_gray_rgb.astype(np.float32) / 255.0
        frame_edges_rgb_f32 = frame_edges_rgb.astype(np.float32) / 255.0
        frame_blur_rgb_f32 = frame_blur_rgb.astype(np.float32) / 255.0

        # Add alpha channel (all opaque)
        frame_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_rgba[:, :, 0:3] = frame_rgb_f32

        frame_gray_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_gray_rgba[:, :, 0:3] = frame_gray_rgb_f32

        frame_edges_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_edges_rgba[:, :, 0:3] = frame_edges_rgb_f32

        frame_blur_rgba = np.ones((self.height, self.width, 4), dtype=np.float32)
        frame_blur_rgba[:, :, 0:3] = frame_blur_rgb_f32

        # Flatten for DearPyGUI
        frame_rgba_flat = frame_rgba.flatten()
        frame_gray_rgba_flat = frame_gray_rgba.flatten()
        frame_edges_rgba_flat = frame_edges_rgba.flatten()
        frame_blur_rgba_flat = frame_blur_rgba.flatten()

        # Update textures
        dpg.set_value("texture_original", frame_rgba_flat)
        dpg.set_value("texture_effect1", frame_gray_rgba_flat)
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
