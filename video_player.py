import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import os

class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.playing = False
        self.texture_id = None
        self.video_path = ""

    def load_video(self, video_path):
        """Load a video file and initialize parameters."""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            return False

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'.")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

        # Set slider max value
        dpg.set_value("frame_slider", 0)
        dpg.configure_item("frame_slider", max_value=self.total_frames-1)

        # Update frame counter text
        self.update_frame_counter()

        # Load first frame
        self.show_frame(0)

        return True

    def show_frame(self, frame_number):
        """Display the specified frame."""
        if self.cap is None:
            return

        # Ensure frame number is within valid range
        frame_number = max(0, min(frame_number, self.total_frames - 1))

        # Set position to the requested frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = self.cap.read()

        if not ret:
            print(f"Error: Could not read frame {frame_number}.")
            return

        # Update current frame number
        self.current_frame = frame_number

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed to fit display area
        display_width = dpg.get_item_width("image_display")
        display_height = dpg.get_item_height("image_display")

        # Calculate aspect ratio preserving resize
        aspect_ratio = self.frame_width / self.frame_height

        if display_width / display_height > aspect_ratio:
            # Height is the limiting factor
            new_height = display_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Width is the limiting factor
            new_width = display_width
            new_height = int(new_width / aspect_ratio)

        # Resize the frame
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

        # Update texture
        if self.texture_id is None:
            with dpg.texture_registry():
                self.texture_id = dpg.add_dynamic_texture(new_width, new_height, frame_resized)
                # Update the image widget to use our new texture
                dpg.configure_item("image", texture_tag=self.texture_id)
        else:
            dpg.set_value(self.texture_id, frame_resized)

        # # Update the image widget
        # dpg.configure_item("image", texture_tag=self.texture_id)

        # Update slider value (without triggering callback)
        dpg.set_value("frame_slider", frame_number)

        # Update frame counter
        self.update_frame_counter()

    def update_frame_counter(self):
        """Update the frame counter text."""
        if self.total_frames > 0:
            dpg.set_value("frame_counter", f"Frame: {self.current_frame + 1}/{self.total_frames}")

    def next_frame(self):
        """Show the next frame."""
        if self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame + 1)

    def prev_frame(self):
        """Show the previous frame."""
        if self.current_frame > 0:
            self.show_frame(self.current_frame - 1)

    def jump_to_frame(self, frame_number):
        """Jump to a specific frame."""
        self.show_frame(int(frame_number))

    def play_pause(self):
        """Toggle play/pause state."""
        self.playing = not self.playing

        # Update button text
        if self.playing:
            dpg.set_item_label("play_pause_button", "Pause")
        else:
            dpg.set_item_label("play_pause_button", "Play")

    def on_frame_change(self, sender, app_data):
        """Callback for slider movement."""
        self.jump_to_frame(app_data)

    def on_file_dialog(self, sender, app_data):
        """Callback for file dialog."""
        file_path = app_data["file_path_name"]
        if file_path:
            self.load_video(file_path)

    def update(self):
        """Update function called every frame."""
        if self.playing and self.cap is not None:
            if self.current_frame < self.total_frames - 1:
                self.next_frame()
            else:
                # Stop playing when reaching the end
                self.playing = False
                dpg.set_item_label("play_pause_button", "Play")


def main():
    # Initialize DearPyGui
    dpg.create_context()
    dpg.create_viewport(title="Video Player", width=800, height=600)
    dpg.setup_dearpygui()

    # Create player instance
    player = VideoPlayer()

    # File dialog callback
    def file_dialog_callback(sender, app_data):
        player.on_file_dialog(sender, app_data)

    # Create file dialog
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=file_dialog_callback,
        id="file_dialog",
        width=700,
        height=400
    ):
        dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
        dpg.add_file_extension(".avi", color=(0, 255, 0, 255))
        dpg.add_file_extension(".mov", color=(0, 255, 0, 255))
        dpg.add_file_extension(".MOV", color=(0, 255, 0, 255))
        dpg.add_file_extension(".mkv", color=(0, 255, 0, 255))
        dpg.add_file_extension(".*")

    # Create a default blank texture
    with dpg.texture_registry():
        # Create a small blank dark gray image as default
        default_width, default_height = 640, 360
        blank_image = np.ones((default_height, default_width, 3), dtype=np.uint8) * 25  # Dark gray
        default_texture = dpg.add_static_texture(default_width, default_height, blank_image)

    # Main window
    with dpg.window(label="Video Player", tag="main_window"):
        # Menu bar
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open Video", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

        # Image display area
        with dpg.group(horizontal=False):
            # Frame display
            with dpg.child_window(width=-1, height=-80, tag="image_display"):
                dpg.add_image(default_texture, tag="image")

            # Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open Video", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_button(label="Previous", callback=lambda: player.prev_frame())
                dpg.add_button(label="Play", tag="play_pause_button", callback=lambda: player.play_pause())
                dpg.add_button(label="Next", callback=lambda: player.next_frame())
                dpg.add_text("Frame: 0/0", tag="frame_counter")

            # Slider
            dpg.add_slider_int(
                label="",
                default_value=0,
                min_value=0,
                max_value=100,
                width=-1,
                tag="frame_slider",
                callback=lambda sender, app_data: player.on_frame_change(sender, app_data)
            )

    # Set main window to fill viewport
    dpg.set_primary_window("main_window", True)

    # Show viewport
    dpg.show_viewport()

    # Main loop
    while dpg.is_dearpygui_running():
        # Update player (for playback)
        player.update()

        # Render frame
        dpg.render_dearpygui_frame()

    # Cleanup
    dpg.destroy_context()


if __name__ == "__main__":
    main()
