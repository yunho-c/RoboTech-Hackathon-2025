import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()

# Create a 100x100 texture with RGBA values [1.0, 0.0, 1.0, 1.0]
texture_data = np.ones((100, 100, 4), dtype=np.float32)
texture_data[:, :, 1] = 0.0  # Set green channel to 0
texture_data = texture_data.flatten()  # Flatten to 1D array for DearPyGUI

with dpg.texture_registry(show=True):
    dpg.add_dynamic_texture(
        width=100, height=100, default_value=texture_data, tag="texture_tag"
    )


def _update_dynamic_textures(sender, app_data, user_data):
    # Get color values and normalize to 0-1 range
    new_color = np.array(dpg.get_value(sender), dtype=np.float32) / 255.0

    # Create a 100x100x4 array with the selected color
    new_texture_data = np.ones((100, 100, 4), dtype=np.float32)
    # Broadcast the color to all pixels
    new_texture_data[:, :] = new_color

    # Flatten the array for DearPyGUI
    new_texture_data = new_texture_data.flatten()

    dpg.set_value("texture_tag", new_texture_data)


with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")
    dpg.add_color_picker(
        (255, 0, 255, 255),
        label="Texture",
        no_side_preview=True,
        alpha_bar=True,
        width=200,
        callback=_update_dynamic_textures,
    )


dpg.create_viewport(title="Custom Title", width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
