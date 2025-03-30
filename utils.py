import numpy as np
import random


def generate_random_color(output_format="list"):
    """
    Generate a random RGB color in specified format.

    Args:
        output_format (str): Either 'np.ndarray' or 'list' (default)

    Returns:
        np.ndarray or list: Random RGB color in specified format
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    if output_format == "np.ndarray":
        return np.array([r, g, b])
    else:
        return [r, g, b]
