import cv2
import numpy as np
import os
import subprocess


def preprocess_frame(frame):
    # Convert from RGB to HSV directly
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define masks for sky, grass, and clouds
    masks = [
        cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255])),  # Sky
        cv2.inRange(hsv, np.array([50, 100, 100]), np.array([75, 255, 255])),  # Grass
        cv2.inRange(hsv, np.array([40, 0, 130]), np.array([100, 120, 255])),  # Clouds
    ]

    # Combine masks for unwanted regions
    combined_mask = cv2.bitwise_or(masks[0], masks[1])
    combined_mask = cv2.bitwise_or(combined_mask, masks[2])

    # Close small gaps in the cloud mask (only needed for clouds)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Invert mask to keep objects of interest
    objects_mask = cv2.bitwise_not(combined_mask)

    # Apply the mask and set non-background pixels to white
    result = np.zeros_like(frame)
    result[objects_mask > 0] = [255, 255, 255]

    # Convert to grayscale and crop
    grayscale_image = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)[:403, :]

    # Normalize and resize in one step
    scaled_image = (
        cv2.resize(grayscale_image, (64, 64), interpolation=cv2.INTER_AREA) / 255.0
    )

    return scaled_image

def clear_terminal():
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)
