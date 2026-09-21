import cv2
import numpy as np
from PIL import Image

def enhance_image(pil_image):
    """
    Apply CLAHE histogram equalisation to enhance
    image contrast without heavy computation.
    """
    # Convert PIL to numpy
    img = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to LAB colour space
    # Only equalise the L (lightness) channel
    # to preserve original colours
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back and convert to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_rgb)