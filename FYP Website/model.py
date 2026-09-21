from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import numpy as np

def load_model(weights_path="weights/best.pt"):
    """Load YOLO26 model wrapped with SAHI."""
    model = AutoDetectionModel.from_pretrained(
        model_type   = "ultralytics",
        model_path   = weights_path,
        confidence_threshold = 0.25,
        device       = "cpu"   # use cpu for free hosting
    )
    return model

def run_inference(model, pil_image):
    """
    Run SAHI sliced inference on high resolution image.
    Slices the 4000x2250 image into 640x640 tiles
    with 20% overlap, then merges detections.
    """
    result = get_sliced_prediction(
        image            = pil_image,
        detection_model  = model,
        slice_height     = 640,
        slice_width      = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio  = 0.2,
        verbose          = 0
    )

    detections = []
    for obj in result.object_prediction_list:
        bbox       = obj.bbox
        class_name = obj.category.name
        score      = obj.score.value
        # Get centre point of bounding box
        cx = int((bbox.minx + bbox.maxx) / 2)
        cy = int((bbox.miny + bbox.maxy) / 2)
        detections.append({
            "class"     : class_name,
            "confidence": round(score, 3),
            "center_x"  : cx,
            "center_y"  : cy,
        })

    return detections



import cv2

def draw_dots_on_image(pil_image, detections):
    """
    Draw red dot for unhealthy, blue dot for healthy trees.
    Returns annotated PIL image and summary counts.
    """
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    healthy_count   = 0
    unhealthy_count = 0
    dot_radius      = 10
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale      = 0.6
    thickness       = 2

    for i, det in enumerate(detections):
        cx = det["center_x"]
        cy = det["center_y"]
        label = f"{det['class']} ({det['confidence']})"

        if det["class"].lower() == "healthy":
            colour = (255, 0, 0)   # blue in BGR
            healthy_count += 1
        else:
            colour = (0, 0, 255)   # red in BGR
            unhealthy_count += 1

        # Draw dot
        cv2.circle(img_bgr, (cx, cy), dot_radius, colour, -1)

        # Draw label number next to dot
        cv2.putText(
            img_bgr,
            str(i + 1),
            (cx + 14, cy + 5),
            font, font_scale,
            colour, thickness
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated = Image.fromarray(img_rgb)

    return annotated, healthy_count, unhealthy_count, detections