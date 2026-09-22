from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import numpy as np
import cv2

def load_model(weights_path="weights/best.pt"):
    model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = weights_path,
        confidence_threshold = 0.25,
        device               = "cpu"
    )
    return model

def apply_nms(detections, iou_threshold=0.4):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    of the same tree across multiple SAHI tiles.

    iou_threshold: lower = more aggressive duplicate removal
                   higher = keeps more detections
    Start with 0.4 and adjust based on your results.
    """
    if len(detections) == 0:
        return detections

    boxes  = []
    scores = []
    cls    = []

    for det in detections:
        cx = det["center_x"]
        cy = det["center_y"]
        w  = det["width"]
        h  = det["height"]
        # Convert centre format to x1y1x2y2 for NMS
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(det["confidence"])
        cls.append(det["class"])

    boxes_np  = np.array(boxes,  dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes          = boxes_np.tolist(),
        scores          = scores_np.tolist(),
        score_threshold = 0.25,
        nms_threshold   = iou_threshold
    )

    if len(indices) == 0:
        return []

    kept = []
    for i in indices.flatten():
        kept.append(detections[i])

    return kept

def run_inference(model, pil_image):
    """
    Run SAHI sliced inference then apply NMS
    to remove duplicate detections of the same tree.
    """
    img_w, img_h = pil_image.size

    result = get_sliced_prediction(
        image                = pil_image,
        detection_model      = model,
        slice_height         = 640,
        slice_width          = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio  = 0.2,
        verbose              = 0,
        # SAHI has its own NMS — enable it as first pass
        postprocess_type           = "NMS",
        postprocess_match_metric   = "IOU",
        postprocess_match_threshold= 0.4,
        postprocess_class_agnostic = False
    )

    detections = []
    for obj in result.object_prediction_list:
        bbox  = obj.bbox
        cx    = int((bbox.minx + bbox.maxx) / 2)
        cy    = int((bbox.miny + bbox.maxy) / 2)
        bw    = bbox.maxx - bbox.minx
        bh    = bbox.maxy - bbox.miny

        detections.append({
            "class"      : obj.category.name,
            "confidence" : round(obj.score.value, 3),
            "center_x"   : cx,
            "center_y"   : cy,
            "width"      : bw,
            "height"     : bh,
        })

    # Apply a second NMS pass using OpenCV
    # to catch any remaining duplicates SAHI missed
    detections = apply_nms(detections, iou_threshold=0.6)

    return detections


def draw_dots_on_image(pil_image, detections):
    """
    Draw red dot for unhealthy, blue dot for healthy trees.
    """
    img     = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    healthy_count   = 0
    unhealthy_count = 0
    dot_radius      = 12
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale      = 0.6
    thickness       = 2

    for i, det in enumerate(detections):
        cx = det["center_x"]
        cy = det["center_y"]

        if det["class"].lower() == "healthy":
            colour = (255, 0, 0)   # blue in BGR
            healthy_count += 1
        else:
            colour = (0, 0, 255)   # red in BGR
            unhealthy_count += 1

        # Draw filled dot
        cv2.circle(img_bgr, (cx, cy), dot_radius, colour, -1)

        # Draw number label next to dot
        cv2.putText(
            img_bgr,
            str(i + 1),
            (cx + 16, cy + 5),
            font, font_scale,
            colour, thickness
        )

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated = Image.fromarray(img_rgb)

    return annotated, healthy_count, unhealthy_count, detections
