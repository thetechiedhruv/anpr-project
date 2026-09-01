# ===========================
# Environment & Logging Setup
# ===========================

import os
import logging
import warnings

# Suppress Paddle & System Logs
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_log_level"] = "3"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_enable_pir"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ===========================
# Imports
# ===========================

import re
import cv2
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ===========================
# Model Paths
# ===========================

PLATE_MODEL_PATH = "models/license-plate-finetune-v1x.pt"
VEHICLE_MODEL_PATH = "yolo12m.pt"

INPUT_VIDEO = "videos/clip_1.mp4"
base_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
OUTPUT_VIDEO = f"output/{base_name}_output.mp4"



# ===========================
# Initialize Models
# ===========================

# License Plate Detector
plate_model = YOLO(PLATE_MODEL_PATH)

# Vehicle Detector
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

# OCR Engine
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    show_log=False
)


# ===========================
# Constants
# ===========================

# Vehicle class IDs (COCO format)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, bus, truck, motorcycle


# ===========================
# Utility Functions
# ===========================

def clean_plate_text(text: str) -> str:
    """
    Clean and validate OCR output.
    """

    if not text:
        return ""

    # Convert to uppercase
    text = text.upper()

    # Keep only letters & numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Must contain letters & digits
    if not re.search(r"[A-Z]", text):
        return ""

    if not re.search(r"[0-9]", text):
        return ""

    # Length validation
    if len(text) < 6 or len(text) > 12:
        return ""

    return text


# ===========================
# OCR Function
# ===========================

def perform_ocr(plate_img: np.ndarray):
    """
    Perform OCR on cropped license plate image.
    """

    raw_text = ""
    best_conf = 0.35

    results = ocr.ocr(plate_img)

    if not results:
        return "", best_conf

    for line in results:

        if line is None:
            continue

        for word_info in line:

            if word_info is None or len(word_info) < 2:
                continue

            text, conf = word_info[1]

            raw_text += text
            best_conf = max(best_conf, conf)

    return raw_text, best_conf


# ===========================
# Plate Detection
# ===========================

def detect_plates(img: np.ndarray):
    """
    Detect license plates and extract text.

    Returns:
        List of detected plates with bbox and confidence.
    """

    detections = []

    results = plate_model(
        img,
        imgsz=1280,
        conf=0.35,
        iou=0.5,
        verbose=False
    )

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            plate_crop = img[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # OCR
            raw_text, conf = perform_ocr(plate_crop)

            # Clean text
            plate_text = clean_plate_text(raw_text)

            # Draw plate box
            draw_plate_box(img, x1, y1, x2, y2, plate_text)

            detections.append({
                "plate": plate_text,
                "box": (x1, y1, x2, y2),
                "confidence": conf
            })

    return detections


# ===========================
# Drawing Functions
# ===========================

def draw_plate_box(img, x1, y1, x2, y2, text):
    """
    Draw bounding box and text for plate.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Text size
    (w, h), b = cv2.getTextSize(text, font, scale, thickness)

    # Background
    cv2.rectangle(
        img,
        (x1, y1 - h - 12),
        (x1 + w, y1),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        img,
        text,
        (x1, y1 - 5),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_vehicle_box(img, x1, y1, x2, y2, label):
    """
    Draw vehicle bounding box.
    """

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )


# ===========================
# Video Processing
# ===========================

def process_video():

    cap = cv2.VideoCapture(INPUT_VIDEO)

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    print("Processing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Detect vehicles
        results = vehicle_model(
            frame,
            imgsz=1920,
            conf=0.3,
            iou=0.5,
            verbose=False
        )

        for box in results[0].boxes:

            cls_id = int(box.cls)

            if cls_id not in VEHICLE_CLASSES:
                continue

            label = vehicle_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            vehicle_roi = frame[y1:y2, x1:x2]

            if vehicle_roi.size == 0:
                continue

            # Detect plates inside vehicle
            plates = detect_plates(vehicle_roi)

            # Draw vehicle
            draw_vehicle_box(frame, x1, y1, x2, y2, label)

            # Print detected plates
            for plate in plates:
                print("Detected Plate:", plate["plate"])

        out.write(frame)

    cap.release()
    out.release()

    print("Processing completed!")
    op_base_name = os.path.splitext(os.path.basename(OUTPUT_VIDEO))[0]
    print("Saved to:", op_base_name)


# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":

    process_video()
