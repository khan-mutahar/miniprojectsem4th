# =========================================================
# ADVANCED REAL-TIME SCENE TEXT DETECTION SYSTEM
# =========================================================
# INSTALL:
#   pip install opencv-python pytesseract numpy imutils
#
# ALSO INSTALL TESSERACT:
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki
#   Linux:   sudo apt install tesseract-ocr
#   Mac:     brew install tesseract
#
# DOWNLOAD EAST MODEL (.pb file, ~90MB):
#   https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.pb
#   OR: https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb
#
# Put frozen_east_text_detection.pb in the SAME folder as this script.
# Put your video file in the same folder and update VIDEO_FILE below.
# =========================================================

import cv2
import pytesseract
import numpy as np
import os
import sys
import time
import platform
from imutils.object_detection import non_max_suppression

# =========================================================
# CONFIG — EDIT THESE
# =========================================================

VIDEO_FILE = "video.mp4"       # ← Change to your actual video filename
CONFIDENCE_THRESHOLD = 0.5     # Raise to reduce false detections (0.0–1.0)
FRAME_SKIP = 3                 # Run OCR every N frames (higher = faster but misses more)
MIN_TEXT_LENGTH = 2            # Ignore text shorter than this

# =========================================================
# TESSERACT PATH (auto-detected, override if needed)
# =========================================================

if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        print("[WARNING] Tesseract not found at default Windows path.")
        print("          Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("          Then update 'tesseract_path' in this script.")
# On Linux/Mac, tesseract is found automatically if installed via apt/brew

# =========================================================
# LOAD EAST MODEL
# =========================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
model_pb = os.path.join(base_dir, "frozen_east_text_detection.pb")

if not os.path.exists(model_pb):
    print("[ERROR] EAST model not found!")
    print(f"        Expected at: {model_pb}")
    print("        Download from:")
    print("        https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb")
    sys.exit(1)

print("[INFO] Loading EAST model...")
net = cv2.dnn.readNet(model_pb)
print("[INFO] EAST model loaded.")

# =========================================================
# OPEN VIDEO
# =========================================================

video_path = os.path.join(base_dir, VIDEO_FILE)

if not os.path.exists(video_path):
    print(f"[ERROR] Video file not found: {video_path}")
    print(f"        Place your video in: {base_dir}")
    print(f"        And update VIDEO_FILE at the top of this script.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    sys.exit(1)

print(f"[INFO] Opened video: {video_path}")

# =========================================================
# VARIABLES
# =========================================================

frame_count = 0
all_detected_text = set()
prev_time = time.time()
last_boxes = []  # Reuse boxes on skipped frames

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# =========================================================
# SAVE FILE
# =========================================================

output_file = open("detected_text.txt", "a", encoding="utf-8")
print("[INFO] Saving detected text to: detected_text.txt")
print("[INFO] Press ESC to quit.\n")

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("[INFO] Video ended.")
        break

    frame_count += 1

    # =====================================================
    # RESIZE FRAME
    # =====================================================

    frame = cv2.resize(frame, (960, 540))
    orig = frame.copy()
    H, W = frame.shape[:2]

    newW, newH = 320, 320
    rW = W / float(newW)
    rH = H / float(newH)

    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    # =====================================================
    # SKIP FRAMES (reuse previous boxes)
    # =====================================================

    if frame_count % FRAME_SKIP != 0:
        boxes = last_boxes
    else:

        # =================================================
        # PREPROCESS FOR EAST
        # =================================================

        resized = cv2.resize(frame, (newW, newH))

        blob = cv2.dnn.blobFromImage(
            resized, 1.0, (newW, newH),
            (123.68, 116.78, 103.94),
            swapRB=True, crop=False
        )

        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # =================================================
        # DECODE BOXES
        # =================================================

        rects = []
        confidences = []

        rows, cols = scores.shape[2], scores.shape[3]

        for y in range(rows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(cols):
                score = scoresData[x]
                if score < CONFIDENCE_THRESHOLD:
                    continue

                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(float(score))

        # =================================================
        # NON-MAX SUPPRESSION
        # =================================================

        if len(rects) > 0:
            boxes = non_max_suppression(
                np.array(rects), probs=confidences
            )
        else:
            boxes = []

        last_boxes = boxes

    # =====================================================
    # PROCESS BOXES — OCR
    # =====================================================

    detected_this_frame = []

    for (startX, startY, endX, endY) in boxes:

        # Scale back to original frame size
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Add padding
        padding = 5
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(W, endX + padding)
        endY = min(H, endY + padding)

        roi = orig[startY:endY, startX:endX]

        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            continue

        # =================================================
        # PREPROCESSING FOR BETTER OCR
        # =================================================

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # =================================================
        # TESSERACT OCR
        # =================================================

        config = r'--oem 3 --psm 7'
        text = pytesseract.image_to_string(thresh, config=config).strip()

        # =================================================
        # FILTER WEAK RESULTS
        # =================================================

        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable())
        text = text.strip()

        if len(text) < MIN_TEXT_LENGTH:
            continue

        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in text):
            continue

        # Save new unique text
        if text not in all_detected_text:
            all_detected_text.add(text)
            output_file.write(text + "\n")
            output_file.flush()
            print(f"[TEXT] {text}")

        detected_this_frame.append((startX, startY, endX, endY, text))

    # =====================================================
    # DRAW BOUNDING BOXES
    # =====================================================

    for (startX, startY, endX, endY, text) in detected_this_frame:

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        label = text[:30]  # Truncate long text for display
        cv2.putText(
            orig, label,
            (startX, max(startY - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2
        )

    # =====================================================
    # FPS OVERLAY
    # =====================================================

    cv2.putText(
        orig, f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 0, 0), 2
    )

    # =====================================================
    # SIDE PANEL
    # =====================================================

    panel = np.zeros((540, 350, 3), dtype=np.uint8)

    cv2.putText(
        panel, "Detected Text",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 255), 2
    )

    y_pos = 70
    recent_texts = list(all_detected_text)[-15:]

    for txt in recent_texts:
        display = txt[:28]  # Prevent overflow
        cv2.putText(
            panel, display,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 1
        )
        y_pos += 28
        if y_pos > 520:
            break

    # =====================================================
    # COMBINE & SHOW
    # =====================================================

    combined = np.hstack((orig, panel))
    cv2.imshow("Advanced Scene Text Detection", combined)

    if cv2.waitKey(1) == 27:  # ESC to quit
        print("[INFO] Exiting...")
        break

# =========================================================
# CLEANUP
# =========================================================

output_file.close()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done. Detected text saved to detected_text.txt")