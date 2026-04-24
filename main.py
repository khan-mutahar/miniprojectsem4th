import cv2
import pytesseract

# Set Tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture('vedio.mp4')

frame_count = 0
prev_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize (increase size for better OCR)
    frame = cv2.resize(frame, None, fx=1.2, fy=1.2)

    display_frame = frame.copy()

    # Skip frames for stability
    if frame_count % 20 != 0:
        # Draw previous boxes (reduces jitter)
        for (x, y, w, h, text) in prev_boxes:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display_frame, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 2)

        cv2.imshow("Text Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # -------- Preprocessing --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur for noise reduction
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # OTSU threshold (more stable than adaptive)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # -------- OCR --------
    config = '--oem 3 --psm 6'

    data = pytesseract.image_to_data(
        thresh,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    current_boxes = []

    for i in range(len(data['text'])):
        try:
            conf = int(float(data['conf'][i]))
        except:
            continue

        text = data['text'][i].strip()

        # Higher confidence for stability
        if conf > 75 and text != "":
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            current_boxes.append((x, y, w, h, text))

    # -------- Jitter Fix --------
    if len(current_boxes) == 0:
        current_boxes = prev_boxes
    else:
        prev_boxes = current_boxes

    # -------- Draw Boxes --------
    for (x, y, w, h, text) in current_boxes:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(display_frame, text, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2)

    cv2.imshow("Text Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()