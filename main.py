import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture('vedio.mp4')

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames
    if frame_count % 10 != 0:
        cv2.imshow("Text Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    config = '--oem 3 --psm 6'

    data = pytesseract.image_to_data(
        thresh,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    for i in range(len(data['text'])):
        try:
            conf = int(float(data['conf'][i]))
        except:
            continue

        text = data['text'][i].strip()

        if conf > 60 and text != "":
            x, y, w, h = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 2)

    cv2.imshow("Text Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()