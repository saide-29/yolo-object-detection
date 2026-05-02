from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # Mac kamerası

while True:
    ret, frame = cap.read()
    if not ret:
        print("kamera açılamadı")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Camera", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
