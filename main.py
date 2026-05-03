from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model(frame)[0]    #the model works

   

    filtered_boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["car", "person"]:
            filtered_boxes.append(box)

    results.boxes = filtered_boxes

    annotated_frame = results.plot()

    ends_time = time.time()

    proccessing_time = ends_time - start_time

    fps = 1/proccessing_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

    cv2.imshow("Filtered Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()