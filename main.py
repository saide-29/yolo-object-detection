from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    height = edges.shape[0]
    mask = np.zeros_like(edges)
    mask[height//2:, :] = 255
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100) #return cordinats x1 y1 x2 y2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # [[]] -> []
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 5)




   



    start_time = time.time()

    results = model.track(frame , tracker="bytetrack.yaml")[0]    #the model works and memory

   

    filtered_boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["car", "person"]:
            filtered_boxes.append(box)
        
        

    results.boxes = filtered_boxes

    annotated_frame = results.plot()

    
    frame_height = frame.shape[0] #shape (H,W,R)

    for box in filtered_boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        height = y2 - y1

        if height > frame_height * 0.35:
        
            cv2.putText(annotated_frame,
                "WARNING",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3)

    ends_time = time.time()

    proccessing_time = ends_time - start_time

    fps = 1/proccessing_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

    cv2.imshow("Lane_Detection", frame)
    #cv2.imshow("Roi", roi)
    #cv2.imshow("Edges", edges)
    #cv2.imshow("Filtered Detection", annotated_frame)
    #cv2.imshow("gray_frame", gray) gray format


    
    

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()