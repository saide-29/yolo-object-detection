from ultralytics import YOLO
import cv2
import time
import numpy as np



# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

# OPEN CAMERA
cap = cv2.VideoCapture(0)

# PREVIOUS LEFT LANE VALUES
prev_left_x1 = 0
prev_left_y1 = 0
prev_left_x2 = 0
prev_left_y2 = 0

# PREVIOUS RIGHT LANE VALUES
prev_right_x1 = 0
prev_right_y1 = 0
prev_right_x2 = 0
prev_right_y2 = 0

while True:

    # START FPS TIMER
    start_time = time.time()

    # READ CAMERA FRAME
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO OBJECT DETECTION + BYTE TRACKING
    results = model.track(
        frame,
        tracker="bytetrack.yaml"
    )[0]

    # FILTER ONLY CAR + PERSON
    filtered_boxes = []

    for box in results.boxes:

        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["car", "person"]:
            filtered_boxes.append(box)


    results.boxes = filtered_boxes

    # DRAW YOLO BOXES
    annotated_frame = results.plot()

    # COLLISION WARNING SYSTEM
    frame_height = frame.shape[0]

    for box in filtered_boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        height = y2 - y1

        # Object too close
        if height > frame_height * 0.35:

            cv2.putText(
                annotated_frame,
                "WARNING",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3
            )

    # LANE DETECTION PREPROCESSING
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    # CREATE ROI MASK
    height = edges.shape[0]

    mask = np.zeros_like(edges)

    # Only lower half of image
    mask[height // 2:, :] = 255

    roi = cv2.bitwise_and(edges, mask)

    # HOUGH LINE DETECTION
   
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        50,
        minLineLength=40,
        maxLineGap=100
    )
    # LEFT / RIGHT LANE CLASSIFICATION
    
    left_line = []
    right_line = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # avoid division by zero
            if x2 - x1 == 0:
                continue

            # line slope
            slope = (y2 - y1) / (x2 - x1)

            # ignore horizontal lines
            if abs(slope) < 0.5:
                continue

            # left lane
            if slope < 0:
                left_line.append(line)

            # right lane
            else:
                right_line.append(line)

        left_x1 = []
        left_y1 = []
        left_x2 = []
        left_y2 = []

        for line in left_line:

            x1, y1, x2, y2 = line[0]

            left_x1.append(x1)
            left_y1.append(y1)
            left_x2.append(x2)
            left_y2.append(y2)

        right_x1 = []
        right_y1 = []
        right_x2 = []
        right_y2 = []

        for line in right_line:

            x1, y1, x2, y2 = line[0]

            right_x1.append(x1)
            right_y1.append(y1)
            right_x2.append(x2)
            right_y2.append(y2)


        if len(left_x1) > 0 and len(right_x1) > 0:
            
            # LANE AVERAGING
            
            avg_x1l = int(sum(left_x1) / len(left_x1))
            avg_y1l = int(sum(left_y1) / len(left_y1))
            avg_x2l = int(sum(left_x2) / len(left_x2))
            avg_y2l = int(sum(left_y2) / len(left_y2))

            avg_x1r = int(sum(right_x1) / len(right_x1))
            avg_y1r = int(sum(right_y1) / len(right_y1))
            avg_x2r = int(sum(right_x2) / len(right_x2))
            avg_y2r = int(sum(right_y2) / len(right_y2))


            # TEMPORAL SMOOTHING
           
            avg_x1l = int(0.8 * prev_left_x1 + 0.2 * avg_x1l)
            avg_y1l = int(0.8 * prev_left_y1 + 0.2 * avg_y1l)
            avg_x2l = int(0.8 * prev_left_x2 + 0.2 * avg_x2l)
            avg_y2l = int(0.8 * prev_left_y2 + 0.2 * avg_y2l)

            avg_x1r = int(0.8 * prev_right_x1 + 0.2 * avg_x1r)
            avg_y1r = int(0.8 * prev_right_y1 + 0.2 * avg_y1r)
            avg_x2r = int(0.8 * prev_right_x2 + 0.2 * avg_x2r)
            avg_y2r = int(0.8 * prev_right_y2 + 0.2 * avg_y2r)


            # update previous values
            prev_left_x1 = avg_x1l
            prev_left_y1 = avg_y1l
            prev_left_x2 = avg_x2l
            prev_left_y2 = avg_y2l

            prev_right_x1 = avg_x1r
            prev_right_y1 = avg_y1r
            prev_right_x2 = avg_x2r
            prev_right_y2 = avg_y2r

            # LANE AREA POLYGON
            points = np.array([
                [avg_x1l, avg_y1l],
                [avg_x2l, avg_y2l],
                [avg_x2r, avg_y2r],
                [avg_x1r, avg_y1r]
            ])

            cv2.fillPoly(
                annotated_frame,
                [points],
                (0, 255, 0)
            )


            # DRAW LANE LINES
            
            cv2.line(
                annotated_frame,
                (avg_x1l, avg_y1l),
                (avg_x2l, avg_y2l),
                (255, 0, 0),
                5
            )

            cv2.line(
                annotated_frame,
                (avg_x1r, avg_y1r),
                (avg_x2r, avg_y2r),
                (0, 255, 0),
                5
            )

            # LANE DEPARTURE WARNING
           
            lane_center = (avg_x1l + avg_x1r) // 2

            frame_center = frame.shape[1] // 2

            if abs(lane_center - frame_center) > 10:

                cv2.putText(
                    annotated_frame,
                    "WARNING: Lane Departure",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )


    # FPS CALCULATION
   
    end_time = time.time()

    processing_time = end_time - start_time

    fps = 1 / processing_time


    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow("Mini_ADAS_System", annotated_frame)


    # Exit with ESC
    if cv2.waitKey(1) == 27:
        break

# RELEASE RESOURCES
cap.release()

cv2.destroyAllWindows()