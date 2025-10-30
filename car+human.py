from ultralytics import YOLO      # YOLOv8 model from Ultralytics
import cv2                        # OpenCV for video capture and drawing
import time                       # Time module for FPS calculation

# Load YOLOv8 nano model (fastest version)
model = YOLO("yolov8n.pt")

# Move model to GPU (RTX 2050) for faster inference
model.to('cuda')  # If CUDA is available, this uses your GPU

# Open webcam or IP camera stream
cap = cv2.VideoCapture("The Fate of the Furious ｜ Harpooning Dom's Car.mp4")  # Replace with 0 for local webcam

while True:
    start_time = time.time()  # Start timer for FPS calculation

    ret, frame = cap.read()   # Read a frame from the video stream
    if not ret:
        break  # Exit loop if no frame is received

    # Resize frame to medium resolution for faster processing
    frame = cv2.resize(frame, (640, 400))

    # Run YOLOv8 detection on the frame (on GPU)
    results = model(frame)

    # Extract bounding box data from results
    detections = results[0].boxes.data

    # Initialize counters
    person_count = 0
    vehicle_count = 0

    # Define vehicle classes to detect
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    # Loop through each detected object
    for box in detections:
        # Extract box coordinates, confidence score, and class ID
        x1, y1, x2, y2, score, cls_id = box[:6]
        cls_id = int(cls_id)
        label_name = model.names[cls_id]

        # Convert coordinates to integers for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Check if it's a person
        if label_name == 'person':
            person_count += 1
            color = (0, 255, 0)  # Green for person
            label = f"Person {person_count} ({score:.2f})"

        # Check if it's a vehicle
        elif label_name in vehicle_classes:
            vehicle_count += 1
            color = (0, 0, 255)  # Red for vehicles
            label = f"{label_name.capitalize()} {vehicle_count} ({score:.2f})"

        else:
            continue  # Skip other classes

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Calculate FPS based on time taken for one loop
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display counts and FPS
    cv2.putText(frame, f"People Count: {person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Show the final frame with annotations
    cv2.imshow("ESP Detection", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Collect detected labels
    detected_labels = []

    for box in detections:
        cls_id = int(box[5])
        label_name = model.names[cls_id]

        # Save only relevant labels (e.g., person, car, etc.)
        if label_name == 'person' or label_name in vehicle_classes:
            detected_labels.append(label_name)

    # Save detected labels to a file
    with open("detected_objects.txt", "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        label_line = ", ".join(detected_labels)
        f.write(f"{timestamp}: {label_line}\n")
# Release video stream and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

