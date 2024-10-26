import cv2
from ultralytics import YOLO

# Load model and initialize video capture
model = YOLO("best.pt")
cap = cv2.VideoCapture('testing1.mp4')

# Initialize data storage for potholes
pothole_data = {}
next_id = 0  # Unique ID for each detected pothole
trackers = []  # List to hold trackers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency (optional)
    frame = cv2.resize(frame, (1020, 500))

    # Use YOLO model to detect potholes in the current frame
    results = model.predict(frame)
    detections = []  # Store detections in this frame

    # Retrieve bounding boxes for detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf
            if conf > 0.5:  # Confidence threshold
                detections.append((x1, y1, x2, y2))

    # Update trackers and remove ones that have lost the object
    updated_trackers = []
    for tracker, pothole_id in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            center_x, center_y = x + w // 2, y + h // 2
            pothole_data[pothole_id]['coordinates'].append((center_x, center_y))
            pothole_data[pothole_id]['detected_frames'].append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            updated_trackers.append((tracker, pothole_id))

    # Replace trackers with updated list
    trackers = updated_trackers

    # Add new detections as new trackers
    for (x1, y1, x2, y2) in detections:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        pothole_data[next_id] = {'coordinates': [(x1 + x2) // 2, (y1 + y2) // 2], 'detected_frames': [cap.get(cv2.CAP_PROP_POS_FRAMES)]}
        trackers.append((tracker, next_id))
        next_id += 1

    # Display the frame
    cv2.imshow('Pothole Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Output the pothole data for reference
for pothole_id, data in pothole_data.items():
    print(f"Pothole ID: {pothole_id}, Coordinates: {data['coordinates']}, Frames: {data['detected_frames']}")
