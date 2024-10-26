from ultralytics import YOLO
import cv2
import numpy as np

# Load a larger, high-accuracy YOLOv8 model (adjust the model path as needed)
model = YOLO("best.pt")  # Replace with your fine-tuned model path
class_names = model.names

# Open the camera feed
cap = cv2.VideoCapture('10.mp4')  # Use 1 if an external camera is connected

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Resize for consistent model input
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # Run detection with a higher confidence threshold for accuracy
    results = model.predict(img, conf=0.5)  # Set confidence threshold

    for r in results:
        boxes = r.boxes
        masks = r.masks
        
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                 
    cv2.imshow('Pothole Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
