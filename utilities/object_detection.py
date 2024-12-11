from ultralytics import YOLO
import cv2
import cvzone
import math

def objects_detection():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    # Resolution set ,for better visibility and consistent performance
    cap.set(3, 1280)
    cap.set(4, 720)
    model = YOLO('../assets/yolov8x.pt')

    # Define class names
    classNames = ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train","truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                  "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                  "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
                  "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    # Use target classes if provided, else default to some classes
    target_classes = ["person", "backpack", "bottle", "chair", "cell phone", "book", "clock"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for result in results:
            boxes = result.boxes #Get detected bounding boxes
            for box in boxes:
                #Extract the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width, height = x2 - x1, y2 - y1
                cls = box.cls[0]
                name = classNames[int(cls)]
                # Check if the detected obj is in the target classes
                if name in target_classes:
                    #Draw rectangle 
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    #Calculate the confidence level 
                    confidence = math.ceil((box.conf[0] * 100))
                    cvzone.putTextRect(img, f'{name} {confidence} %', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("Image", img)

        # Check if the window is closed (if exit button is clicked)
        key = cv2.waitKey(100)
        if key:
            if key == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
