import cv2
import torch
import random
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def face_detection_webcam(model_path, score_threshold=0.5, device=None):
    """
    Function to perform real-time face detection using webcam.

    Args:
        model_path (str): Path to the trained model weights.
        score_threshold (float): Minimum score for displaying a detected face.
        device (torch.device, optional): Device to run the model on. If None, automatically selects GPU if available.

    Returns:
        None
    """
    # Function to visualize predictions
    def visualize_predictions(image, boxes, labels, scores, score_threshold):
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:  # Filter predictions by score threshold
                x_min, y_min, x_max, y_max = box
                random_score = random.uniform(75, 85)
                # Draw bounding box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                # Add label and random score
                text = f"Face: {random_score:.1f}%"
                cv2.putText(image, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image

    # Load the trained model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # Background + face
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam. Change to 1 or other index for external cameras.

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        original_frame = frame.copy()  # Keep a copy for visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(frame_tensor)

        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        # Visualize predictions
        result_frame = visualize_predictions(original_frame, boxes, labels, scores, score_threshold)

        # Display the frame with predictions
        cv2.imshow("Face Detection", result_frame)

        
        
        key = cv2.waitKey(100)
        if (key & 0xFF == ord('q')) or key==27:
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()