import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from NeuralNetworkMain import ConvNN
import os
from collections import deque

def main():
    # Model configuration
    MODEL_PATH = 'acc81.5/best_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load classes/labels based on folder names
    classes = [
        "two of clubs", "two of diamonds", "two of hearts", "two of spades",
        "three of clubs", "three of diamonds", "three of hearts", "three of spades",
        "four of clubs", "four of diamonds", "four of hearts", "four of spades",
        "five of clubs", "five of diamonds", "five of hearts", "five of spades",
        "six of clubs", "six of diamonds", "six of hearts", "six of spades",
        "seven of clubs", "seven of diamonds", "seven of hearts", "seven of spades",
        "eight of clubs", "eight of diamonds", "eight of hearts", "eight of spades",
        "nine of clubs", "nine of diamonds", "nine of hearts", "nine of spades",
        "ten of clubs", "ten of diamonds", "ten of hearts", "ten of spades",
        "jack of clubs", "jack of diamonds", "jack of hearts", "jack of spades",
        "queen of clubs", "queen of diamonds", "queen of hearts", "queen of spades",
        "king of clubs", "king of diamonds", "king of hearts", "king of spades",
        "ace of clubs", "ace of diamonds", "ace of hearts", "ace of spades",
        "joker"
    ]
    
    # Load and prepare model with 53 classes (adjust if needed)
    num_classes = 53 if len(classes) > 53 else len(classes)
    model = ConvNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    # Improved preprocessing pipeline with additional augmentations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight color adjustment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize webcam with higher resolution if possible
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Webcam initialized. Press 'q' to quit.")
    print(f"Ready to recognize {len(classes)} different cards.")
    
    # For temporal smoothing
    prediction_history = deque(maxlen=5)
    confidence_threshold = 40.0  # Only show predictions with confidence > 40%
    
    # Create processing region (center of the frame)
    roi_size = min(frame_width, frame_height) // 2
    roi_x = (frame_width - roi_size) // 2
    roi_y = (frame_height - roi_size) // 2
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Draw region of interest for card placement
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), 
                     (0, 255, 0), 2)
        
        # Extract ROI (where card should be placed)
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        
        # Apply some pre-processing to improve contrast
        roi_processed = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
        
        # Convert to RGB for PIL
        rgb_roi = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_roi)
        
        # Preprocess image for model
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence_score = confidence.item() * 100
            predicted_card = classes[predicted_idx.item() % len(classes)]  # Ensure index is in range
            
            # Add to history for smoothing
            prediction_history.append((predicted_card, confidence_score))
        
        # Apply temporal smoothing
        if len(prediction_history) >= 3:  # Only after collecting enough samples
            # Count occurrences of each prediction
            predictions = {}
            for pred, conf in prediction_history:
                if pred not in predictions:
                    predictions[pred] = {"count": 0, "total_conf": 0}
                predictions[pred]["count"] += 1
                predictions[pred]["total_conf"] += conf
            
            # Find the most frequent prediction with good confidence
            most_frequent = None
            max_count = 0
            for pred, data in predictions.items():
                avg_conf = data["total_conf"] / data["count"]
                if data["count"] > max_count and avg_conf > confidence_threshold:
                    most_frequent = pred
                    max_count = data["count"]
                    smoothed_confidence = avg_conf
            
            if most_frequent:
                predicted_card = most_frequent
                confidence_score = smoothed_confidence
        
        # Enhance display
        # Create a semi-transparent overlay for the main prediction
        overlay = frame.copy()
        
        # Display prediction on frame if confidence is good enough
        if confidence_score > confidence_threshold:
            # Display main prediction with good confidence in green
            cv2.putText(overlay, f"{predicted_card}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(overlay, f"Confidence: {confidence_score:.1f}%", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Low confidence warning in yellow
            cv2.putText(overlay, "Place card in green box", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Show top 3 alternate predictions
        top_k = min(3, len(classes))
        topk_prob, topk_indices = torch.topk(probabilities, top_k)
        
        y_offset = 130
        for i in range(top_k-1):
            idx = topk_indices[0, i+1].item() % len(classes)  # Ensure index is in range
            prob = topk_prob[0, i+1].item() * 100
            
            if prob > 15:  # Only show reasonably confident alternatives
                alt_text = f"Alt #{i+1}: {classes[idx]} ({prob:.1f}%)"
                cv2.putText(overlay, alt_text, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                y_offset += 35
        
        # Add instructions
        cv2.putText(overlay, "Place card in box | 'q' to quit", 
                    (20, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend the overlay with the original frame
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Display frame
        cv2.imshow('Playing Card Recognition', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()