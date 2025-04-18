import cv2
import torch
from PIL import Image
from neuralNetwork import ConvNN, datasets, transforms
from collections import deque

def main():
    DIR_PATH = 'dataset'
    TEST_PATH = f'{DIR_PATH}/test'
    BATCH_SIZE = 32
    MODEL_PATH = 'acc81.5/best_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=val_test_transform)
    classes = test_dataset.classes


    
    num_classes = 53 if len(classes) > 53 else len(classes)
    model = ConvNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Webcam initialized. Press 'q' to quit.")
    print(f"Ready to recognize {len(classes)} different cards.")
    
    prediction_history = deque(maxlen=5)
    confidence_threshold = 40.0  
    
    roi_size = min(frame_width, frame_height) // 2
    roi_x = (frame_width - roi_size) // 2
    roi_y = (frame_height - roi_size) // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), 
                     (0, 255, 0), 2)
        
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        
        roi_processed = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)

        rgb_roi = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_roi)

        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence_score = confidence.item() * 100
            predicted_card = classes[predicted_idx.item() % len(classes)]  # Ensure index is in range

            prediction_history.append((predicted_card, confidence_score))

        if len(prediction_history) >= 3:
            predictions = {}
            for pred, conf in prediction_history:
                if pred not in predictions:
                    predictions[pred] = {"count": 0, "total_conf": 0}
                predictions[pred]["count"] += 1
                predictions[pred]["total_conf"] += conf

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

        overlay = frame.copy()

        if confidence_score > confidence_threshold:
            cv2.putText(overlay, f"{predicted_card}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(overlay, f"Confidence: {confidence_score:.1f}%", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "Place card in green box", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        top_k = min(3, len(classes))
        topk_prob, topk_indices = torch.topk(probabilities, top_k)
        
        y_offset = 130
        for i in range(top_k-1):
            idx = topk_indices[0, i+1].item() % len(classes)
            prob = topk_prob[0, i+1].item() * 100
            
            if prob > 15: 
                alt_text = f"Alt #{i+1}: {classes[idx]} ({prob:.1f}%)"
                cv2.putText(overlay, alt_text, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                y_offset += 35

        cv2.putText(overlay, "Place card in box | 'q' to quit", 
                    (20, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        alpha = 0.7  
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.imshow('Playing Card Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()