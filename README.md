# 🃏 Playing Card Recognition

A deep learning project for playing card recognition using Convolutional Neural Networks (CNNs) in PyTorch. Includes scripts for training, evaluation, testing, and real-time camera-based recognition.

---

## 📁 Project Structure

```
├── cameraRecognition.py       # Real-time card recognition via webcam
├── modelTesting.ipynb         # Model evaluation and visualization notebook
├── NeuralNetworkMain.py       # Neural network architecture, training, and utilities
├── transform.py               # Data augmentation and transformation helpers
├── acc81.5/                   # Trained models, results, plots, and confusion matrices
├── dataset/                   # Dataset (train/valid/test splits)
├── requirements.txt           # Liblaries and dependencies for python
└── README.md                  # Project documentation
```

---

## 🛠️ Requirements

- Python = 3.12.9  
- PyTorch  
- torchvision  
- scikit-learn  
- matplotlib  
- seaborn  
- OpenCV
- tqdm
- pillow

Install all dependencies with:

```
pip install -r requirements.txt
```

---

## 🗂️ Dataset Structure

```
dataset/
├── train/
├── valid/
└── test/
```

Each subfolder should contain one folder per card class with images inside.

---

## 🧠 Model Training

Train your neural network using the functions in `NeuralNetworkMain.py`.

---

## 📊 Model Evaluation

Use the `modelTesting.ipynb` notebook to:

- Load the best model (`acc81.5/best_model.pth`)
- Evaluate accuracy on the test set
- Generate and visualize confusion matrices for:
  - Color
  - Rank
  - Suit
  - Full classes

---

## 🎥 Real-Time Recognition

Run the following command:

```
python cameraRecognition.py
```

The script will activate your webcam and start recognizing visible cards in real time.

---

## 🖼️ Results (Examples)

### 🔍 Live Recognition Sample:
<p align="center">
  <img src="acc81.5/liveRecognition/EOD98.5.png" width="400" alt="Live Demo – EOD 98.5%"/>
  <img src="acc81.5/liveRecognition/KOD95.3.png" width="400" alt="Live Demo – KOD 95.3%"/>
</p>
<p align="center">
  <img src="acc81.5/liveRecognition/SOH75.8.png" width="400" alt="Live Demo – SOH75.8.png"/>
  <img src="acc81.5/liveRecognition/AOS88.png" width="400" alt="Live Demo – AOS88.png"/>
</p>



### 📈 Learning Curve:
<p align="center">
  <img src="acc81.5/modelAccTrainingNNdokladnoscepoka.png" width="800" alt="Learning Curve"/>
</p>

### ♠️♥️♦️♣️ Confusion Matrices – Rank / Suit:
<p align="center">
  <img src="acc81.5/confusionMatrix/englishFigureCM.png" width="800" alt="confusion-rank"/>
</p>
<p align="center">
  <img src="acc81.5/confusionMatrix/englishCardColorCM.png" width="800" alt="confusion-suit"/>
</p>


---

## 📦 Contents of `acc81.5/`

- `best_model.pth` — best performing trained model
- Training/validation loss and accuracy plots
- Confusion matrices for various categories

---

## 📄 License

This project is intended for educational and research purposes only.
