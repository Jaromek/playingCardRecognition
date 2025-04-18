# Playing Card Recognition

This project implements a deep learning pipeline for recognizing playing cards using convolutional neural networks (CNNs) in PyTorch. It includes scripts for training, evaluating, and testing models, as well as tools for live camera-based recognition.

## Project Structure

cameraRecognition.py # Live camera card recognition script modelTesting.ipynb # Jupyter notebook for model evaluation and visualization NeuralNetworkMain.py # Main neural network architecture, training, and utility functions transform.py # Data augmentation and transformation utilities acc81.5 # Directory with trained models, results, and visualizations dataset/ # Dataset directory (train/valid/test splits) README.md # Project documentation pycache/ # Python cache files


## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- OpenCV (for camera recognition)

Install dependencies with:

```sh
pip install torch torchvision scikit-learn matplotlib seaborn opencv-python
```

Dataset
Place your dataset in the dataset/ directory with the following structure:

dataset/
    train/
    valid/
    test/

    Each subdirectory should contain one folder per card class, with images inside.

Training
Train the neural network using the functions in NeuralNetworkMain.py.

Model Evaluation
Use modelTesting.ipynb to:

Load a trained model (acc81.5/best_model.pth)
Evaluate accuracy on the dataset
Generate and visualize confusion matrices for color, suit, rank, and all classes
Live Recognition
Run cameraRecognition.py to recognize cards in real-time using your webcam.

Results
Trained models, accuracy plots, and confusion matrices are saved in the acc81.5/ directory.

License
This project is for educational and research purposes.

