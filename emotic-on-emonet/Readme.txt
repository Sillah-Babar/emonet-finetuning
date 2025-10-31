EmoNet: Multi-label Emotion Recognition on EMOTIC Dataset
A PyTorch implementation of EmoNet with architectural variations for multi-label emotion recognition, trained on the EMOTIC dataset. This model predicts 26 discrete emotions along with continuous valence and arousal values.
Overview
This repository provides:

Face visibility computation using MediaPipe Face Mesh
Data preprocessing with normalized valence/arousal values
Multi-label emotion classification with continuous affect prediction
Comprehensive evaluation and visualization tools
Pre-trained model weights

Dataset
This project uses the EMOTIC dataset available on Kaggle.
The dataset includes:

26 discrete emotion categories
Continuous valence and arousal annotations
Context and body images with facial crops

Installation
Requirements
Create a virtual environment and install dependencies:
bashCopypip install -r requirements.txt
Required Libraries
Copytorch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pandas>=1.3.0
opencv-python>=4.5.0
mediapipe>=0.8.0
Pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.62.0
Preprocessing Pipeline
Step 1: Add Face Visibility Column
Use face_visibility.py to compute face visibility scores for all facial crops:
bashCopypython face_visibility.py
This script:

Uses MediaPipe Face Mesh to detect facial landmarks
Computes visibility percentage (0-100%) based on detected landmarks
Adds Face_Visibility column to the annotation CSV
Generates distribution histogram

Input:

Original annotation CSV (e.g., annot_arrs_extra_train.csv)
Facial crop numpy arrays (.npy files)

Output:

Updated CSV with Face_Visibility column
Visualization of face visibility distribution

Step 2: Normalize Valence and Arousal
Use preprocessing.py to normalize continuous affect values:
bashCopypython preprocessing.py
This script:

Loads raw valence and arousal values
Applies min-max normalization to [-1, 1] range
Creates normalized dataset CSV files

Configuration:
Update paths in preprocessing.py:
pythonCopydata_path = Path('/path/to/emotic/archive_emot/annots_arrs')
img_dir = Path('/path/to/emotic/archive_emot/img_arrs/')
Output:

normalized_dataset_train.csv
normalized_dataset_val.csv
normalized_dataset_test.csv

Model Training
The model architecture and training procedure are provided in emonet-training-from-scratch.ipynb.
Architecture Highlights

Backbone: ResNet-50 (pre-trained on ImageNet)
Dual-head output:

Classification head: 26 discrete emotions (sigmoid activation)
Regression head: Valence and Arousal (tanh activation)


Loss functions:

Asymmetric Loss for multi-label classification
Smooth L1 Loss for regression
Combined weighted loss



Training Configuration
Key hyperparameters:
pythonCopybatch_size = 32
learning_rate = 1e-4
epochs = 50
optimizer = AdamW
scheduler = ReduceLROnPlateau
Data augmentation:

Random horizontal flip
Color jitter
Random rotation
Normalization (ImageNet statistics)

Model Weights
Download pre-trained weights from Google Drive:
Model Checkpoint
Place the downloaded best_model.pth in your working directory.
Inference and Evaluation
Running Predictions
Use the provided notebook or run inference directly:
pythonCopyfrom inference_visualization import review_predictions_with_visualization

results = review_predictions_with_visualization(
    model_checkpoint_path='/path/to/best_model.pth',
    test_csv_path='/path/to/normalized_dataset_test.csv',
    img_arrays_dir='/path/to/img_arrs',
    output_dir='./prediction_review',
    num_samples=20,
    detailed_samples=20,
    sort_by='random'  # Options: 'random', 'best', 'worst', 'confidence'
)
Evaluation Metrics
The evaluation pipeline computes:

Per-emotion precision, recall, F1-score
Overall classification metrics
Mean Absolute Error (MAE) for valence/arousal
Confusion matrices and co-occurrence analysis

Visualization Outputs
Generated visualizations include:

Grid View: Multiple predictions with ground truth comparison
Detailed Analysis: Individual sample breakdown with:

Original image
Emotion prediction bars vs ground truth
Valence-Arousal space plot
Classification metrics summary


Confusion Heatmap: Emotion co-occurrence matrix
