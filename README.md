# Brain Tumor Detection Using AI

This repository contains a simple AI-based approach for brain tumor detection using image histograms. The model is trained on grayscale histograms extracted from medical images and predicts whether an image contains a tumor or not based on Euclidean distance comparison.

## 📌 Features
- **Preprocessing**: Converts images to grayscale and splits them into regions.
- **Feature Extraction**: Generates histograms for each region.
- **Training**: Computes optimal histograms for tumor and non-tumor images.
- **Testing**: Classifies new images based on distance comparison with trained histograms.
- **Accuracy Calculation**: Evaluates model performance on a test dataset.

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MohaYass92/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Dataset Download
Download the dataset from the following link and place it in the `dataset/` folder:
📂 **[[Dataset Link](https://drive.google.com/file/d/18kPRNhMXLVhGVWocH33hzHQH9kGzaDYt/view?usp=sharing)](#)** 

### 4️⃣ Training the Model
Run the training script to process the dataset and generate trained histograms:
```bash
python train.py
```
The script will ask for:
- Number of rows and columns to split images
- Number of bins for histograms
- Paths to tumor and non-tumor image folders

### 5️⃣ Testing the Model
Run the testing script to evaluate the model:
```bash
python test.py
```
It will ask for:
- The test dataset path
- Number of rows, columns, and bins used in training

The accuracy will be disp
