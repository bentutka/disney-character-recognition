# Disney Animated Character Recognition

### Team Members  
Kaitlyn Le, Anh Tran, Benjamin Tutka  
University of Arkansas — Group 9

---

## Overview

This project explores **automated recognition of Disney animated characters** using computer vision and deep learning.  
Our goal is to build a model that identifies characters from movie scenes — regardless of art style, setting, or outfit variation. The motivation comes from real research efforts at Disney’s DTCI team, who found that traditional methods such as **HOG + SVM** performed well for live-action characters but struggled with stylized, non-human designs.  

Our models aim to bridge that gap by comparing traditional convolutional networks against a **Siamese CNN** designed for one-shot similarity learning.

---

## Dataset

Each team member collected roughly 25 images for 4 distinct Disney characters, resulting in **≈ 300 images total** across 12 unique characters.  
Key details:

- Images taken from various movie scenes or video frames  
- Each frame contains **one visible character face**  
- Converted to PNG format  
- Resized to **28×28 pixels**, maintaining aspect ratio with padding  
- Standardized filenames: `charactername1.png`, `charactername2.png`, etc.  

The dataset was structured by character folder to simplify training and evaluation.

---

## Methods

### 1. Simple CNN (Baseline)

A traditional convolutional neural network trained to classify images directly.  
- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Cross Entropy  
- **Batch Size:** 32  
- **Epochs:** 40  

**Results:**  
- Train Accuracy = 0.99 Precision = 0.99 Recall = 0.99  
- Test Accuracy = 0.65 Precision = 0.70 Recall = 0.65  

---

### 2. Siamese CNN (Similarity Learning)

A pairwise model that learns whether two input images depict the same character.  
- **Optimizer:** Adam  
- **Loss:** Binary Cross Entropy  
- **Batch Size:** 64  
- **Epochs:** 100  

During inference, the Siamese network produces a **similarity score** between a test image and one reference image per class. The class with the highest similarity score is chosen as the prediction.

**Results:**  
- Train Accuracy = 0.86 Precision = 0.86 Recall = 0.85  
- Test Accuracy = 0.68 Precision = 0.71 Recall = 0.67  

Although the Siamese CNN generalized well conceptually, some **overfitting** was observed due to limited training data.

---

## Evaluation Metrics

- **Accuracy:** Proportion of total correct predictions  
- **Precision:** Fraction of positive predictions that were correct  
- **Recall:** Fraction of actual positives correctly identified  

Because our dataset was balanced, these metrics provided a clear picture of model performance.

---

## Key Findings

- The **Simple CNN** slightly outperformed the **Siamese CNN** for direct classification.  
- **Siamese CNNs** are better suited for similarity or verification tasks rather than strict multi-class labeling.  
- Performance was likely constrained by the small dataset and lack of fine-grained feature diversity.  
- Proper **image preprocessing and standardization** were critical to ensure reproducibility.  

---

## Next Steps

- Implement **data augmentation** to expand training variability.  
- Evaluate Siamese similarity thresholds on unseen or “unknown” characters.  
- Explore **transfer learning** with larger pre-trained models for better feature extraction.  
- Test hybrid approaches combining Siamese similarity with classification heads.

---

## Repository Structure

```
├── DisneyConv.ipynb              # Data preparation and conversion notebook
├── simple_cnn.ipynb              # Baseline CNN model
├── simaese_cnn.ipynb             # Siamese CNN architecture and training
├── similarity_scores.ipynb       # Evaluation of similarity-based classification
├── FirstProgressReport.docx      # Detailed project write-up
├── FirstProgressReportSlides.pptx# Presentation slides
└── README.md                     # Project overview
```

---

## References

- [Disney + PyTorch: Animated Character Recognition](https://medium.com/pytorch/how-disney-uses-pytorch-for-animated-character-recognition-a1722a182627)  
- IEEE Conference Papers on Facial Recognition and Image Classification (2018–2022)  
- Kaggle: Art Image Classification Dataset  
