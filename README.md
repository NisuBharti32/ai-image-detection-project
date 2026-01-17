# AI Image Detection Project (AI vs Real)

This project classifies images as **AI-generated** or **Real** using a deep learning model based on the **MobileNet architecture**.  
The model is trained using transfer learning and achieves **very high accuracy** on validation and test data.


##  Features
- Binary image classification: **AI-generated vs Real**
- Transfer Learning using **MobileNet**
- High accuracy and strong generalization
- Clean GitHub repository (dataset & virtual environment excluded)
- Ready for inference and deployment

---
##  Tech Stack
- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Architecture:** MobileNet (CNN)  
- **Image Processing:** OpenCV  
- **Evaluation:** Accuracy, Confusion Matrix, Classification Report  

##  Project Structure
```
ai-image-detection-project/
│
├── model.py # Model training & evaluation
├── app.py # Inference / Flask app
├── ai_vs_real_model.h5 # Trained model
├── requirements.txt # Dependencies
├── .gitignore # Ignored files (dataset, venv)
└── README.md # Project documentation
```

##  Dataset

The dataset used for training is **not included** in this repository due to size constraints.

- Contains two classes:
  - **AI-generated images**
  - **Real images**
- Images were preprocessed and resized to **160×160**
- Dataset was split into **training** and **validation** sets

##  Model Architecture

- Base model: **MobileNet** (pre-trained on ImageNet)
- Transfer learning approach
- Custom classification head added
- Final output layer uses **Sigmoid activation** for binary classification

**Model Summary:**
- Total parameters: ~2.26 million
- Trainable parameters: 1,281
- Non-trainable parameters: ~2.25 million

## Training & Evaluation Results

- **Validation Accuracy:** 99.91%
- **Test Accuracy:** 99.91%
- Only **1 misclassification** out of **1150 test images**
 

### Classification Report
```
              precision    recall  f1-score   support

AI              1.00       1.00       1.00       358
REAL            1.00       1.00       1.00       792

accuracy                              1.00      1150
macro avg        1.00       1.00       1.00      1150
weighted avg     1.00       1.00       1.00      1150

```

### Confusion Matrix
```
[[358   0]
 [  1 791]]
```

##  How to Run the Project

### 1️ Clone the repository
```
git clone https://github.com/NisuBharti32/ai-image-detection-project.git
cd ai-image-detection-project
```
### 2 Install dependencies
```
 pip install -r requirements.txt
```
### 3️ Train the model
```
 python model.py
```
### 4 Run inference / application
```
 python app.py
```

##  Best Practices

- Dataset excluded due to large size
- Virtual environment (`venv`) not tracked
- Reproducible training pipeline
- Clean and modular code structure
- Proper evaluation using standard metrics

##  Future Improvements

- Multi-class AI image classification
- Explainable AI using Grad-CAM
- Web-based user interface
- Cloud deployment (Render / Hugging Face)
- Model optimization for faster inference
- 
 ## Team & Contribution

This project was developed as a **group project**.

###  My Role
**Nisu Bharti**
- Model training & evaluation
- Transfer learning using MobileNet
- Dataset preprocessing & splitting
- Performance analysis (accuracy, confusion matrix, classification report)

###  Team Members
- Member 1
- Member 2
- Member 3


##  License
This project is for **academic and learning purposes only**.

