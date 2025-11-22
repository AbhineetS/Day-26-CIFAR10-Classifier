# 📘 **Day 26 — CIFAR-10 Image Classifier (Custom CNN)**

A deep-learning project where we build, train, evaluate, and save a **Convolutional Neural Network (CNN)** from scratch to classify **CIFAR-10 images** across 10 categories.  
This project demonstrates image preprocessing, model building, regularization, training visualization, and exporting trained artifacts.

---

## 🚀 **Project Highlights**

- Built a **custom CNN** with multiple convolution blocks  
- Achieved **~70% accuracy** on CIFAR-10  
- Included **dropout** and **data normalization** for better generalization  
- Saved trained model as `.keras`  
- Generated training accuracy & loss plot  
- Clean, modular, production-style Python code  
- Repo stays lightweight due to proper `.gitignore`

---

## 🧠 **Dataset Used — CIFAR-10**

CIFAR-10 contains:

- **60,000 images** (32×32 RGB)
- **10 Classes:**
  - airplane  
  - automobile  
  - bird  
  - cat  
  - deer  
  - dog  
  - frog  
  - horse  
  - ship  
  - truck

---

## 📂 **Project Structure**

```
Day-26-CIFAR10-Classifier/
│
├── train_cifar10.py                # Main training script
├── requirements.txt                # Dependencies
├── .gitignore                      # Ignore venv + models + images
├── training_history_day26.png      # Training plot (ignored)
├── cnn_cifar10_day26.keras         # Saved model (ignored)
└── venv/                           # Virtual environment (ignored)
```

---

## ⚙️ **Installation & Setup**

### **1️⃣ Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ **Run the Training Script**

```bash
python3 train_cifar10.py
```

Running this will:

- Train the CNN  
- Print accuracy  
- Save model → `cnn_cifar10_day26.keras`  
- Save training plot → `training_history_day26.png`  

---

## 🏗️ **Model Architecture Overview**

- **Rescaling Layer** (normalize input images)  
- **3 Convolution Blocks**  
- **Dropout layers** for regularization  
- **Flatten → Dense(256) → Softmax(10)**  
- Designed to balance speed and performance  

---

## 📊 **Results Summary**

| Metric | Value |
|--------|--------|
| **Final Accuracy** | **0.7048** |
| **Loss** | Continually improved |
| **Epochs** | 12 |

---

## 🖼️ **Training Visualization**

Automatically generated:

```
training_history_day26.png
```

Contains:

- Accuracy curve  
- Loss curve  
- Clear overfitting/underfitting indicators  

---

## 📦 **Saved Model**

Stored locally as:

```
cnn_cifar10_day26.keras
```

File is intentionally **ignored** via `.gitignore` to avoid large GitHub uploads.

---

## 🔮 **Future Improvements**

- Add image augmentation  
- Implement ResNet-style skip connections  
- Increase model depth  
- Use Transfer Learning (MobileNetV2)  
- Create a Flask or FastAPI backend for predictions  
- Build a Streamlit frontend UI  

---

# 🟩 **10 Commit-friendly Blocks (for green squares)**

Copy each block into separate files/commits to grow your GitHub activity.

### **1️⃣ Block 1 — Project Summary**

```
## Project Summary
This project builds a CNN from scratch for CIFAR-10 image classification using TensorFlow.
```

### **2️⃣ Block 2 — Dataset**

```
## Dataset
CIFAR-10 contains 60k 32x32 RGB images across 10 distinct categories.
```

### **3️⃣ Block 3 — Model Architecture**

```
## Model Architecture
The CNN uses three convolution blocks with dropout and a dense classifier head.
```

### **4️⃣ Block 4 — Training Process**

```
## Training
The model is trained for 12 epochs using Adam optimizer and sparse categorical loss.
```

### **5️⃣ Block 5 — Achieved Accuracy**

```
## Accuracy
The final validation accuracy achieved was ~70%, a strong baseline for CIFAR-10.
```

### **6️⃣ Block 6 — Requirements Summary**

```
## Requirements
TensorFlow, NumPy, Matplotlib, scikit-learn, Pillow are required to run the project.
```

### **7️⃣ Block 7 — How to Run**

```
## Running Instructions
Activate your virtual env, install dependencies, and run train_cifar10.py.
```

### **8️⃣ Block 8 — Saved Artifacts**

```
## Artifacts
The trained model is saved as cnn_cifar10_day26.keras locally and excluded from Git.
```

### **9️⃣ Block 9 — Future Improvements**

```
## Future Work
Potential improvements include data augmentation, deeper network, or API deployment.
```

### **🔟 Block 10 — Author**

```
## Author
Created by Abhineet Singh as part of the 64-Day AI Challenge series.
```

---
