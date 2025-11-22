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



