# Vision Transformers for Image Processing  
🚀 Implementation of **"Pre-training Vision Transformers with Very Limited Synthesized Images"**  

---

## 📚 **1. Paper Summary**  
This project implements the methodology from the **Q1 research paper** focusing on:  
- **Vision Transformers (ViT)** for image classification  
- **Pre-training on synthetic datasets**  
- **Fine-tuning on real datasets (CIFAR-10)**  
- **Performance evaluation and comparison**  

🔗 **Paper Link**: [arXiv:2307.14710](https://arxiv.org/abs/2307.14710)  

---

## 🧪 **2. Methodology**  
### **2.1 Overview**  
This project explores **pre-training ViTs on limited synthetic images** and fine-tuning them on **CIFAR-10**.  

### **2.2 Steps**  
✔ **Synthetic Data Generation**: Create random image patterns as a pre-training dataset.  
✔ **Pre-train ViT Model**: Train a ViT model on synthetic images.  
✔ **Fine-tune on CIFAR-10**: Adapt the model for real-world image classification.  
✔ **Evaluate Performance**: Compute accuracy, confusion matrix, and visualizations.  

---

## 🔄 **3. Workflow**  

The following **workflow** outlines the entire pipeline:  

### **Step 1: Setup and Install Dependencies**  
- Install required Python libraries.  
- Download or generate datasets (CIFAR-10 & synthetic images).  

### **Step 2: Pre-Training the Vision Transformer (ViT)**  
- Use **synthetic images** for pre-training.  
- Train the ViT model using **limited synthesized data**.  
- Save the pre-trained model for fine-tuning.  

### **Step 3: Fine-Tuning on Real-World Data**  
- Load the pre-trained ViT model.  
- Fine-tune on **CIFAR-10** dataset.  
- Adjust hyperparameters to improve accuracy.  

### **Step 4: Evaluation & Performance Analysis**  
- Compute **test accuracy** and generate confusion matrix.  
- Compare performance before & after fine-tuning.  
- Visualize results (accuracy graphs, misclassified images).  

### **Step 5: Deployment & Inference**  
- Use the trained model to predict new images.  
- Save logs and evaluation results for documentation.  

---

## 📂 **4. Repository Structure**  

The repository follows this structure:  

```
📂 project-folder  
 ├── 📄 README.md               # Project documentation  
 ├── 📂 data/                   # Dataset links & instructions  
 ├── 📂 src/                    # Source code for training, inference, and evaluation  
 │   ├── train.py               # Training script  
 │   ├── finetune.py            # Fine-tuning script  
 │   ├── test.py                # Model evaluation  
 │   ├── predict.py             # Single-image prediction  
 │   ├── requirements.txt       # Required dependencies  
 ├── 📂 notebooks/              # Jupyter notebooks for experiments  
 │   ├── vit.ipynb     # ViT pre-training notebo 
 ├── 📂 results/                # Performance metrics, graphs, logs  
 │   ├── confusion_matrix.png   # Confusion matrix image 
 │   ├── output.png   # output image
 
```

---

## ⚙️ **5. Installation & Setup**  
### **5.1 Install Dependencies**  
Ensure you have Python 3.8+ installed, then run:  
```bash
pip install -r src/requirements.txt
```

### **5.2 Download & Prepare Datasets**  
- **CIFAR-10 dataset** is automatically downloaded.  
- **Synthetic data** is generated during training.  

---

## 🚀 **6. Training & Evaluation**  
### **6.1 Pre-train Vision Transformer on Synthetic Data**  
```bash
python src/train.py
```

### **6.2 Fine-tune on CIFAR-10**  
```bash
python src/finetune.py
```

### **6.3 Evaluate Model**  
```bash
python src/test.py
```

### **6.4 Make Predictions on a New Image**  
```bash
python src/predict.py --image path/to/image.jpg
```

---

## 📊 **7. Results & Observations**  
- ✅ **Test Accuracy**: **95.14%** on CIFAR-10  
- 📉 **Loss reduction observed** after pre-training  
- 🔎 **Confusion Matrix**:  
    ![cm](https://github.com/user-attachments/assets/79c3b0a4-0466-4f8c-a2d1-a06b4f357920)
    ![output](https://github.com/user-attachments/assets/275fb489-bde0-40cd-9277-b45981c91c57)


## 📘 **8. Challenges & Solutions**  
### ❌ **Issue: Training was slow**  
✅ **Solution**: Used **Mixed Precision (`torch.cuda.amp`)**, **batch size optimization**, and **PyTorch 2.0 compilation**.  

### ❌ **Issue: Input size mismatch error (128x128 vs 224x224)**  
✅ **Solution**: Resized images correctly to **224×224** before feeding into ViT.  

---

## 👨‍💻 **9. Contributors**  
- **Your Name** ([@yourGitHubHandle](https://github.com/yourGitHubHandle))  

---



