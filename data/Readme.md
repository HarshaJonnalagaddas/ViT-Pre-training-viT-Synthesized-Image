# 📌 CIFAR-10 Dataset

This project uses the **CIFAR-10 Dataset** for fine-tuning and evaluation.

---

## 🔹 CIFAR-10 Dataset
- Used for **training, fine-tuning, and evaluation**.
- Standard dataset with **60,000 images** across **10 classes**.
- Automatically downloaded when running training or testing scripts.

### **How It’s Downloaded**
- The dataset is automatically downloaded when running `train.py` and `test.py`:
  ```python
  import torchvision.datasets as datasets
  train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
  test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
  ```

---

## 🚀 Notes
- **No need to manually download CIFAR-10** – it is handled automatically.
- If a custom dataset is needed, modify `train.py` and `test.py`.

For any issues, refer to `src/train.py` and `src/finetune.py` for dataset handling. 🚀

