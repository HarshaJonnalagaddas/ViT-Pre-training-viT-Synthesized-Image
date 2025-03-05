import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification
import argparse

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load fine-tuned model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)
model.load_state_dict(torch.load("src/fine_tuned_vit.pth"))
model.to(device)
model.eval()

# CIFAR-10 class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Function to predict an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image).logits
        _, predicted = torch.max(outputs, 1)
    
    print(f"Predicted Class: {class_names[predicted.item()]}")

# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    args = parser.parse_args()
    predict_image(args.image)
