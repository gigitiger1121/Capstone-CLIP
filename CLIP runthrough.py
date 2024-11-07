import os
import clip
import torch
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set your CASIA dataset path
dataset_path = "C:/Users/gigi1/datasets/casia-2.0-dataset"

def load_images_from_folder(dataset_path):
    images = []
    labels = []
    for folder in ['authentic', 'tampered']:  # Loop over the folder names
        label_path = os.path.join(dataset_path, folder)
        print(f"Reading from: {label_path}")  # Debugging print
        for filename in os.listdir(label_path):  # List files in each folder
            img_path = os.path.join(label_path, filename)
            if not (filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))):  # Added more extensions
                print(f"Skipping non-image file: {filename}")  # Debugging print
                continue
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # Ensure the image is in RGB format
                    images.append(img)
                    labels.append(folder)  # Append folder name as label (authentic or tampered)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping invalid image file: {img_path}")  # Debugging print
                continue
    return images, labels

authentic_folder = os.path.join(dataset_path, 'authentic')
tampered_folder = os.path.join(dataset_path, 'tampered')

print(f"Files in authentic folder: {os.listdir(os.path.join(dataset_path, 'authentic'))}")
print(f"Files in tampered folder: {os.listdir(os.path.join(dataset_path, 'tampered'))}")


# Load and preprocess images
images, labels = load_images_from_folder(dataset_path)

#Image Poisoning Techniques
def add_gaussian_noise(image, mean=0, std=0.1):
    noisy_image = image + std * torch.randn_like(image) + mean
    return torch.clamp(noisy_image, 0, 1)

def apply_edge_detection(image):
    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    gray_img = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(edges_rgb.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

def add_watermark(image, text="WATERMARK"):
    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    pil_img = Image.fromarray((image_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    return preprocess(pil_img).unsqueeze(0).to(device)

#Generate CLIP Embeddings

def get_image_embedding(image):
    with torch.no_grad():
        return model.encode_image(image)

# Generate embeddings for authentic images
# Assuming `images` and `labels` are lists of images and their respective labels.
authentic_embeddings = [get_image_embedding(images[i]) for i in range(len(images)) if labels[i] == 0]


# Generate embeddings for tampered images
tampered_images = [add_gaussian_noise(images[i]) for i in range(len(images)) if labels[i] == 1]
tampered_embeddings = [get_image_embedding(img) for img in tampered_images]


#Anaylsis and Classification

# Ensure embeddings lists are not empty
if len(authentic_embeddings) == 0:
    print("No authentic images found.")
else:
    authentic_avg = torch.stack(authentic_embeddings).mean(dim=0)
    print(f"Number of authentic images: {len(authentic_embeddings)}")

# Similarly, check tampered embeddings
if len(tampered_embeddings) == 0:
    print("No tampered images found.")
else:
    tampered_avg = torch.stack(tampered_embeddings).mean(dim=0)

# Compute cosine similarity between the average authentic and tampered embeddings
if len(authentic_embeddings) > 0 and len(tampered_embeddings) > 0:
    similarity_score = cosine_similarity(authentic_avg.cpu().numpy().reshape(1, -1), tampered_avg.cpu().numpy().reshape(1, -1))
    print(f"Similarity between authentic and tampered images: {similarity_score[0][0]}")

    # Measure similarity individually for each pair of authentic and tampered embeddings
    for auth_emb, tamp_emb in zip(authentic_embeddings, tampered_embeddings):
        similarity = cosine_similarity(auth_emb.cpu().numpy().reshape(1, -1), tamp_emb.cpu().numpy().reshape(1, -1))
        print(f"Similarity: {similarity[0][0]}")

#Classification and Reporting

# Threshold similarity score to classify images as authentic or tampered
threshold = 0.8
predictions = [1 if cosine_similarity(auth.cpu().numpy(), tamp.cpu().numpy()) < threshold else 0
               for auth, tamp in zip(authentic_embeddings, tampered_embeddings)]

# Classification report
print(classification_report(labels, predictions, target_names=["Authentic", "Tampered"]))

# Confusion matrix
conf_matrix = confusion_matrix(labels, predictions)
print("Confusion Matrix:\n", conf_matrix)