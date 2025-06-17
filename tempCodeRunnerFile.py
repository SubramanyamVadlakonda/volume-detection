import torch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a faster MiDaS model from PyTorch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# Load transforms from PyTorch Hub
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Input images
image_paths = ["images/box_full.jpg", "images/box_half.jpg"]

# Dictionary to store depth sums for analysis
depth_sums = {}

for img_path in image_paths:
    # Read and convert image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform and prepare image
    input_batch = transform(img).to(device)

    # Prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    output = prediction.cpu().numpy()
    depth_sums[os.path.basename(img_path)] = np.sum(output)

    # Normalize depth for visualization
    depth_min = output.min()
    depth_max = output.max()
    depth_vis = (output - depth_min) / (depth_max - depth_min)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    # Save depth image
    os.makedirs("output", exist_ok=True)
    depth_filename = os.path.basename(img_path).replace(".jpg", "_depth.jpg")
    depth_path = os.path.join("output", depth_filename)
    cv2.imwrite(depth_path, depth_vis)

    # Save raw depth data
    depth_raw_path = os.path.join("output", depth_filename.replace(".jpg", ".npy"))
    np.save(depth_raw_path, output)

# Simple volume check logic
full_volume = depth_sums.get("box_full.jpg", 0)
half_volume = depth_sums.get("box_half.jpg", 0)

if half_volume < full_volume * 0.95:  # Adjust the threshold as needed
    print("⚠️ Container is not full.")
else:
    print("✅ Container appears full.")
