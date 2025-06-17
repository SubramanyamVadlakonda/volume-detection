# 📦 Volume Detection using MiDaS AI

This project estimates the percentage of how much a container (e.g., a box or truck) is filled using depth estimation from 2D images. It uses the **MiDaS** deep learning model to infer depth from single images and compare two images: one with the container full and one current image to evaluate.

---

## 🔧 Features

- Uses Intel ISL MiDaS pre-trained models (`DPT_Large`)
- Compares average depth from two images
- Estimates percentage fill level
- Returns simple volume estimation in the terminal

---

## 📁 Project Structure

volume_detection/
│
├── images/
│ ├── box_full.jpg # Reference image of the full container
│ ├── box_half.jpg # Current image for comparison
│
├── output/
│ ├── box_half_result.jpg # Depth output for current image
│
├── weights/ # MiDaS weights folder
│
├── depth_estimation.py # Main Python script
└── README.md # You're reading it!

---

## ✅ Requirements

### 🔹 Python Version:
```bash
Python 3.8 or above


🔹 Required Libraries
Install all required packages using:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python matplotlib

How to Run
Clone the repository or download the code:
git clone https://github.com/SubramanyamVadlakonda/volume-detection.git
cd volume-detection

Run the main script:
python depth_estimation.py
