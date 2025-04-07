# pip install torch torchvision numpy opencv-python yolov5

import torch
import cv2
import time
import numpy as np
# from yolov5 import YOLOv5
from ultralytics import YOLO

# Load YOLOv5 model (Replace 'best.pt' with your trained model)
path = rpath = r"C:\Users\LBIIO_C\PycharmProjects\TDS\results\2025_02_26_01\train6\weights\best.pt"
model_path = path #"best.pt"  # Path to trained YOLOv5 model


image_path = r"C:\Users\LBIIO_C\PycharmProjects\TDS\dataset_2025_02_26_1\images\1740482239889.jpg" #"test_image.jpg"  # Path to test image

# Load image
image = cv2.imread(image_path)

# Define function to run YOLOv5 inference
def run_inference(model, device):
    model.to(device)
    model.eval()
    
    # Convert image to YOLO format
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Measure time
    start_time = time.time()
    with torch.no_grad():
        results = model(img_tensor)
    end_time = time.time()

    return results, end_time - start_time

# Load YOLOv5 models on CPU and GPU
# model_cpu = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
# model_gpu = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
model_cpu = YOLO(path)
model_gpu = YOLO(path)

# Run inference on CPU
cpu_results, cpu_time = run_inference(model_cpu, "cpu")

# Run inference on GPU (if available)
if torch.cuda.is_available():
    gpu_results, gpu_time = run_inference(model_gpu, "cuda")
else:
    gpu_time = None
    print("CUDA not available. Running only on CPU.")

# Print execution time
print(f"CPU Inference Time: {cpu_time:.4f} seconds")
if gpu_time is not None:
    print(f"GPU Inference Time: {gpu_time:.4f} seconds")

# Compare FPS
cpu_fps = 1 / cpu_time
gpu_fps = 1 / gpu_time if gpu_time else None

print(f"CPU FPS: {cpu_fps:.2f}")
if gpu_fps:
    print(f"GPU FPS: {gpu_fps:.2f}")

# Display results
print("-----------------------")
print('CPU:')
print(cpu_results)
# cpu_results.show()  # Display CPU results
if gpu_time:
    #gpu_results.show()  # Display GPU results
    print("-----------------------")
    print('GPU:')
    print(gpu_results)
