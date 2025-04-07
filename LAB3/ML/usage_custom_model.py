import torch

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

img = r""  # Example image
results = model(img)
results.print()  # Print results to console
results.show()  # Display results in a window
results.save()  # Save results to runs/detect/exp
