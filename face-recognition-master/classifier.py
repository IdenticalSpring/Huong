from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLOv3 model
model = YOLO("model/yolo.weights")  # Replace with the actual path to your YOLOv3 weights

# Load the image and ensure it's in RGB mode (needed for YOLO)
image_path = "output_frames/frame_0000.jpg"
image = Image.open(image_path).convert("RGB")  # Convert to RGB to avoid potential issues

# Convert the image to a numpy array and back to PIL to ensure correct formatting
image = np.array(image)  # Convert to numpy array
image = Image.fromarray(image)  # Convert back to PIL Image

# Run predictions
results = model(image)  # Use the loaded model for inference

# Print the results
print(results)  # This will display predictions such as bounding boxes, classes, and confidence scores

# Optional: Visualize results
results.show()  # Display the image with bounding boxes and labels
