# object_detection.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the model
model = tf.saved_model.load("Dataset/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")

# Load an image
img_path = 'assets/fruits.jpg'  # Replace with your image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(img_rgb)
input_tensor = input_tensor[tf.newaxis, ...]

# Run detection
detections = model(input_tensor)

# Extract results
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() 
              for key, value in detections.items()}
detections['num_detections'] = num_detections

boxes = detections['detection_boxes']
classes = detections['detection_classes'].astype(np.int64)
scores = detections['detection_scores']

# Display results
for i in range(num_detections):
    if scores[i] > 0.5:
        box = boxes[i]
        y1, x1, y2, x2 = box
        (h, w) = img.shape[:2]
        cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
