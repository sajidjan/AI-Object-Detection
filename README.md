# Object Detection with TensorFlow

This project demonstrates object detection using a pre-trained TensorFlow model (SSD MobileNet V2) and OpenCV. It includes a Python script for running detections on images and a simple web interface with a dark theme.

## Features

- Detects objects in images using TensorFlow and OpenCV
- Visualizes detections with bounding boxes
- Simple web UI with dark theme (see `static/styles.css`)

## Requirements

- Python 3.10
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ObjectDetection.git
   cd ObjectDetection
   ```

2. **Create and activate a virtual environment (Python 3.10):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained model:**
   - Place the SSD MobileNet V2 model in `Dataset/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model`
   - Place your test images in the `assets/` folder

## Usage

Run the detection script:
```bash
python object_detection.py
```

## Project Structure

```
ObjectDetection/
│
├── object_detection.py
├── requirements.txt
├── static/
│   └── styles.css
├── assets/
│   └── fruits.jpg
├── Dataset/
│   └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
│       └── saved_model/
└── README.md
```

## License

MIT License