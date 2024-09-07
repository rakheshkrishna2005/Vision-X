# Vision X - Real Time Object Tracker

## Description

- Real-time object detection and tracking application. Uses YOLOv8 for efficient and accurate object detection.
- Built with Streamlit for an interactive web interface.
- Supports webcam input and video file upload.

## Demo

**Watch the demo video of the project:** [Demo Video](https://drive.google.com/file/d/1HYMSpNCDvfNKre6_LChZdRX6dNvlyOFL/view?usp=sharing)

## Features

- Live object detection from webcam feed
- Object detection on uploaded video files
- Customizable confidence threshold
- Multi-class object selection
- GPU acceleration support (if available)
- Real-time performance metrics display (FPS, tracked objects, detected classes)
- Responsive design for various screen sizes

## Tech Stack
- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Computer Vision Library:** OpenCV
- **Deep Learning Framework:** PyTorch
- **Object Detection Model:** Ultralytics YOLOv8
- **GPU Acceleration:** CUDA (if available)
- **Data Processing:** NumPy

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rakheshkrishna2005/Vision-X.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd Vision-X
   ```
   
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

4. Activate the Virtual Environment:
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

7. Open your web browser and navigate to the URL `http://localhost:8501`

## Web Page


## Additional Information
- **Streamlit Documentation:** [Streamlit Documentation](https://docs.streamlit.io/)
- **YOLOv8 Documentation:** [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **OpenCV Documentation:** [OpenCV Documentation](https://docs.opencv.org/)
- **PyTorch Documentation:** [PyTorch Documentation](https://pytorch.org/docs/)
- **NumPy Documentation:** [NumPy Documentation](https://numpy.org/doc/stable/)
