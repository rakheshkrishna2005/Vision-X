import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time
import os

# COCO dataset classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

st.set_page_config(initial_sidebar_state='expanded')

with open('main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# Function to perform object detection
def detect_objects(image, model, classes, conf):
    results = model(image, conf=conf, classes=classes)
    annotated_image = results[0].plot()
    return annotated_image, results[0].boxes.cls.tolist()

# Function to check CUDA availability
def is_cuda_available():
    return torch.cuda.is_available()

# Main Streamlit app
def main():
    # Initialize session state variables
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = 0
    if 'tracked_objects' not in st.session_state:
        st.session_state.tracked_objects = 0
    if 'detected_classes' not in st.session_state:
        st.session_state.detected_classes = 0

    st.markdown("<h1 style='text-align: center;'>Vision X - Real Time Object Tracker", unsafe_allow_html=True)
    st.markdown("---")
	
    # Sidebar options
    st.sidebar.title("⚙️ Settings")
    st.markdown("""
                <style>
                .stButton > button {
    				width: 100%;
				}
                </style>""", 
                unsafe_allow_html=True)
    
    # Check CUDA availability
    cuda_available = is_cuda_available()
    
    # Wrap settings in a container
    st.sidebar.markdown('<div class="settings-container">', unsafe_allow_html=True)
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.sidebar.markdown("---")
    enable_gpu = st.sidebar.checkbox("🤖 Enable GPU", value=False, disabled=not cuda_available)    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    use_webcam = st.sidebar.button("Use Webcam")
    
    selected_classes = st.sidebar.multiselect("Select Classes", CLASSES, default=['person', 'car'])
    
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")
        st.sidebar.info("To enable GPU acceleration, make sure you have a CUDA-capable GPU and PyTorch is installed with CUDA support.")
    
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    st.sidebar.markdown('''
	Created with ❤️ by [Rakhesh Krishna](https://github.com/rakheshkrishna2005/)
	''')
    # Convert selected classes to their indices
    class_indices = [CLASSES.index(cls) for cls in selected_classes]

    # Load YOLO model
    model = YOLO('yolov8n.pt')
    if enable_gpu and cuda_available:
        model.to('cuda')
        st.sidebar.success("GPU enabled successfully!")
    else:
        model.to('cpu')
        st.sidebar.info("Using CPU for processing.")
    
    # Create placeholder for video display
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

	# Create persistent placeholders for KPI metrics
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    tracked_objects_metric = kpi_col1.empty()
    frame_rate_metric = kpi_col2.empty()
    classes_metric = kpi_col3.empty()

    # Initialize KPI metrics with default values
    tracked_objects_metric.metric("Tracked Objects", "0")
    frame_rate_metric.metric("Frame Rate", "0.00 FPS")
    classes_metric.metric("Classes", "0")

	# Create placeholder for object count
    object_count_placeholder = st.empty()

    # Add CSS for the detected object table
    st.markdown("""
    <style>
    .detected-object-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
    }
    .detected-object-table th, .detected-object-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .detected-object-table th {
        background-color: var(--background-color);
    }
    .detected-object-table tr:nth-child(even) {
        background-color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)

    processed_frames = []

    if use_webcam:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update KPI metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update every second
                frame_rate = frame_count / elapsed_time
                unique_classes = len(np.unique(detected_classes))
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Display object count with updated CSS
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

            if not use_webcam:
                break

        cap.release()

    elif uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        vf = cv2.VideoCapture(tfile.name)
        frame_count = 0
        start_time = time.time()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update KPI metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.5:  # Update every half second
                frame_rate = frame_count / elapsed_time
                unique_classes = len(np.unique(detected_classes))
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Display object count with updated CSS
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        vf.release()

    # Update KPI metrics display
    st.markdown(f"""
    <script>
		document.getElementById('tracked-objects').innerText = "{st.session_state.tracked_objects}";
        document.getElementById('frame-rate').innerText = "{st.session_state.frame_rate:.2f} FPS";
        document.getElementById('detected-classes').innerText = "{st.session_state.detected_classes}";
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()