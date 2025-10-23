import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile # To handle video file saving
import time     # To control video playback speed

# --- Constants and Model Loading ---
DEMO_IMAGE = 'stand.jpg' # Make sure this file exists in the same directory

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Model input dimensions
width = 368
height = 368
inWidth = width
inHeight = height

# Load the network
# Make sure graph_opt.pb is in the same directory
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except cv2.error as e:
    st.error(f"Error loading the model file (graph_opt.pb): {e}")
    st.error("Please ensure the 'graph_opt.pb' file is in the same directory as the script.")
    st.stop() # Stop execution if model loading fails

# --- Pose Detection Function ---
# Using st.cache_data for better caching in newer Streamlit versions
@st.cache_data # Cache the results of this function
def poseDetector(frame, thres):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Prepare the frame for the network
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    # Run forward pass
    out = net.forward()
    # MobileNet output [1, 57, 46, 46], we only need the first 19 elements
    out = out[:, :19, :, :]  

    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    # Find the location of each body part
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :] # Slice heatmap of corresponding body's part.
        _, conf, _, point = cv2.minMaxLoc(heatMap) # Find the maximum confidence and its location
        # Scale the point coordinates back to the original frame size
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add point if confidence is above threshold, otherwise add None
        points.append((int(x), int(y)) if conf > thres else None)
        
    # Draw skeleton lines and keypoints
    frameCopy = frame.copy() # Work on a copy to avoid modifying the original frame directly
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # Draw line and ellipses if both points are detected
        if points[idFrom] and points[idTo]:
            cv2.line(frameCopy, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frameCopy, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frameCopy, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
    return frameCopy # Return the frame with pose drawn

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Human Pose Estimation")

st.title("Human Pose Estimation OpenCV 🤸‍♂️")
st.write("---")

st.sidebar.header("Options")

# Mode selection
mode = st.sidebar.radio(
    "Select Mode:",
    ("Image", "Video"),
    index=0 # Default to Image
)

st.sidebar.markdown("---")

# Threshold slider
thres_val = st.sidebar.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
st.sidebar.markdown("*(Lower threshold detects more points, but might include false positives)*")
st.sidebar.markdown("---")

stframe_original = st.empty()
stframe_processed = st.empty()


if mode == "Image":
    st.sidebar.subheader("Upload Image")
    img_file_buffer = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        st.sidebar.write("Using Demo Image.")
        try:
            image = np.array(Image.open(DEMO_IMAGE))
        except FileNotFoundError:
            st.error(f"Demo image '{DEMO_IMAGE}' not found. Please place it in the directory.")
            st.stop()

    # Convert PIL image (RGB) to OpenCV format (BGR)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    stframe_original.subheader("Original Image")
    stframe_original.image(image, caption="Original Image", use_column_width=True)

    # Process and display
    output_image = poseDetector(image_cv, thres_val)
    # Convert back to RGB for Streamlit display
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    stframe_processed.subheader("Pose Estimated")
    stframe_processed.image(output_image_rgb, caption="Pose Estimated", use_column_width=True)


elif mode == "Video":
    st.sidebar.subheader("Upload Video")
    vid_file_buffer = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if vid_file_buffer is not None:
        # Save uploaded video to a temporary file
        tffile = tempfile.NamedTemporaryFile(delete=False)
        tffile.write(vid_file_buffer.read())
        vf = cv2.VideoCapture(tffile.name)
        
        st.sidebar.text("Processing Video...")
        
        # Placeholders for video frames
        col1, col2 = st.columns(2)
        with col1:
            stframe_original.subheader("Original Video")
            video_placeholder_original = st.empty()
        with col2:
            stframe_processed.subheader("Pose Estimation")
            video_placeholder_processed = st.empty()

        prev_time = 0
        curr_time = 0

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                st.sidebar.write("Video processing finished.")
                break # End of video
            
            # --- Performance Measurement ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # --- Pose Estimation ---
            output_frame = poseDetector(frame, thres_val)
            
            # Add FPS text to the processed frame
            cv2.putText(output_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # --- Display ---
            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            video_placeholder_original.image(frame_rgb, channels="RGB", use_column_width=True)
            video_placeholder_processed.image(output_frame_rgb, channels="RGB", use_column_width=True)

            # Control playback speed slightly (optional)
            # time.sleep(0.01) 

        vf.release()
        cv2.destroyAllWindows() # Clean up any OpenCV windows if they were created

    else:
        st.sidebar.warning("Please upload a video file.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """    
    **How it works:**
    1. Upload an image or video.
    2. Adjust the confidence threshold.
    3. The app uses an OpenCV DNN model (based on OpenPose) to detect key body points.
    4. Detected points above the threshold are connected to form a skeleton.
    """
)
st.sidebar.markdown("Developed with using Streamlit and OpenCV by Haswanth.")