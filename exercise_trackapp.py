import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile 
import time
import math

# --- Constants and Model Loading ---
DEMO_IMAGE = 'stand.jpg' # Make sure this file exists

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

# --- Colors for Form Feedback ---
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Load the network
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except cv2.error as e:
    st.error(f"Error loading the model file (graph_opt.pb): {e}")
    st.error("Please ensure the 'graph_opt.pb' file is in the same directory as the script.")
    st.stop()

# --- Helper Function to Calculate Angle ---
def calculate_angle(a, b, c):
    """Calculates angle ABC (in degrees) formed by points A, B, C."""
    if not all([a, b, c]): # Check if all points were detected
        return None

    a = np.array(a) # First point
    b = np.array(b) # Middle point (vertex)
    c = np.array(c) # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# --- Pose Detection Function (Now returns points) ---
# We'll do the drawing in the main loop to control colors
def poseDetector(frame, thres):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = [None] * len(BODY_PARTS) # Initialize points list with None
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        if conf > thres:
            points[i] = (int(x), int(y)) # Store detected points
            
    return points # Return ONLY the points list

# --- Drawing Function ---
def draw_skeleton(frame, points, form_colors):
    """Draws the skeleton and highlights key joints based on form."""
    frame_drawn = frame.copy()
    
    # Draw skeleton lines
    for pair in POSE_PAIRS:
        idFrom = BODY_PARTS[pair[0]]
        idTo = BODY_PARTS[pair[1]]

        if points[idFrom] and points[idTo]:
            # Use white for base skeleton lines
            cv2.line(frame_drawn, points[idFrom], points[idTo], WHITE, 2)
            
    # Draw key joints (circles) with form-specific colors
    for part, color in form_colors.items():
        point_id = BODY_PARTS.get(part)
        if point_id is not None and points[point_id]:
            cv2.circle(frame_drawn, points[point_id], 7, color, thickness=-1, lineType=cv2.FILLED)
            
    return frame_drawn

# --- Initialize Session State ---
if 'pushup_counter' not in st.session_state:
    st.session_state.pushup_counter = 0
if 'pushup_stage' not in st.session_state:
    st.session_state.pushup_stage = "up"
if 'squat_counter' not in st.session_state:
    st.session_state.squat_counter = 0
if 'squat_stage' not in st.session_state:
    st.session_state.squat_stage = "up"
# Add pullup state
if 'pullup_counter' not in st.session_state:
    st.session_state.pullup_counter = 0
if 'pullup_stage' not in st.session_state:
    st.session_state.pullup_stage = "down"


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Exercise Coach")

st.title("AI Exercise Coach 🏋️‍♀️")
st.write("---")

st.sidebar.header("Controls")

# Mode selection
mode = st.sidebar.radio(
    "Select Input Source:",
    ("Image", "Video File", "Live Camera"),
    index=0
)

st.sidebar.markdown("---")

# Exercise selection
exercise = st.sidebar.selectbox(
    "Select Exercise to Track:",
    ("None", "Pushups", "Squats", "Pullups (Experimental)"),
    index=0
)

st.sidebar.markdown("---")

# Threshold slider
thres_val = st.sidebar.slider('Pose Confidence Threshold', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
st.sidebar.markdown("*(Lower threshold detects more points, but might include false positives)*")
st.sidebar.markdown("---")

# Placeholder for calibration button
st.sidebar.button("Calibrate (Future Feature)", disabled=True)
st.sidebar.caption("*Calibration will be needed for exercises like pullups to set bar height.*")

# --- UI Placeholders ---
col1, col2 = st.columns(2)
with col1:
    stframe_original = st.empty()
    stframe_original.subheader("Original Input")
    placeholder_original = st.empty()
with col2:
    stframe_processed = st.empty()
    stframe_processed.subheader("Pose Estimation & Analysis")
    placeholder_processed = st.empty()

# Sidebar placeholders for metrics
status_text = st.sidebar.empty()
counter_placeholder_col = st.sidebar.container()
feedback_placeholder = st.sidebar.empty()


# --- Mode Logic ---

if mode == "Image":
    st.sidebar.subheader("Upload Image")
    img_file_buffer = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        status_text.write("Using Demo Image.")
        try:
            image = np.array(Image.open(DEMO_IMAGE))
        except FileNotFoundError:
            st.error(f"Demo image '{DEMO_IMAGE}' not found.")
            st.stop()

    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    placeholder_original.image(image, caption="Original Image", use_column_width=True)

    # Process and display
    points = poseDetector(image_cv, thres_val)
    output_image = draw_skeleton(image_cv, points, {}) # No form colors for static
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    placeholder_processed.image(output_image_rgb, caption="Pose Estimated", use_column_width=True)
    status_text.success("Image processed.")

elif mode == "Video File" or mode == "Live Camera":
    vf = None
    if mode == "Video File":
        st.sidebar.subheader("Upload Video")
        vid_file_buffer = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        start_frame_slider = st.sidebar.empty() 

        if vid_file_buffer is not None:
            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(vid_file_buffer.read())
            vf = cv2.VideoCapture(tffile.name)
        else:
            status_text.warning("Please upload a video file.")

    elif mode == "Live Camera":
        st.sidebar.subheader("Live Camera Feed")
        st.sidebar.warning("Note: This will use your webcam. Grant permissions when prompted.")
        try:
            vf = cv2.VideoCapture(0) # Use 0 for default camera
            if not vf.isOpened():
                st.error("Could not open webcam. Check permissions or if it's in use.")
                st.stop()
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
            st.stop()

    if vf: # If VideoCapture object is valid
        # Get video properties
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Video File" else -1
        fps_video = vf.get(cv2.CAP_PROP_FPS) if mode == "Video File" else 30 

        start_frame = 0
        if mode == "Video File" and total_frames > 0:
            start_frame = start_frame_slider.slider(
                'Start processing from frame:', 0, total_frames - 1, 0, 1
            )
            st.sidebar.info(f"Video: {total_frames} frames (~{total_frames/fps_video:.1f}s @ {fps_video:.1f} FPS)")
            vf.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        elif mode == "Live Camera":
             st.sidebar.info("Processing live feed...")

        stop_processing = st.sidebar.button("Stop Processing", key="stop_button")
        status_text.text("Processing...")

        prev_time = 0
        frame_counter = start_frame

        # Reset counters/stage when starting new video/live feed
        st.session_state.pushup_counter = 0
        st.session_state.pushup_stage = "up"
        st.session_state.squat_counter = 0
        st.session_state.squat_stage = "up"
        st.session_state.pullup_counter = 0
        st.session_state.pullup_stage = "down"

        while vf.isOpened():
            if stop_processing:
                status_text.warning("Processing stopped by user.")
                break

            ret, frame = vf.read()
            if not ret:
                status_text.success("Video processing finished or stream ended.")
                break
                
            frame = cv2.flip(frame, 1) if mode == "Live Camera" else frame

            curr_time = time.time()
            fps_proc = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # --- Pose Estimation ---
            points = poseDetector(frame, thres_val) 

            # --- Exercise Logic & Form Feedback ---
            form_feedback = "Form: N/A"
            form_colors = {} # Dict to store colors for key joints
            
            # Default all key joints to blue (detected)
            for part in BODY_PARTS.keys():
                form_colors[part] = BLUE

            if exercise == "Pushups":
                l_shoulder = points[BODY_PARTS["LShoulder"]]
                l_elbow = points[BODY_PARTS["LElbow"]]
                l_wrist = points[BODY_PARTS["LWrist"]]
                l_hip = points[BODY_PARTS["LHip"]]
                l_ankle = points[BODY_PARTS["LAnkle"]]
                
                # Check for visibility
                if not all([l_shoulder, l_elbow, l_wrist, l_hip, l_ankle]):
                    form_feedback = "Form: Body not in view"
                    form_colors["LShoulder"] = RED; form_colors["LHip"] = RED
                else:
                    elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle) # Check for straight back
                    
                    form_feedback = "Form: OK"
                    form_colors["LElbow"] = GREEN
                    form_colors["LHip"] = GREEN
                    
                    # Form Check 1: Back straight
                    if hip_angle < 160 or hip_angle > 190: # Allow 180 +/- 15 degrees
                        form_feedback = "Form: Keep back straight!"
                        form_colors["LHip"] = RED
                    
                    # Counter Logic
                    if elbow_angle:
                        if elbow_angle > 160: # Arms mostly straight
                            st.session_state.pushup_stage = "up"
                            form_colors["LElbow"] = GREEN
                        elif elbow_angle < 90 and st.session_state.pushup_stage == "up":
                            st.session_state.pushup_stage = "down"
                            st.session_state.pushup_counter += 1
                            st.balloons() # Rep celebration!
                        
                        if st.session_state.pushup_stage == "down" and elbow_angle < 90:
                            form_colors["LElbow"] = GREEN # Good depth
                        elif st.session_state.pushup_stage == "up" and elbow_angle < 160:
                            form_colors["LElbow"] = RED # Not fully extended
                            form_feedback = "Form: Extend arms fully!"

            elif exercise == "Squats":
                l_shoulder = points[BODY_PARTS["LShoulder"]]
                l_hip = points[BODY_PARTS["LHip"]]
                l_knee = points[BODY_PARTS["LKnee"]]
                l_ankle = points[BODY_PARTS["LAnkle"]]

                if not all([l_shoulder, l_hip, l_knee, l_ankle]):
                    form_feedback = "Form: Body not in view"
                    form_colors["LHip"] = RED; form_colors["LKnee"] = RED
                else:
                    knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    hip_angle = calculate_angle(l_shoulder, l_hip, l_knee) # Check for back bend
                    
                    form_feedback = "Form: OK"
                    form_colors["LKnee"] = GREEN
                    form_colors["LHip"] = GREEN
                    
                    # Form Check 1: Back angle (prevent leaning too far fwd)
                    if hip_angle < 70:
                        form_feedback = "Form: Keep back straighter!"
                        form_colors["LHip"] = RED
                    
                    # Counter Logic
                    if knee_angle:
                        if knee_angle > 165: # Standing straight
                            st.session_state.squat_stage = "up"
                        elif knee_angle < 90 and st.session_state.squat_stage == "up": # Reached depth
                            st.session_state.squat_stage = "down"
                            st.session_state.squat_counter += 1
                            st.balloons()
                            
                        # Form Check 2: Depth
                        if st.session_state.squat_stage == "down" and knee_angle > 90:
                            form_feedback = "Form: Go deeper!"
                            form_colors["LKnee"] = RED

            elif exercise == "Pullups (Experimental)":
                l_shoulder = points[BODY_PARTS["LShoulder"]]
                l_elbow = points[BODY_PARTS["LElbow"]]
                nose = points[BODY_PARTS["Nose"]]
                
                form_feedback = "Form: (Calibrate First)"
                if not all([l_shoulder, l_elbow, nose]):
                    form_feedback = "Form: Body not in view"
                else:
                    # Simple logic: count when nose goes above shoulder
                    if nose[1] < l_shoulder[1]: # Y-coord is smaller when higher
                        st.session_state.pullup_stage = "up"
                    elif nose[1] > l_shoulder[1] and st.session_state.pullup_stage == "up":
                        st.session_state.pullup_stage = "down"
                        st.session_state.pullup_counter += 1
                        st.balloons()

            # --- Draw skeleton and info on frame ---
            output_frame = draw_skeleton(frame, points, form_colors)
            
            # --- Display Info on Frame ---
            cv2.putText(output_frame, f"Proc FPS: {int(fps_proc)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            frame_info = f"Frame: {frame_counter}"
            if total_frames > 0: frame_info = f"Frame: {frame_counter}/{total_frames}"
            cv2.putText(output_frame, frame_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

            # Display Exercise Info
            info_y = 90
            if exercise == "Pushups":
                cv2.putText(output_frame, f"Pushups: {st.session_state.pushup_counter}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                cv2.putText(output_frame, f"Stage: {st.session_state.pushup_stage}", (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            elif exercise == "Squats":
                cv2.putText(output_frame, f"Squats: {st.session_state.squat_counter}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                cv2.putText(output_frame, f"Stage: {st.session_state.squat_stage}", (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            elif exercise == "Pullups (Experimental)":
                cv2.putText(output_frame, f"Pullups: {st.session_state.pullup_counter}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                cv2.putText(output_frame, f"Stage: {st.session_state.pullup_stage}", (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

            # Display Form Feedback
            feedback_color = RED if "Form:" in form_feedback and "OK" not in form_feedback else GREEN
            cv2.putText(output_frame, form_feedback, (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

            # --- Display Frames in Streamlit ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            placeholder_original.image(frame_rgb, channels="RGB", use_column_width=True)
            placeholder_processed.image(output_frame_rgb, channels="RGB", use_column_width=True)

            # --- Update Sidebar Metrics ---
            counter_placeholder_col.empty() # Clear previous metrics
            feedback_placeholder.empty()
            if exercise == "Pushups":
                counter_placeholder_col.metric(label="Pushup Count", value=st.session_state.pushup_counter)
                feedback_placeholder.info(form_feedback)
            elif exercise == "Squats":
                counter_placeholder_col.metric(label="Squat Count", value=st.session_state.squat_counter)
                feedback_placeholder.info(form_feedback)
            elif exercise == "Pullups (Experimental)":
                counter_placeholder_col.metric(label="Pullup Count", value=st.session_state.pullup_counter)
                feedback_placeholder.info(form_feedback)

            frame_counter += 1
            
            # Control speed for recorded video playback
            if mode == "Video File":
                 sleep_time = max(0, (1.0/fps_video) - (time.time() - curr_time))
                 time.sleep(sleep_time)

        vf.release()
        cv2.destroyAllWindows()
        status_text.empty()

# --- Sidebar Info ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **How it works:**
    1.  Select Input Source & Exercise.
    2.  Upload file or use camera.
    3.  (Video) Use slider to set start frame.
    4.  Adjust confidence threshold.
    5.  App tracks keypoints, calculates angles, and counts reps.
    6.  Key joints are colored **Green** (good form) or **Red** (check form).
    
    *For best results, ensure your full body is visible from the side.*
    
    *Live camera streaming is experimental. For deployment, `streamlit-webrtc` is recommended.*
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with using Streamlit and OpenCV by Haswanth.")