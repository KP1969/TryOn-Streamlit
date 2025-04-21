import streamlit as st

from utils.ui import set_background
set_background()

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Set custom styles for title and file uploader
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .stFileUploader label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom title
st.markdown('<div class="title"> Virtual T-Shirt Try-On </div>', unsafe_allow_html=True)

# Use columns to center the camera feed
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    stframe = st.image([])

# File uploader
uploaded_file = st.file_uploader("Upload a clothing image (PNG with transparency)", type=["png"])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

if uploaded_file is not None:
    clothing_img = Image.open(uploaded_file).convert("RGBA")
    clothing_img = np.array(clothing_img)
    clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_RGBA2BGRA)

    # Convert clothing image to mask (assuming alpha channel is mask)
    cloth_mask = clothing_img[:, :, 3] > 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_chest = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]  # Approximate chest width
            right_chest = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Compute width using chest area instead of just shoulders
            chest_width = int(abs(left_chest.x - right_chest.x) * frame.shape[1] * 1.2)
            shoulder_width = int(abs(left_shoulder.x - right_shoulder.x) * frame.shape[1] * 1.2)
            width = max(chest_width, shoulder_width)  # Take the larger width

            # Compute height dynamically using torso height
            torso_height = int(abs(left_shoulder.y - (left_hip.y + right_hip.y) / 2) * frame.shape[0] * 1.5)
            width, torso_height = max(100, width), max(100, torso_height)

            # Resize clothing image
            resized_clothing = cv2.resize(clothing_img, (width, torso_height))
            resized_mask = cv2.resize(cloth_mask.astype(np.uint8) * 255, (width, torso_height))

            # Compute position
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            x = int(neck_x * frame.shape[1] - width / 2)
            y = int(neck_y * frame.shape[0] - torso_height * 0.15)  # Adjust Y for better fit

            # Ensure within bounds
            x, y = max(0, x), max(0, y)
            clothing_crop_x = min(width, frame.shape[1] - x)
            clothing_crop_y = min(torso_height, frame.shape[0] - y)
            resized_clothing = resized_clothing[:clothing_crop_y, :clothing_crop_x]
            resized_mask = resized_mask[:clothing_crop_y, :clothing_crop_x]

            # Overlay clothing with mask
            overlay = frame.copy()
            alpha = resized_mask / 255.0
            for c in range(3):
                overlay[y:y + clothing_crop_y, x:x + clothing_crop_x, c] = (
                        (1 - alpha) * frame[y:y + clothing_crop_y, x:x + clothing_crop_x, c] +
                        alpha * resized_clothing[:, :, c]
                )

            frame = overlay

        with col2:
            stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

