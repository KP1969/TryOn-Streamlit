import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils.ui import set_background
set_background()

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Set custom styles
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<div class="title"> Virtual T-Shirt Try-On </div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a clothing image (PNG with transparency)", type=["png"])

# Load MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

# Global variables for the uploaded T-shirt and mask
tshirt_image = None
cloth_mask = None

# Handle clothing image upload
if uploaded_file is not None:
    clothing_img = Image.open(uploaded_file).convert("RGBA")
    clothing_img = np.array(clothing_img)
    clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_RGBA2BGRA)
    tshirt_image = clothing_img
    cloth_mask = clothing_img[:, :, 3] > 0  # Alpha channel mask

# Define the video transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        global tshirt_image, cloth_mask

        img = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(img, 1)  # Mirror effect for webcam
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks (for debugging alignment)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks and tshirt_image is not None:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_chest = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_chest = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Calculate dimensions
            chest_width = int(abs(left_chest.x - right_chest.x) * frame.shape[1] * 1.2)
            shoulder_width = int(abs(left_shoulder.x - right_shoulder.x) * frame.shape[1] * 1.2)
            width = max(chest_width, shoulder_width, 100)
            torso_height = int(abs(left_shoulder.y - (left_hip.y + right_hip.y) / 2) * frame.shape[0] * 1.5)
            torso_height = max(100, torso_height)

            # Resize clothing and mask
            resized_clothing = cv2.resize(tshirt_image, (width, torso_height))
            resized_mask = cv2.resize(cloth_mask.astype(np.uint8) * 255, (width, torso_height))

            # Position the shirt at neck center
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            x = int(neck_x * frame.shape[1] - width / 2)
            y = int(neck_y * frame.shape[0] - torso_height * 0.15)

            # Clamp to stay within frame bounds
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))

            # Crop if shirt goes off screen
            clothing_crop_x = min(width, frame.shape[1] - x)
            clothing_crop_y = min(torso_height, frame.shape[0] - y)
            resized_clothing = resized_clothing[:clothing_crop_y, :clothing_crop_x]
            resized_mask = resized_mask[:clothing_crop_y, :clothing_crop_x]

            # Blend clothing onto frame
            alpha = resized_mask / 255.0
            for c in range(3):
                frame[y:y + clothing_crop_y, x:x + clothing_crop_x, c] = (
                    (1 - alpha) * frame[y:y + clothing_crop_y, x:x + clothing_crop_x, c] +
                    alpha * resized_clothing[:, :, c]
                )
        return frame

# Start the webcam regardless of upload
webrtc_streamer(key="try-on", video_transformer_factory=VideoTransformer)
