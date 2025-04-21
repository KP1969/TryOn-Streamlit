import streamlit as st

from utils.ui import set_background
set_background()

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Custom styles for title and uploader
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
st.markdown('<div class="title"> Long Sleeve / Sweatshirt Virtual Try-On</div>', unsafe_allow_html=True)

# Center webcam feed
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    stframe = st.image([])

# File uploader
uploaded_file = st.file_uploader("Upload a long sleeve or sweatshirt image (PNG with transparency)", type=["png"])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

if uploaded_file is not None:
    clothing_img = Image.open(uploaded_file).convert("RGBA")
    clothing_img = np.array(clothing_img)
    clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_RGBA2BGRA)
    cloth_mask = clothing_img[:, :, 3] > 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            img_h, img_w = frame.shape[:2]

            # Width: max of shoulder/elbow span with margin
            shoulder_span = abs(r_shoulder.x - l_shoulder.x) * img_w
            elbow_span = abs(r_elbow.x - l_elbow.x) * img_w
            width = int(max(shoulder_span, elbow_span) * 1.4)

            # Height: from shoulders to mid-forearm 
            torso = abs(((l_hip.y + r_hip.y) / 2) - ((l_shoulder.y + r_shoulder.y) / 2)) * img_h
            arm_extension = abs((l_wrist.y + r_wrist.y) / 2 - (l_shoulder.y + r_shoulder.y) / 2) * img_h
            height = int((torso + arm_extension * 0.5) * 1.4)

            width = max(100, min(width, img_w))
            height = max(150, min(height, img_h))

            # Resize clothing
            resized_clothing = cv2.resize(clothing_img, (width, height))
            resized_mask = cv2.resize(cloth_mask.astype(np.uint8) * 255, (width, height))

            # Position: centered at neck
            neck_x = (l_shoulder.x + r_shoulder.x) / 2
            neck_y = (l_shoulder.y + r_shoulder.y) / 2
            x = int(neck_x * img_w - width / 2)
            y = int(neck_y * img_h - height * 0.15)

            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            clothing_crop_x = min(width, img_w - x)
            clothing_crop_y = min(height, img_h - y)

            if clothing_crop_x > 0 and clothing_crop_y > 0:
                resized_clothing = resized_clothing[:clothing_crop_y, :clothing_crop_x]
                resized_mask = resized_mask[:clothing_crop_y, :clothing_crop_x]

                alpha = resized_mask / 255.0
                overlay = frame.copy()
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

