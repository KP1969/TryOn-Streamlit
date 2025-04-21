import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from utils.ui import set_background  # Optional if you have a background helper

# Set background and styling
set_background()

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

# App title
st.markdown('<div class="title"> Virtual Pants Try-On </div>', unsafe_allow_html=True)

# Centered webcam feed
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    stframe = st.image([])

# File uploader
uploaded_file = st.file_uploader("Upload a pants image (PNG with transparency)", type=["png"])

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

# Crop transparent clothing image
def extract_clothing_bbox(clothing_img):
    alpha = clothing_img[:, :, 3]
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    return clothing_img[y:y + h, x:x + w]

if uploaded_file is not None:
    clothing_img = Image.open(uploaded_file).convert("RGBA")
    clothing_img = np.array(clothing_img)
    clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_RGBA2BGRA)

    clothing_img = extract_clothing_bbox(clothing_img)
    cloth_mask = clothing_img[:, :, 3] > 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        h, w, _ = frame.shape

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            hip_width = abs(left_hip.x - right_hip.x) * w
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
            body_width = max(hip_width, shoulder_width)

            pants_width = int(body_width * 1.3)
            pants_width = max(80, min(pants_width, w - 20))

            hip_y = int((left_hip.y + right_hip.y) / 2 * h)
            ankle_y = int((left_ankle.y + right_ankle.y) / 2 * h)
            pants_height = int(abs(ankle_y - hip_y) * 1.2)
            pants_height = max(80, min(pants_height, h - 20))

            if pants_width < 10 or pants_height < 10:
                continue

            resized_cloth = cv2.resize(clothing_img, (pants_width, pants_height))
            resized_mask = resized_cloth[:, :, 3] > 0

            hip_x = int((left_hip.x + right_hip.x) / 2 * w)
            x1 = max(0, int(hip_x - pants_width / 2))
            y1 = max(0, int(hip_y - pants_height / 4))  # adjust vertically a bit

            x2 = min(x1 + pants_width, w)
            y2 = min(y1 + pants_height, h)

            crop_x = max(1, x2 - x1)
            crop_y = max(1, y2 - y1)

            cloth_crop = resized_cloth[:crop_y, :crop_x]
            mask_crop = resized_mask[:crop_y, :crop_x]
            roi = frame[y1:y1 + crop_y, x1:x1 + crop_x]

            if roi.shape[:2] != cloth_crop.shape[:2]:
                continue

            alpha = mask_crop.astype(np.float32)
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            roi = roi.astype(np.float32)
            cloth_rgb = cloth_crop[:, :, :3].astype(np.float32)
            blended = (1 - alpha) * roi + alpha * cloth_rgb
            frame[y1:y1 + crop_y, x1:x1 + crop_x] = blended.astype(np.uint8)

        with col2:
            stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

