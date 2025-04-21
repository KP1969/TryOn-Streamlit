import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from utils.ui import set_background  # Optional: for setting a custom background

set_background()  # Remove if not using background

# Custom styling
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

st.markdown('<div class="title"> Virtual Dress Try-On</div>', unsafe_allow_html=True)

# Layout with centered webcam view
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    stframe = st.image([])

# File uploader
uploaded_file = st.file_uploader("Upload a dress image (PNG with transparency)", type=["png"])

# Setup MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# Run only when a file is uploaded
if uploaded_file:
    dress_img = Image.open(uploaded_file).convert("RGBA")
    dress_np = np.array(dress_img)
    dress_bgra = cv2.cvtColor(dress_np, cv2.COLOR_RGBA2BGRA)

    alpha_mask = dress_np[:, :, 3]
    visible_y = np.where(alpha_mask > 0)[0]
    top_offset = visible_y.min() if visible_y.size > 0 else 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            lm = result.pose_landmarks.landmark

            def pt(name):
                p = lm[mp_pose.PoseLandmark[name]]
                return int(p.x * w), int(p.y * h)

            ls, rs = pt("LEFT_SHOULDER"), pt("RIGHT_SHOULDER")
            lh, rh = pt("LEFT_HIP"), pt("RIGHT_HIP")
            le, re = pt("LEFT_ELBOW"), pt("RIGHT_ELBOW")

            neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            hips_known = (lh[1] > 0 and rh[1] > 0)
            torso_bottom = ((lh[1] + rh[1]) // 2) if hips_known else ((le[1] + re[1]) // 2)
            torso_height = torso_bottom - neck[1]

            shoulder_width = abs(rs[0] - ls[0])
            dress_width = int(shoulder_width * 1.9)
            dress_height = int(torso_height * 2.4)

            if dress_width < 80 or dress_height < 80:
                st.warning("Move back a little for better detection.")
                continue

            resized_dress = cv2.resize(dress_bgra, (dress_width, dress_height))
            resized_mask = resized_dress[:, :, 3]

            distance_factor = np.clip(1.0 - (shoulder_width / w), 0.0, 1.0)
            vertical_correction = int(35 * distance_factor)
            horizontal_correction = -8

            dress_center_x = neck[0]
            dress_top_y = neck[1] - top_offset - vertical_correction
            x1 = dress_center_x - dress_width // 2 + horizontal_correction
            y1 = dress_top_y
            x2, y2 = x1 + dress_width, y1 + dress_height

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_w, crop_h = x2 - x1, y2 - y1

            if crop_w > 0 and crop_h > 0:
                frame_crop = frame[y1:y2, x1:x2]
                dress_crop = resized_dress[0:crop_h, 0:crop_w]
                mask_crop = resized_mask[0:crop_h, 0:crop_w]

                alpha = mask_crop / 255.0
                for c in range(3):
                    frame_crop[:, :, c] = (
                        alpha * dress_crop[:, :, c] + (1 - alpha) * frame_crop[:, :, c]
                    )

                frame[y1:y2, x1:x2] = frame_crop

        with col2:
            stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
