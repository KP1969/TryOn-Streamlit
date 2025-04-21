# Inside utils/ui.py
import os

def set_background():
    import streamlit as st
    import base64

    image_path = os.path.join(os.path.dirname(__file__), '..', 'image.png')  # One level up to main folder
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    background_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stSidebar"] {{ display: none; }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    html, body, [class*="css"] {{
        font-family: 'Source Sans Pro', sans-serif;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
