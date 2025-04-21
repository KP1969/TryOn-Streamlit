import streamlit as st
import base64

# Set page config
st.set_page_config(
    page_title="Virtual Try-On App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Background styling function
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    background_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stSidebar"] {{
        display: none;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    .custom-selectbox select {{
        width: 250px !important;
        margin: 0 auto;
        display: block;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Apply background
set_background("image.png")

# Add spacing
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# Center title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: white; font-size: 50px;'>🛍️ Virtual Try-On App</h1>
            <p style='color: white; font-size: 20px;'>Choose a clothing type below to begin . .</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dropdown
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    with st.container():
        st.markdown('<div class="custom-selectbox">', unsafe_allow_html=True)
        clothing_type = st.selectbox(
            "",
            [
                "Select an option",
                "T Shirt",
                "Full Sleeve Shirt / Sweatshirt",
                "Pants",
                "Skirts / Shorts",
                "Dresses"
            ],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Redirect logic
page_mapping = {
    "T Shirt": "pages/tshirt.py",
    "Full Sleeve Shirt / Sweatshirt": "pages/sweatshirt.py",
    "Pants": "pages/pants.py",
    "Skirts / Shorts": "pages/skirts.py",
    "Dresses": "pages/dresses.py"
}

if clothing_type in page_mapping:
    st.session_state["selected_clothing"] = clothing_type
    st.switch_page(page_mapping[clothing_type])
