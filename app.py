import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# Initialize the EasyOCR reader with the English language model
reader = easyocr.Reader(['en'])

# Set the page configuration
st.set_page_config(
    page_title="SnapText",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #c708b4;
        color: #333333;
    }
    .main {
        background-color: #4c6378;
        padding: 2rem;
        border-radius: 10px;
        color: #000000;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #001F3F;
        text-align: center; 
        margin-top: -50px;
        margin-bottom: 20px;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    .uploadedImage, .processedImage {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        padding: 10px;
        width: 45%;
    }
    .footer {
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        text-align: center;
        padding: 10px 0;
        color: #888888;
        font-size: 0.9rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# App title
main_heading = "<h1>📸 SnapText </h1>"
st.markdown(main_heading, unsafe_allow_html=True)

# File uploader for images
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Minimum confidence input
min_confidence = st.number_input("Minimum confidence:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Button to detect text
if st.button("Detect text"):
    if uploaded_file is not None:
        # Read and display the original image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption='Original Image', use_column_width=True)
        
        # Convert the image to a format suitable for OpenCV
        image = np.array(original_image)
        
        st.write("Detecting text...")
        result = reader.readtext(image)
        
        for res in result:
            if res[2] < min_confidence:
                continue
            # Draw rectangle and put text on the image
            start_x, start_y = int(res[0][0][0]), int(res[0][0][1])
            end_x, end_y = int(res[0][2][0]), int(res[0][2][1])
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(image, res[1], (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            st.write(f"Detected text: **{res[1]}** (Confidence: {res[2]:.2f})")
        
        # Display the processed image with detected text
        processed_image = Image.fromarray(image)
        st.image(processed_image, caption='Processed Image', use_column_width=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ by Rupanshu Kapoor</p>
    </div>
""", unsafe_allow_html=True)
