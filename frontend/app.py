import streamlit as st
from ultralytics import YOLO
from PIL import Image
model = YOLO("models/best.pt")
st.title("BrailleVision")
option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Webcam"]
)
if option == "Upload Image":
    uploaded = st.file_uploader(
        "Upload Braille Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded)
        with st.spinner("Detecting Braille..."):
            results = model(image)
            letters = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = float((x1 + x2) / 2)
                letters.append((center_x, label))
            letters.sort(key=lambda x: x[0])
            detected_text = "".join(
                letter for _, letter in letters
            )
            annotated = results[0].plot()
            st.image(
                annotated,
                use_container_width=True
            )
            st.success(detected_text)
elif option == "Use Webcam":
    picture = st.camera_input(
        "Take a picture"
    )
    if picture:
        image = Image.open(picture)
        with st.spinner("Detecting Braille"):
            results = model(image)
            letters = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = float((x1 + x2) / 2)
                letters.append((center_x, label))
            letters.sort(key=lambda x: x[0])
            detected_text = "".join(
                letter for _, letter in letters
            )
            annotated = results[0].plot()
            st.image(
                annotated,
                use_container_width=True
            )
            st.success(detected_text)
