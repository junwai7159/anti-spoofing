import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('./runs/detect/train/weights/best.pt')

st.title("YOLOv8 Object Detection")
st.write("Upload an image to detect live/spoof using YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Run inference
    print('Inference')
    results = model.predict(source=img)
    print('Done')
    
    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = 'spoof' if int(box.cls.item()) else 'live'
            confidence = box.conf.item()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the results
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Processed Image', use_column_width=True)