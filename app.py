import streamlit as st
import numpy as np
import pandas as pd
import cv2
import face_recognition
import os
from datetime import datetime, date
from PIL import Image
from io import BytesIO

# --- CONFIG ---
IMAGES_PATH = 'images'
ATTENDANCE_CSV = 'Attendance.csv'

# --- UTILS ---
def load_known_faces(images_path):
    known_encodings = []
    known_names = []
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img = face_recognition.load_image_file(os.path.join(images_path, filename))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def mark_attendance(name):
    today = date.today().isoformat()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(ATTENDANCE_CSV):
        df = pd.DataFrame(columns=['Name', 'Timestamp', 'Date'])
        df.to_csv(ATTENDANCE_CSV, index=False)
    df = pd.read_csv(ATTENDANCE_CSV)
    if not ((df['Name'] == name) & (df['Date'] == today)).any():
        df = df.append({'Name': name, 'Timestamp': timestamp, 'Date': today}, ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)

def get_today_attendance():
    if not os.path.exists(ATTENDANCE_CSV):
        return []
    df = pd.read_csv(ATTENDANCE_CSV)
    today = date.today().isoformat()
    return df[df['Date'] == today]['Name'].tolist()

def recognize_faces_in_image(image, known_encodings, known_names):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        names.append(name)
    return names, face_locations

def draw_boxes(image, face_locations, names):
    for (top, right, bottom, left), name in zip(face_locations, names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.rectangle(image, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return image

# --- STREAMLIT UI ---
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.title("Face Recognition Based Attendance System")

# Sidebar dashboard
today_attendance = get_today_attendance()
st.sidebar.header("Today's Attendance")
st.sidebar.write(f"Total: {len(today_attendance)}")
if today_attendance:
    st.sidebar.write("\n".join(today_attendance))
else:
    st.sidebar.write("No attendance marked yet.")

# Load known faces
with st.spinner('Loading known faces...'):
    known_encodings, known_names = load_known_faces(IMAGES_PATH)

# Image input
st.header("Mark Attendance")
input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert('RGB')
            image = np.asarray(pil_img, dtype=np.uint8)
            st.write(f"[Upload] type: {type(image)}, dtype: {image.dtype}, shape: {getattr(image, 'shape', None)}")
            if (
                isinstance(image, np.ndarray)
                and image.dtype == np.uint8
                and image.ndim == 3
                and image.shape[2] == 3
                and image.shape[0] > 0
                and image.shape[1] > 0
            ):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                st.error(f"Uploaded image is not a valid uint8 RGB image. Type: {type(image)}, Dtype: {getattr(image, 'dtype', None)}, Shape: {getattr(image, 'shape', None)}")
                image = None
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            image = None

elif input_method == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        try:
            img_bytes = picture.getvalue()
            pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
            image = np.asarray(pil_img, dtype=np.uint8)
            st.write(f"[Webcam] type: {type(image)}, dtype: {image.dtype}, shape: {getattr(image, 'shape', None)}")
            if (
                isinstance(image, np.ndarray)
                and image.dtype == np.uint8
                and image.ndim == 3
                and image.shape[2] == 3
                and image.shape[0] > 0
                and image.shape[1] > 0
            ):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                st.error(f"Captured image is not a valid uint8 RGB image. Type: {type(image)}, Dtype: {getattr(image, 'dtype', None)}, Shape: {getattr(image, 'shape', None)}")
                image = None
        except Exception as e:
            st.error(f"Error loading image from camera: {e}")
            image = None

if image is not None:
    names, face_locations = recognize_faces_in_image(image, known_encodings, known_names)
    marked = []
    for name in names:
        if name != "Unknown":
            mark_attendance(name)
            marked.append(name)
    image_with_boxes = draw_boxes(image.copy(), face_locations, names)
    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
    if names:
        st.success(f"Detected: {', '.join(names)}")
        if marked:
            st.info(f"Attendance marked for: {', '.join(set(marked))}")
        else:
            st.warning("No known faces detected for attendance.")
    else:
        st.warning("No faces detected.")

# --- END --- 