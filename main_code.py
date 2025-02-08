#------->> libraries used <<-----------------------------------------------------------------------------------------------------------------------------------------------------------


import cv2
import numpy as np
import face_recognition
import os
import streamlit as st
from datetime import datetime
import pandas as pd


#------->> encodings saved in Numpy format  <<-----------------------------------------------------------------------------------------------------------------------------------------


def load_known_encodings(directory="./known_encodings/"):
    known_encodings = []
    known_names = []
    if not os.path.exists(directory):
        os.makedirs(directory)
        return known_encodings, known_names
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            encodings = np.load(os.path.join(directory, file), allow_pickle=True)
            known_encodings.extend(encodings)
            known_names.extend([name] * len(encodings))
    return known_encodings, known_names


#------->> detect and save encodings <<--------------------------------------------------------------------------------------------------------------------------------------------------


def save_encoding(name):
    save_path = './known_encodings/'
    os.makedirs(save_path, exist_ok=True)
    
    # Check for duplicate name
    existing_files = [f.split('.')[0] for f in os.listdir(save_path) if f.endswith('.npy')]
    if name in existing_files:
        st.error(f"Error: The name '{name}' is already used. Please choose a different name.")
        return

    cap = cv2.VideoCapture(0)
    encodings = []

    st.info(f"Looking for a face to save for {name}. Press 's' to save and 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from the webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.imshow("Save Encoding", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(face_locations) > 0:
                encoding = face_recognition.face_encodings(rgb_frame, face_locations)
                if encoding:
                    encodings.append(encoding[0])
                    st.success(f"Face encoding captured for {name}.")

    if encodings:
        filename = f"{name}.npy"
        np.save(os.path.join(save_path, filename), encodings)
        st.success(f"Encodings saved for {name}.")
    else:
        st.warning("No encodings to save.")
    
    cap.release()
    cv2.destroyAllWindows()


#------->> to compare encodings using face recognition <<---------------------------------------------------------------------------------------------------------------------------------------------


def recognize_faces():
    known_encodings, known_names = load_known_encodings()
    if not known_encodings:
        st.warning("No known encodings found. Please add some first.")
        return

    log_file = "recognition_log.csv"
    if not os.path.exists(log_file):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(log_file, index=False)

    cap = cv2.VideoCapture(0)
    st.info("Looking for faces. The camera will close after a match is found or if no match is found.")

    recognized_names = set()
    encodings_not_matched = False  

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from the webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                best_match_index = matches.index(True)
                name = known_names[best_match_index]
                if name not in recognized_names:
                    recognized_names.add(name)
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")
                    st.success(f"Recognized: {name} (Date: {date}, Time: {time})")
                    
                    new_record = pd.DataFrame([{"Name": name, "Date": date, "Time": time}])
                    df = pd.read_csv(log_file)
                    df = pd.concat([df, new_record], ignore_index=True)
                    df.to_csv(log_file, index=False)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            else:
                encodings_not_matched = True

        if encodings_not_matched:
            st.warning("Encodings not matched.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if encodings_not_matched:
        st.warning("Camera closed due to no matches found.")


#-------->> using main function to access all other functions <<---------------------------------------------------------------------------------------------------------------------------


def main():
    st.title("Attendence Automation \n")

    menu = ["Home", "Save Encoding", "Recognize Faces", "View Attendence Records"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Face Recognition Application")
        st.markdown("""
        Use this app to:
        - Save face encodings.
        - Recognize faces using your webcam.
        - View recognition records.
        Select an option from the sidebar to get started.
        """)
    elif choice == "Save Encoding":
        st.subheader("Save a Face Encoding")
        name = st.text_input("Enter the name for the face encoding:")
        
        if st.button("Save Encoding"):
            if name:
                save_encoding(name)
            else:
                st.warning("Please enter a name before saving.")
    elif choice == "Recognize Faces":
        st.subheader("Recognize Faces")
        if st.button("Start Recognition"):
            recognize_faces()
    elif choice == "View Attendence Records":
        st.subheader("Recognition Records")
        log_file = "recognition_log.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            st.dataframe(df)
        else:
            st.warning("No recognition records found.")

if __name__ == "__main__":
    main()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    