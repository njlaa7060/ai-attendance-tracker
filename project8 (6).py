import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import pandas as pd
from sqlalchemy import create_engine
import time

# Define the model path
model_path = '/Users/linafaisal/Downloads/attendance-capstone-project/active_class_best.pt'
model = YOLO(model_path)

# Database connection setup
DATABASE_URI = 'postgresql+psycopg2://postgres:Lina1234@localhost:5432/T5'
engine = create_engine(DATABASE_URI)

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize dictionaries to store the number of frames each person is detected in for attendance and active class
attendance_frames_detected = {}
active_frames_detected = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Perform object detection on the current frame
        results = model(frame)

        # Display the frame
        cv2.imshow('Camera Frame', frame)

        for result in results:
            hands = []
            faces = []
            for box in result.boxes:
                if box.cls == 0:
                    hands.append(box)
                else:
                    faces.append(box)

            for face in faces:
                name = model.names[face.cls.tolist()[0]]
                
                # Draw bounding box around the face
                x1, y1, x2, y2 = map(int, face.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Write class name on the bounding box
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Increment the number of frames detected for attendance
                attendance_frames_detected[name] = attendance_frames_detected.get(name, 0) + 1
                
                # Increment the number of frames detected for active class
                if name not in active_frames_detected:
                    active_frames_detected[name] = 0
                if len(hands) > 0:
                    active_frames_detected[name] += 1

        # Wait for 10 milliseconds before capturing the next frame
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(600)
finally:
    # Calculate attendance percentage and active percentage for each person
    total_frames = sum(attendance_frames_detected.values())
    attendance_percentage = {name: (frames / total_frames) * 100 for name, frames in attendance_frames_detected.items()}
    active_percentage = {name: (active_frames / total_frames) * 100 for name, active_frames in active_frames_detected.items()}
    
    # Prepare attendance and active entries
    current_date = datetime.datetime.now()
    attendance_entries = [{'name': name, 'attendance_percentage': percentage, 'active_percentage': active_percentage.get(name, 0), 'date': current_date} for name, percentage in attendance_percentage.items()]
    
    # Convert the attendance entries to a DataFrame
    attendance_df = pd.DataFrame(attendance_entries)
    
    # Insert DataFrame into the database
    attendance_df.to_sql('student', engine, if_exists='append', index=False)
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
