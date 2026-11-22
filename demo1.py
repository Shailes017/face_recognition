import cv2
import face_recognition
import json
import os
import numpy as np  # <-- Make sure to add this import

DB_FILE = r"C:\Users\achar\PycharmProjects\PythonProject\face_db.json"

# 1. Load the "database"
if not os.path.exists(DB_FILE):
    print("Error: Database file 'face_db.json' not found. Please enroll users first.")
    exit()

with open(DB_FILE, 'r') as f:
    data = json.load(f)

# Create lists of known face encodings and their names
known_face_encodings = [user["encoding"] for user in data]
known_face_names = [user["name"] for user in data]

print("Database loaded. Starting webcam...")

# 2. Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # 3. Find all faces and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 4. Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # --- THIS IS THE NEW "SMART" LOGIC ---
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the index (position) of the *best* match (smallest distance)
        best_match_index = np.argmin(face_distances)

        # Check if the best match is good enough (lower is stricter)
        # You can adjust this 0.55 value
        if face_distances[best_match_index] < 0.55:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        # --- END OF NEW LOGIC ---

        # 5. Draw a box and label
        # Draw a box around the face (I made it green)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face (with the 50px fix)
        cv2.rectangle(frame, (left, bottom - 30  ), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()