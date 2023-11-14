import numpy as np
import cv2
import face_recognition
from keras.models import load_model
import os


model = load_model('model.h5')

class_labels = ["Satvik", "Nancy", "Marco", "Sukrit", "Ari_big_sig"]

video_capture = cv2.VideoCapture(1)
video_capture.set(3, 1280)
video_capture.set(4, 720)

process_this_frame = True

while True:
    ret, frame = video_capture.read()

    frame = cv2.flip(frame, 1)

    # Resize the frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process every other frame to speed up face recognition
    if process_this_frame:
        # Find all face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_names = []
        for face_location in face_locations:
            # Extract the face from the frame
            top, right, bottom, left = face_location
            face_image = rgb_small_frame[top:bottom, left:right]

            # Resize the face image to match the CNN model's input size
            face_image = cv2.resize(face_image, (224, 224))

            # Convert the image to grayscale (if your model expects grayscale input)
            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

            # Normalize pixel values to the range [0, 1]
            face_image_gray = face_image_gray / 255.0

            # Reshape the image to match your model's input shape
            face_image_gray = face_image_gray.reshape(1, 224, 224, 1)

            # Make predictions using your CNN model
            predicted_class = np.argmax(model.predict(face_image_gray))

            # Map the predicted class to a name using class_labels
            name = class_labels[predicted_class]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
