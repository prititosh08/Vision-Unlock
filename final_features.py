import cv2
import numpy as np
from PIL import Image
import os
import time

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    face_count = len(features)  # Count detected faces

    for (x, y, w, h) in features:
        # Default green box
        box_color = color
        text_color = color

        # Predict the face
        id, pred = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100 * (1 - pred / 300))

        # Check confidence to determine recognized or unknown
        if confidence > 75:
            if id == 1:
                cv2.putText(img, "legend", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                print("✅ Access Granted: Welcome, legend!")
            if id == 2:
                cv2.putText(img, "Manish", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                print("✅ Access Granted: Welcome, Manish!")
        else:
            # Change box and text to red for unknown faces
            box_color = (0, 0, 255)  # Red color
            text_color = (0, 0, 255)
            cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)
            print("⛔ Access Denied: Unknown person detected!")

        # Draw the rectangle with the chosen color
        cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)

    # Show the face count on the feed
    cv2.putText(img, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

    return img

# Load classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained model
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Start video capture
video_capture = cv2.VideoCapture(0)

# FPS tracker
start_time = time.time()

while True:
    ret, img = video_capture.read()

    # Calculate FPS
    fps = int(1 / (time.time() - start_time))
    start_time = time.time()

    # Display FPS on the feed
    cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    img = draw_boundary(img, faceCascade, 1.3, 6, (0, 255, 0), clf)
    cv2.imshow("Face Detection", img)

    # Press 'Enter' to break the loop
    if cv2.waitKey(1) == 13:
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
