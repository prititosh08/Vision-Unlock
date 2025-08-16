import cv2
def generate_dataset():
    # Load the Haar cascade classifier for face detection
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Function to detect and crop the face from the image
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        # If no faces are detected, return None
        if len(faces) == 0:  
            return None

        # Crop the first detected face
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    # Initialize video capture from default camera (0)
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        # If a cropped face is detected
        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save the image
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)

            # Display the face with the image count
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", face)

        # Exit if Enter (ASCII 13) is pressed or 200 images are captured
        if cv2.waitKey(1) == 13 or img_id == 200:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

# Run the dataset generator
generate_dataset()
