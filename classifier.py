import os
import cv2
from PIL import Image  # pip install pillow
import numpy as np     # pip install numpy

def train_classifier(data_dir):
    # Collect paths of images in the data directory
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    faces = []
    ids = []

    for image in path:
        try:
            # Load the image and convert it to grayscale
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            
            # Extract the ID from the filename (e.g., user.1.4.jpg -> id=1)
            id = int(os.path.split(image)[1].split(".")[1])
            
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Skipping {image} due to error: {e}")

    # Convert ids to a NumPy array
    ids = np.array(ids)

    # Create and train the classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Save the trained classifier to a file
    clf.write("classifier.xml")
    print("Training completed and saved as 'classifier.xml'")

train_classifier("data")
