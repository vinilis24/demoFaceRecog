import cv2
import numpy as np
from PIL import Image
import os

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.createLBPHFaceRecognizer()

# Path to the dataset
path = "datasets"

# Function to get image IDs and faces
def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagePath:
        # Open image and convert to grayscale
        faceImage = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImage)

        # Extract ID from the filename
        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces.append(faceNP)
        ids.append(Id)

        cv2.imshow("Training", faceNP)
        cv2.waitKey(10)  # Adjust the wait time between images

    return ids, faces

# Get image IDs and faces
IDs, facedata = getImageID(path)

# Train the LBPH face recognizer
recognizer.train(facedata, np.array(IDs))

# Save the trained model to a file
recognizer.write("Trainer.yml")

# Close any open OpenCV windows
cv2.destroyAllWindows()

# Print a completion message
print("Training Completed............")
