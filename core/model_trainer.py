import cv2
import numpy as np
from PIL import Image
import os

def train_model(dataset_path="datasets", model_path="models/Trainer.yml"):
    """
    Train the LBPH face recognizer model using images from the dataset.
    
    Args:
        dataset_path (str): Path to the directory containing face images.
        model_path (str): Path where the trained model will be saved.
    """
    # Ensure the model directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    # Create the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Function to get image IDs and face data
    def get_image_id_and_faces(path):
        faces = []
        ids = []
        try:
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        except FileNotFoundError:
            print(f"Error: Dataset directory '{path}' not found.")
            return [], []

        for image_path in image_paths:
            try:
                # Open image and convert to grayscale
                face_image = Image.open(image_path).convert('L')
                face_np = np.array(face_image, 'uint8')

                # Extract ID from the filename (User.<ID>.<count>.jpg)
                filename = os.path.split(image_path)[-1]
                user_id = int(filename.split('.')[1])

                faces.append(face_np)
                ids.append(user_id)

                # Optional: Show the image being trained on
                # cv2.imshow("Training", face_np)
                # cv2.waitKey(10)
            except Image.UnidentifiedImageError as e:
                print(f"Error: Invalid image format for {image_path}. {str(e)}")
                continue
            except ValueError as e:
                print(f"Error: Unable to parse user ID from filename {image_path}. {str(e)}")
                continue
            except IndexError as e:
                print(f"Error: Unexpected filename structure for {image_path}. {str(e)}")
                continue

        return ids, faces

    # Get the face data and IDs
    print("Loading dataset for training...")
    ids, face_data = get_image_id_and_faces(dataset_path)

    if len(ids) == 0:
        print("No valid face data found. Training aborted.")
        return

    # Train the recognizer
    print(f"Training model with {len(ids)} face samples...")
    recognizer.train(face_data, np.array(ids))

    # Save the trained model
    recognizer.write(model_path)
    print(f"Training completed. Model saved to {model_path}.")

    # Cleanup
    cv2.destroyAllWindows()