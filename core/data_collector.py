import cv2
import os

def collect_face_data(user_id, camera_index=0, dataset_path="datasets", max_images=500):
    """
    Collect face images for training.
    
    Args:
        user_id (int): The ID of the user (used in the filename).
        camera_index (int): Index of the camera to use (default is 0).
        dataset_path (str): Path to the directory where images will be saved.
        max_images (int): Maximum number of images to capture.
    """
    # Ensure the dataset directory exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created dataset directory: {dataset_path}")

    # Initialize the camera
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print("Error: Could not open camera.")
        return

    # Load the Haar cascade
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    count = 0
    print(f"Starting data collection for User ID: {user_id}. Press 'q' to quit early.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            # Save the grayscale face image
            filename = f"{dataset_path}/User.{user_id}.{count}.jpg"
            cv2.imwrite(filename, gray[y:y+h, x:x+w])
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Collecting Face Data", frame)

        # Exit on 'q' key or when max_images is reached
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection complete. {count} images saved for User ID {user_id}.")