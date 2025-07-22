import cv2

def recognize_faces(camera_index=0, model_path="models/Trainer.yml", name_list=None, confidence_threshold=90):
    """
    Perform real-time face recognition using the trained LBPH model.
    
    Args:
        camera_index (int): Index of the camera to use.
        model_path (str): Path to the trained model file.
        name_list (list): List of names corresponding to user IDs.
        confidence_threshold (int): Confidence threshold for a valid recognition.
    """
    if name_list is None:
        name_list = ["Unknown", "Unknown", "Unknown", "Unknown"]  # Default fallback

    # Initialize the camera
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print("Error: Could not open camera.")
        return

    # Load the Haar cascade and recognizer
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    try:
        recognizer.read(model_path)
        print(f"Model loaded from {model_path}")
    except cv2.error:
        print(f"Error: Could not load model from {model_path}. Make sure it exists and is trained.")
        video.release()
        cv2.destroyAllWindows()
        return

    print("Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the face region
            gray_face = gray[y:y+h, x:x+w]
            serial, conf = recognizer.predict(gray_face)
            print(f"ID: {serial}, Confidence: {conf}")

            # Determine the label
            if conf < confidence_threshold:
                if 0 <= serial < len(name_list):
                    label = name_list[serial]
                else:
                    label = "Unknown"
            else:
                label = "Unknown"

            print(f"Detected Face - Name: {label}")

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resize and display the frame
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()