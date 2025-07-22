import cv2
import face_recognition
import os

# Function to load and encode images from a folder
def load_images_from_folder(folder_path):
    images = []
    encodings = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)

            if len(encoding) > 0:
                images.append(image)
                encodings.append(encoding[0])
                labels.append(filename.split('.')[0])  # Assuming the filename is the label

    return images, encodings, labels

# Replace 'dataset_folder' with the path to your dataset folder
dataset_folder = '/home/vinilis/Documents/Projects/demoFaceRecog/datasets'
known_images, known_encodings, known_labels = load_images_from_folder(dataset_folder)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_labels[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
