import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import face_recognition
from PIL import Image, ImageDraw

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = (255, 0, 0)  # Blue color (BGR format)
TEXT_COLOR = (255, 255, 255)  # White color (BGR format)


# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)")
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
parser.add_argument("--realtime", action="store_true", help="Enable real-time face detection using the camera")
args = parser.parse_args()


def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), name, fill=TEXT_COLOR)


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)


def detect_faces_realtime(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Performs real-time face detection using the camera feed.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    video_capture = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = video_capture.read()  # Read a frame from the camera feed

        # Resize the frame to reduce processing time (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR frame to RGB

        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = _recognize_face(face_encoding, loaded_encodings)
            if not name:
                name = "Unknown"

            # Scale the face locations back to the original size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), BOUNDING_BOX_COLOR, 2)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                TEXT_COLOR,
                1,
            )

        cv2.imshow("Video", frame)  # Display the resulting frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Break the loop if 'q' is pressed

    video_capture.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.realtime:
        detect_faces_realtime(model=args.m)
