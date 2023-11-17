import cv2
import csv
import datetime

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("busfinal.mp4")
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Joshua Vinilis"]
CONFIDENCE_THRESHOLD = 50
CONSISTENCY_THRESHOLD = 5

csv_file_path = "arrival_log.csv"
csv_header = ["ID", "Name", "Timestamp"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

known_faces = {}  # Dictionary to store IDs of known faces
unknown_counter = 1  # Counter for different unknown faces

def recognize_known_face(frame, x, y, w, h):
    global known_faces

    gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    serial, conf = recognizer.predict(gray_face)
    print("Known Face ID:", serial, "Confidence:", conf)

    if conf < CONFIDENCE_THRESHOLD:
        if 0 <= serial < len(name_list):
            name = name_list[serial]
            if serial not in known_faces:
                known_faces[serial] = {'counter': 0}
                print(f"Detected Face - Name: {name}")
            else:
                known_faces[serial]['counter'] += 1
                print(f"Detected Face - Name: {name}")

                if known_faces[serial]['counter'] < CONSISTENCY_THRESHOLD:
                    return  # Skip recording if not consistent over multiple frames
        else: 
            name = "Unknown"
            print(f"Detected Face - Name: {name}")

    else:
        name = "Unknown"
        global unknown_counter
        name = f"Unknown_{unknown_counter}"
        unknown_counter += 1

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append arrival log to CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([serial, name, timestamp])

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, f"{name} (ID: {serial})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Separate recognition for known faces
        recognize_known_face(frame, x, y, w, h)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
