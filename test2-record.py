import cv2
import csv
import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Joshua Vinilis"]
CONFIDENCE_THRESHOLD = 50

csv_file_path = "arrival_log.csv"
csv_header = ["Name", "Timestamp"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

unknown_counter = 1  # Initialize counter for different unknown faces

def recognize_face(frame, x, y, w, h):
    global unknown_counter

    gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    _, conf = recognizer.predict(gray_face)
    print("Confidence:", conf)

    if conf < CONFIDENCE_THRESHOLD:
        name = name_list[1]  # Assuming the second name in the list is the known person
    else:
        name = f"Unknown_{unknown_counter}"
        unknown_counter += 1

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append arrival log to CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        recognize_face(frame, x, y, w, h)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
