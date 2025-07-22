import cv2
import csv
import datetime
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Joshua Vinilis"]

# CSV file setup
csv_file_path = "arrival_log.csv"
csv_header = ["ID", "Name", "Timestamp"]

if not os.path.isfile(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# Dictionary to store the last recorded timestamp for each person
last_recorded_timestamp = {}

# Counter for assigning unique IDs to unknown persons
unknown_counter = 1

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 50:
            name = name_list[serial]

            # Check if enough time has passed since the last recording
            current_timestamp = datetime.datetime.now()
            last_timestamp = last_recorded_timestamp.get(name, datetime.datetime.min)

            time_difference = current_timestamp - last_timestamp
            if time_difference.total_seconds() >= 600:  # 600 seconds = 10 minutes
                # Record arrival log in CSV
                timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                arrival_log = [serial, name, timestamp]

                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(arrival_log)

                # Update the last recorded timestamp for this person
                last_recorded_timestamp[name] = current_timestamp

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Record arrival log for unknown persons
            unknown_name = f"Unknown_{unknown_counter}"
            unknown_counter += 1

            # Record arrival log in CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            arrival_log = [serial, unknown_name, timestamp]

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(arrival_log)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, unknown_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
