import cv2
# import csv
# import datetime

video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("busfinal.mp4")
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Joshua",  "Vinilis", "Julfadzly"]
CONFIDENCE_THRESHOLD = 60

def recognize_face(frame, x, y, w, h):
    gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    serial, conf = recognizer.predict(gray_face)

    if conf < CONFIDENCE_THRESHOLD:
        if 0 <= serial < len(name_list):
            name = name_list[serial]
            print(f"Detected Face - Name: {name}")
        else: 
            name = "Unknown"
            print(f"Detected Face - Name: {name}")
    else:
        name = "Unknown"
        print(f"Detected Face - Name: {name}")

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, f"{name} (ID: {serial})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# counter=[]

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Recognition for faces
        recognize_face(frame, x, y, w, h)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

