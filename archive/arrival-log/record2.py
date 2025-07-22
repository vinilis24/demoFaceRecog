import cv2
import telepot
from telepot.loop import MessageLoop
from time import sleep
import datetime
import csv

# Replace 'your_bot_token' with your Telegram Bot Token
bot_token = '5695099094:AAH7KJWoetsoj0GRNEh5Zg_pwD9aNEzxiDU'  # Replace with your Telegram bot token
chat_id = '-1001695220317'  # Replace with your Telegram Chat ID

csv_file_path = 'record_arrival.csv'  # Replace with the desired CSV file path
csv_header = ["ID", "Name", "Timestamp"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

bot = telepot.Bot(token=bot_token)

# rtsp://<username>:<verification_code>@<camera_ip>/h264/ch1/main/av_stream
# Replace 'your_camera_url' with the actual URL or IP address of your IP camera
camera_url = 'rtsp://<username>:<verification_code>@<camera_ip>/h264/ch1/main/av_stream'

# video = cv2.VideoCapture("busfinal.mp4")
video = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

width, height = 1280, 720  # Set the desired width and height

# Set the video capture resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Joshua"]
CONFIDENCE_THRESHOLD = 60

count_ids = {}  # Initialize a dictionary to store counts for each label

unknown_count_threshold = 20

def recognize_face(frame, x, y, w, h):
    gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    serial, conf = recognizer.predict(gray_face)

    if conf < CONFIDENCE_THRESHOLD:
        if 0 <= serial < len(name_list):
            label = name_list[serial]
            print(f"Detected Face - Name: {label}")
        else:
            label = "Unknown"
            print(f"Detected Face - Name: {label}")
    else:
        label = "Unknown"

    # Draw rectangles around the detected faces and increment count_id for each label
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return label, conf

def send_telegram_photo(photo_path):
    try:
        bot.sendPhoto(chat_id, photo=open(photo_path, 'rb'))
    except telepot.exception.TelegramError as e:
        print(f"Telegram Error: {e}")

def save_photo(frame):
    photo_path = "unknown_person.jpg"
    cv2.imwrite(photo_path, frame)
    return photo_path

def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(f"Chat ID: {chat_id}")
    if content_type == 'text':
        bot.sendMessage(chat_id, 'Received a text message')

MessageLoop(bot, handle).run_as_thread()

while True:
    try:
        # sleep(5)
        ret, frame = video.read()

        if not ret:
            raise Exception("Error reading video frame")

        # Resize the frame
        resized_frame = cv2.resize(frame, (width, height))

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray_frame, scaleFactor=2, minNeighbors=5)

        # Draw rectangles around the detected faces and increment count_id for each label
        for (x, y, w, h) in faces:
            label, conf = recognize_face(resized_frame, x, y, w, h)

            if label not in count_ids:
                count_ids[label] = 1
            else:
                count_ids[label] += 1

            print(f"Count ID for {label}: {count_ids[label]}")

            if label == "Unknown" and count_ids[label] == unknown_count_threshold:
                # Save and send photo when count_id for Unknown reaches 20
                photo_path = save_photo(resized_frame)
                send_telegram_photo(photo_path)

                # Log arrival in CSV file
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([label, timestamp])

                # Reset count ID for Unknown
                count_ids[label] = 0

        # Display the frame with face detection
        cv2.imshow('Face Detection', resized_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Release resources
video.release()
cv2.destroyAllWindows()
