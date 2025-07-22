import cv2
import telebot
import numpy as np
import time
import threading
import os
import face_recognition
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram configuration from environment
CHAT_ID = os.getenv("CHAT_ID")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# Face detection parameters
DETECTION_COOLDOWN = 30  # seconds
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RESIZE_DIM = (1280, 720)

last_alert_time = 0

def send_telegram_alert(frame):
    """Send image to Telegram asynchronously"""
    try:
        img_encoded = cv2.imencode('.jpg', frame)[1]
        bot.send_photo(CHAT_ID, photo=img_encoded.tobytes())
        print("Alert sent successfully")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")

def main():
    global last_alert_time

    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("Camera initialized successfully")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, retrying...")
                time.sleep(0.5)
                continue

            # Create a clean copy of the frame (without rectangles)
            clean_frame = frame.copy()

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

            # Detect faces using face_recognition
            face_locations = face_recognition.face_locations(rgb_small_frame)

            current_time = time.time()

            # Draw rectangles around detected faces
            for top, right, bottom, left in face_locations:
                # Scale back up face locations to original frame size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Send alert if faces detected and cooldown period passed
            if len(face_locations) > 0 and (current_time - last_alert_time) > DETECTION_COOLDOWN:
                print("Face detected! Sending alert...")

                # Send clean_frame (without rectangles)
                alert_thread = threading.Thread(
                    target=send_telegram_alert, args=(clean_frame,)
                )
                alert_thread.daemon = True
                alert_thread.start()

                # Send frame (with rectangles)
                alert_thread_with_rectangles = threading.Thread(
                    target=send_telegram_alert, args=(frame,)
                )
                alert_thread_with_rectangles.daemon = True
                alert_thread_with_rectangles.start()

                last_alert_time = current_time

            # Show the frame with detection rectangles
            cv2.imshow('Face Detection', frame)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()
