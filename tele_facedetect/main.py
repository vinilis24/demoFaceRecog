import cv2
import telebot
import numpy as np
import time
import threading

# Telegram configuration
CHAT_ID = "-1002539424458"
bot = telebot.TeleBot("6549359951:AAEZsditszvMfqietyZxA-bSc3awJ7MZooc")  

# Face detection parameters
DETECTION_COOLDOWN = 30  # seconds
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RESIZE_DIM = (1280, 720)  # Reduced resolution for processing if needed

# Initialize camera with error handling
def setup_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

# Load pre-trained Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

last_alert_time = 0

def send_telegram_alert(frame):
    """Send image to Telegram asynchronously"""
    try:
        # Create a clean copy without detection rectangles
        img_encoded = cv2.imencode('.jpg', frame)[1]
        bot.send_photo(CHAT_ID, photo=img_encoded.tobytes())
        print("Alert sent successfully")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")

def main():
    global last_alert_time
    
    try:
        cap = setup_camera()
        print("Camera initialized successfully")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, retrying...")
                time.sleep(0.5)  # Add small delay before retry
                continue
                
            # Downscale frame for processing
            resized = cv2.resize(frame, RESIZE_DIM)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(60, 60),
            )
            
            current_time = time.time()
            
            # Create a separate clean frame for Telegram (without rectangles)
            clean_frame = frame.copy()
            
            # Draw all detected faces on display frame
            display_frame = frame.copy()
            scale_factor = frame.shape[1] / RESIZE_DIM[0]  # Calculate actual scale factor
            
            for (x, y, w, h) in faces:
                # Scale coordinates back to original resolution
                x_orig = int(x * scale_factor)
                y_orig = int(y * scale_factor)
                w_orig = int(w * scale_factor)
                h_orig = int(h * scale_factor)
                
                cv2.rectangle(
                    display_frame, 
                    (x_orig, y_orig), 
                    (x_orig + w_orig, y_orig + h_orig), 
                    (0, 255, 0), 
                    2
                )
            
            # Send alert if faces detected and cooldown period passed
            if len(faces) > 0 and (current_time - last_alert_time) > DETECTION_COOLDOWN:
                print(f"Face detected! Sending alert...")
                # Send alert in separate thread to prevent blocking
                # Use clean_frame to send image without detection rectangles
                alert_thread = threading.Thread(target=send_telegram_alert, args=(clean_frame,))
                alert_thread.daemon = True  # Make thread daemon so it exits with main program
                alert_thread.start()
                last_alert_time = current_time
            
            # Show preview with detection rectangles
            cv2.imshow('Face Detection', display_frame)
            
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
