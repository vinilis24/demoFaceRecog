import cv2
import telebot
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Telegram bot
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("Warning: Telegram BOT_TOKEN or CHAT_ID not set in environment variables.")

bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None

def send_telegram_alert(frame, with_rectangles=True):
    """
    Send an image to Telegram asynchronously.
    
    Args:
        frame (numpy.ndarray): The image frame to send.
        with_rectangles (bool): Whether to send the frame with detection rectangles.
    """
    if not bot:
        print("Error: Telegram bot is not configured. Check your environment variables.")
        return

    def _send():
        try:
            # Encode the frame to JPEG
            img_encoded = cv2.imencode('.jpg', frame)[1]
            bot.send_photo(CHAT_ID, photo=img_encoded.tobytes())
            print("Telegram alert sent successfully.")
        except Exception as e:
            print(f"Telegram send failed: {str(e)}")

    # Send in a separate thread to prevent blocking
    alert_thread = threading.Thread(target=_send)
    alert_thread.daemon = True
    alert_thread.start()

def is_telegram_configured():
    """
    Check if Telegram is properly configured.
    
    Returns:
        bool: True if BOT_TOKEN and CHAT_ID are set, False otherwise.
    """
    return bool(BOT_TOKEN and CHAT_ID)