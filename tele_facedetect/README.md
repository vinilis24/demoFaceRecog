# Face Detection with Telegram Alerts

This project implements a face detection system using OpenCV and sends alerts with detected faces to a Telegram chat.

## Setup

Follow these steps to set up the environment and run the project:

### 1. Create a Python Virtual Environment

```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Linux/MacOS
source env/bin/activate
# On Windows
env\Scripts\activate
```

### 2. Install Required Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Add Your Telegram Bot Token and Chat ID

Update the following variables in [`tele_facedetect/main.py`](tele_facedetect/main.py):

- Replace `bot = telebot.TeleBot("<your-bot-token>")` with your Telegram bot token.
- Replace `CHAT_ID = "<your-chat-id>"` with your Telegram chat ID.

### 5. Run the Program

Run the main script to start the face detection system:

```bash
python tele_facedetect/main.py
```

## Requirements

The following Python packages are required:

- `opencv-python`
- `opencv-python-headless`
- `numpy`
- `pyTelegramBotAPI`

These are listed in the `requirements.txt` file.

## Features

- Detects faces using OpenCV's Haar Cascade classifier.
- Sends detected face images to a Telegram chat.
- Adjustable detection cooldown to prevent spamming alerts.

## Notes

- Ensure your webcam is connected and accessible.
- Make sure the `haarcascade_frontalface_default.xml` file is available in your OpenCV installation directory.
- The program uses a resolution of 1280x720 for face detection. Adjust `FRAME_WIDTH` and `FRAME_HEIGHT` in the code if needed.

## Troubleshooting

- If the webcam cannot be accessed, ensure no other application is using it.
- If Telegram alerts fail, verify your bot token and chat ID.

## License

This project is licensed under the MIT License.
