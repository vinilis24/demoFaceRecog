# Face Detection with Telegram Alerts

This project implements a face detection system using OpenCV and sends alerts with detected faces to a Telegram chat.

## Setup

Follow these steps to set up the environment and run the project:

## How to Get Your Telegram Bot Token and Chat ID

### a. Create a Telegram Bot

1. Open Telegram and search for the **BotFather**.
2. Start a chat with **BotFather** and send the command `/newbot`.
3. Follow the instructions to set up your bot. You will receive a **bot token** (e.g., `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`).

### b. Get Your Chat ID

1. Start a chat with your bot by searching for its username in Telegram and sending a message (e.g., "Hello").
2. Open your browser and navigate to:  
   `https://api.telegram.org/bot<your-bot-token>/getUpdates`  
   Replace `<your-bot-token>` with the token you received from **BotFather**.
3. Look for the `chat` object in the response. Your `id` field in the `chat` object is your **chat ID**.

### Example Response:

```json
{
  "ok": true,
  "result": [
    {
      "update_id": 123456789,
      "message": {
        "chat": {
          "id": -1001234567890,
          "title": "My Group",
          "type": "group"
        },
        "text": "Hello"
      }
    }
  ]
}
```

In this example, the chat ID is `-1001234567890`.

### c. Add the Bot Token and Chat ID to Your Code

1. Open the `.env` file in your project directory.
2. Add the following lines:
   ```plaintext
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   CHAT_ID=-1001234567890
   ```

Now your bot is ready to send messages to your chat!

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
