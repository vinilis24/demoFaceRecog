# Face Recognition System - Easy Setup Guide

Welcome! This guide will help you get started with the face recognition system in just a few simple steps.

## 🚀 Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r ../requirements.txt
   ```

2. **Set Up Your Environment**

   - Copy the `.env_sample` file from `tele_facedetect/` to `.env` in the project root.
   - Edit `.env` and add your Telegram Bot Token and Chat ID.
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token_here
     CHAT_ID=your_chat_id_here
     ```

3. **Run the Easy Setup Script**

   ```bash
   python easy_setup/run.py
   ```

4. **Follow the Menu**
   - **Option 1**: Collect face data for a new person.
   - **Option 2**: Train the model on the collected data.
   - **Option 3**: Run face recognition (basic).
   - **Option 4**: Run face recognition with Telegram alerts.

## 📂 Project Structure

- `easy_setup/`: Beginner-friendly scripts and config.
- `core/`: Core logic and modules.
- `datasets/`: Stores collected face images.
- `models/`: Stores the trained model (`Trainer.yml`).

## ⚙️ Configuration

Edit `easy_setup/config.ini` to customize:

- User names and IDs
- Camera settings
- Telegram alert settings
- Model paths

That's it! You're ready to go. 🎉
