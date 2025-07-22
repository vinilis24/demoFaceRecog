# demoFaceRecog

## 🚀 Project Overview

This is a face recognition system with Telegram alert integration. The project has been restructured for better organization and ease of use.

### New Structure

```
demoFaceRecog/
├── easy_setup/               # Beginner-friendly setup and menu
│   ├── run.py                # Main entry point
│   ├── config.ini            # Configuration file
│   └── README.md             # Simple guide for beginners
├── core/                     # Core logic modules
│   ├── data_collector.py
│   ├── model_trainer.py
│   ├── face_recognizer.py
│   └── telegram_notifier.py
├── datasets/                 # Collected face images
├── models/                   # Trained model (Trainer.yml)
├── tele_facedetect/          # Legacy Telegram scripts (for reference)
├── requirements.txt
└── README.md                 # This file
```

## 🛠 Setup

1. **Create a Python virtual environment (optional but recommended)**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Telegram (optional)**
   - Copy `tele_facedetect/.env_sample` to `.env` in the project root.
   - Edit `.env` and add your Telegram Bot Token and Chat ID.

## 🧩 Easy Setup (Recommended for Beginners)

Use the new menu-driven system:

```bash
python easy_setup/run.py
```

Follow the on-screen menu to:

- Collect face data
- Train the model
- Run face recognition (basic or with Telegram alerts)

For more details, see [`easy_setup/README.md`](easy_setup/README.md:1).

## 🧠 Advanced Usage

You can still use the core modules directly:

- **Collect Data**: `python -c "from core.data_collector import collect_face_data; collect_face_data(user_id=1)"`
- **Train Model**: `python -c "from core.model_trainer import train_model; train_model()"`
- **Run Recognition**: `python -c "from core.face_recognizer import recognize_faces; recognize_faces()"`

## 📂 Datasets

Example datasets can be found here:  
[Google Drive Link](https://drive.google.com/file/d/1Uo808jFDhK97ae7zPob7KiKHU6pBl2Wv/view?usp=sharing)

## 📝 Notes

- The `others/` and `arrival-log/` directories contain experimental or legacy code.
- The `tele_facedetect/` directory is preserved for reference but is superseded by the new modular design.
