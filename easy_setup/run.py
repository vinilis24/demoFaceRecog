import os
import time
import configparser
from core.data_collector import collect_face_data
from core.model_trainer import train_model
from core.face_recognizer import recognize_faces
from core.telegram_notifier import send_telegram_alert, is_telegram_configured

# Load configuration
config = configparser.ConfigParser()
config.read('easy_setup/config.ini')

def get_config(section, key, fallback=None, type_cast=str):
    """Helper to get config values with type casting."""
    try:
        if type_cast == bool:
            return config.getboolean(section, key)
        elif type_cast == int:
            return config.getint(section, key)
        else:
            return config.get(section, key, fallback=fallback)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return fallback

def main_menu():
    """Display the main menu and handle user choices."""
    while True:
        print("\n" + "="*50)
        print("       Face Recognition System - Easy Setup")
        print("="*50)
        print("1. Collect Face Data")
        print("2. Train Model")
        print("3. Run Face Recognition (Basic)")
        print("4. Run Face Recognition with Telegram Alerts")
        print("5. Exit")
        print("-"*50)
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            user_id = input("Enter User ID (number): ").strip()
            try:
                user_id = int(user_id)
                dataset_path = get_config('model', 'dataset_path', 'datasets')
                collect_face_data(user_id, dataset_path=dataset_path)
            except ValueError:
                print("Invalid ID. Please enter a number.")
        
        elif choice == '2':
            dataset_path = get_config('model', 'dataset_path', 'datasets')
            model_path = get_config('model', 'model_path', 'models/Trainer.yml')
            train_model(dataset_path=dataset_path, model_path=model_path)
        
        elif choice == '3':
            camera_index = get_config('camera', 'camera_index', 0, int)
            model_path = get_config('model', 'model_path', 'models/Trainer.yml')
            name_list = get_config('settings', 'name_list', 'Unknown', str)
            name_list = [name.strip() for name in name_list.split(',')]
            confidence_threshold = get_config('settings', 'confidence_threshold', 90, int)
            recognize_faces(
                camera_index=camera_index,
                model_path=model_path,
                name_list=name_list,
                confidence_threshold=confidence_threshold
            )
        
        elif choice == '4':
            if not is_telegram_configured():
                print("Error: Telegram is not configured. Please set BOT_TOKEN and CHAT_ID in your .env file.")
                continue
            
            camera_index = get_config('camera', 'camera_index', 0, int)
            model_path = get_config('model', 'model_path', 'models/Trainer.yml')
            name_list = get_config('settings', 'name_list', 'Unknown', str)
            name_list = [name.strip() for name in name_list.split(',')]
            confidence_threshold = get_config('settings', 'confidence_threshold', 90, int)
            
            print("Running face recognition with Telegram alerts...")
            print("Note: Alerts are sent with a cooldown to prevent spam.")
            
            # Integrate Telegram alerts into the face recognition process.
            last_alert_time = 0  # Track the last alert time to enforce cooldown
            cooldown_seconds = 60  # Cooldown period to prevent spam

            def alert_callback(name, confidence):
                nonlocal last_alert_time
                current_time = time.time()
                if current_time - last_alert_time >= cooldown_seconds:
                    alert_message = f"Face recognized: {name} with confidence {confidence}%"
                    try:
                        send_telegram_alert(alert_message)
                        print(f"Alert sent: {alert_message}")
                        last_alert_time = current_time
                    except Exception as e:
                        print(f"Failed to send Telegram alert: {e}")

            recognize_faces(
                camera_index=camera_index,
                model_path=model_path,
                name_list=name_list,
                confidence_threshold=confidence_threshold,
                on_recognize=alert_callback  # Pass the alert callback to the recognizer
            )
        
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main_menu()