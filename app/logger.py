import logging
import os
import datetime

def setup_logging():
    logger = logging.getLogger("TranscriptionApp")
    logger.setLevel(logging.INFO)
    log_directory = "./logs"
    os.makedirs(log_directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}.log"
    log_path = os.path.join(log_directory, log_filename)
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Logging is set up successfully. Logs are being saved to {log_path}")
    return logger