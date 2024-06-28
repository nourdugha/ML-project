'''
Logging in Python is a crucial aspect of application development, as it allows you to record information about the execution of your code, such as errors, warnings, informational messages, and debug information.
'''
import logging
import os
from datetime import datetime

# file_name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# creates the directory where the log file will be stored is
os.makedirs(logs_path, exist_ok=True)

# constructs the full path of the log file is
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
