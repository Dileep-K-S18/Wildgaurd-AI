
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os                       # Standard Python library to read system variables
from dotenv import load_dotenv  # Imports the tool you just installed

load_dotenv()                   # 👈 THIS LINE opens your .env file and loads the variables!

# Now you fetch the values using os.getenv("VARIABLE_NAME")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")


def alert_sender(message_body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        
        # ADD TIMESTAMP TO SUBJECT LINE:
        # This breaks Gmail's auto-threading so every alert shows up separately!
        current_time = datetime.now().strftime("%H:%M:%S")
        msg['Subject'] = f"🚨 FORESTGAURD WILDLIFE ALERT SYSTEM [{current_time}]"

        msg.attach(MIMEText(message_body, 'plain'))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print(f"Email sent successfully to {RECEIVER_EMAIL}")

    except Exception as e:
        print(f"Failed to send email: {e}")