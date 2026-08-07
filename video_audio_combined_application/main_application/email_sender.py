
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "thefinisherm@gmail.com"
APP_PASSWORD = "eopifxllamiyutdc"
RECEIVER_EMAIL = "goat82279@gmail.com"

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