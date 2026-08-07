import smtplib
from email.message import EmailMessage

def send_wildlife_alert(animal_name, confidence, image_path=None):
    # --- Configuration ---
    SENDER_EMAIL = "your_email@gmail.com"
    SENDER_PASSWORD = "your_app_password"  # Google App Password (not your normal password)
    RECEIVER_EMAIL = "recipient_email@gmail.com"

    # --- Build Email Message ---
    msg = EmailMessage()
    msg['Subject'] = f"🚨 ALERT: {animal_name.upper()} Detected!"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.set_content(f"A {animal_name} was detected with {confidence*100:.1f}% confidence.\nLocation: Field Camera 1")

    # --- Attach Image (Optional) ---
    if image_path:
        with open(image_path, 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename="detection.jpg")

    # --- Send via SMTP Server ---
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
            print(f"📧 Alert email sent for {animal_name}!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")