import cv2
from ultralytics import YOLO
import time
import os
import winsound

# --- DYNAMIC PATH SETUP (The fix for your folder move) ---
# This line automatically finds the folder where this script is saved
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Joining paths intelligently using the BASE_PATH found above
ON_IMG_PATH = os.path.join(BASE_PATH, "office_on.jpg")
OFF_IMG_PATH = os.path.join(BASE_PATH, "office_off.png")
LOG_FILE = os.path.join(BASE_PATH, "room_log.txt")
MODEL_PATH = os.path.join(BASE_PATH, "yolov8n.pt")

# Load YOLO
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

# Load Images
img_on = cv2.imread(ON_IMG_PATH)
img_off = cv2.imread(OFF_IMG_PATH)

# Resize images to match (e.g., 800x500)
if img_on is not None and img_off is not None:
    img_on = cv2.resize(img_on, (800, 500))
    img_off = cv2.resize(img_off, (800, 500))
else:
    print(f"Error: Files missing in {BASE_PATH}")
    print("Ensure office_on.jpg, office_off.png, and yolov8n.pt are in the same folder.")
    exit()

# Logic Variables
OFF_DELAY = 5 
last_seen_time = 0
light_is_on = False

def log_event(status):
    """Writes the event to a text file with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] Light State: {status}\n")

# Initialize windows
cv2.namedWindow('Webcam Analysis')
cv2.namedWindow('Virtual Office')

print("System Active. Monitoring room...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detection
    results = model(frame, verbose=False)
    person_in_frame = False
    
    for result in results:
        for box in result.boxes:
            # Class 0 is 'person' in YOLO
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                person_in_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_time = time.time()

    # State Switching Logic
    if person_in_frame:
        last_seen_time = current_time
        if not light_is_on:
            light_is_on = True
            log_event("ON")
            winsound.PlaySound("SystemAsterisk", winsound.SND_ASYNC) 
    else:
        if light_is_on and (current_time - last_seen_time > OFF_DELAY):
            light_is_on = False
            log_event("OFF")
            winsound.PlaySound("SystemExclamation", winsound.SND_ASYNC)

    # Display correct image
    display_img = img_on.copy() if light_is_on else img_off.copy()
    
    # Overlay Status Text
    status_text = "STATUS: LIGHTS ON" if light_is_on else "STATUS: LIGHTS OFF"
    color = (0, 255, 0) if light_is_on else (0, 0, 255)
    cv2.putText(display_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if not person_in_frame and light_is_on:
        countdown = int(OFF_DELAY - (current_time - last_seen_time))
        cv2.putText(display_img, f"Closing in {countdown}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Show Windows
    cv2.imshow('Webcam Analysis', frame)
    cv2.imshow('Virtual Office', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()