🏠 Smart Room: AI-Powered Lighting & Automation
An intelligent room automation system that uses computer vision to monitor human presence and manage office lighting. The system detects inhabitants in real-time, manages lighting states with a delayed-off feature to save energy, and provides auditory feedback.

🌟 Key Features
Real-Time Human Detection: Uses a lightweight YOLOv8 nano model for high-speed person tracking.

Smart State Logic: Includes a 5-second "Auto-Off" delay to prevent lights flickering when someone momentarily leaves the frame.

Dynamic Visuals: Dual-window interface showing the raw AI analysis and a virtual office visualization.

Event Logging: Automatically records every "ON/OFF" event with timestamps in room_log.txt.

Auditory Alerts: Integrated Windows system sounds for state changes.

🛠️ Tech Stack
Language: Python 3.x

AI Model: Ultralytics YOLOv8

Computer Vision: OpenCV

OS Integration: winsound for alerts, os for dynamic pathing

🚀 Getting Started
Prerequisites
Install the required libraries:

Bash

pip install ultralytics opencv-python

Installation & Usage
Clone this repository or download the folder.

Ensure your images (office_on.jpg, office_off.png) and model (yolov8n.pt) are in the same directory as the script.

Run the application:

Bash

python smart_room.py

Press 'q' to exit the system.

📁 Project Structure
Plaintext

smart_room/
├── smart_room.py      # Main application logic
├── yolov8n.pt         # Pre-trained AI model weights
├── office_on.jpg      # Virtual office 'ON' state image
├── office_off.png     # Virtual office 'OFF' state image
└── room_log.txt       # Generated event history

📝 Future Scope
Hardware Integration: Link the script to an Arduino/ESP32 to control actual physical lights via Relay.

Multi-Person Logic: Adjust brightness or fan speed based on the number of people detected.

Energy Dashboard: A web-based UI to visualize energy savings based on the log file.
