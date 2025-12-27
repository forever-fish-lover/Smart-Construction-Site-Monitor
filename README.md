# Smart Construction Safety Monitor 👷‍♂️🚧

A real-time AI surveillance system for construction sites detecting PPE violations (No Helmet/Vest) and danger zone intrusions.

## 🌟 Key Features

* **🛡️ PPE Violation Detection**: Automatically detects workers without **Hard Hats** or **Safety Vests**.
* **🧠 Smart Counting Logic**: Uses geometric alignment to merge "Head" and "Body" detections, ensuring accurate violation counts (avoids double-counting the same person).
* **🚧 Geofencing & Intrusion Alert**: Monitors restricted danger zones and triggers alerts when ANY human-like object enters.
* **⏯️ Full Playback Control**: Includes Pause, Replay, and **Time Seeking (Fast Forward/Rewind)** for detailed inspection.

## 🛠️ Tech Stack 
* **Core**: Python 3.x
* **AI Model**: YOLOv8 (Ultralytics) - Custom trained & Logic-enhanced
* **Computer Vision**: OpenCV, CVZone

## 🎮 Controls 

When the system is running, use the following keys to control the video:

| Key | Function | Description |
| :---: | :--- | :--- |
| **[P]** | **Pause / Resume** | Toggle video playback. |
| **[R]** | **Replay** | Restart video from the beginning (resets counters). |
| **[D]** | **5s >>** | **Fast Forward** (Skip 5 seconds). |
| **[A]** | **<< 5s** | **Rewind** (Go back 5 seconds). |
| **[Q]** | **Quit** | Close the application. |
- **Demo Video**: [DEMO](https://youtu.be/EV67g86_x7c)

   
