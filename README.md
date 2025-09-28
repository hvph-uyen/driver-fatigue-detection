# Driver Fatigue Detection and Warning Application

A lightweight machine learning–based system for detecting driver fatigue using steering wheel angle (SWA) and yaw rate (YR) signals.  
It provides real-time alerts via a desktop interface and can be connected to hardware (buzzer/vibration), helping drivers avoid accidents caused by drowsiness.


## Features

- **Non-intrusive monitoring**: Uses existing vehicle signals (SWA, YR), no camera or wearable needed.
- **Lightweight neural network**: Detects 3 states — Awake (A), Drowsy (B), Very Drowsy (C).
- **Entropy-based features**: Computes Approximate Entropy (ApEn) from sensor signals.
- **Real-time alerts**: Changes UI status (`SAFE`, `DANGER`, `SUPERDANGER`) and sends commands to Arduino.
- **Modular design**: Model logic (`model_utils.py`) separated from application logic (`app.py`).


## 📂 Project Structure
```
├── app.py # Application logic (UI + Arduino serial + monitoring loop)
├── model_utils.py # ML utilities (ApEn, model, scaler, prediction)
├── fatigue_model.pth # Trained PyTorch model (not tracked by git if large)
├── scaler.pkl # Scaler for preprocessing
├── data/
│ └── ABC.csv # Example driving data
├── templates/ # HTML templates for UI
│ ├── safe.html
│ ├── danger.html
│ └── superdanger.html
├── documents/ # extended documents of this project
│ └── research_document.pdf
│ └── software_design.pdf
└── requirements.txt 
```

## Installation & Run 

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>

2. **Create & activate virtual environment**

- Linux / macOS:
```
python3 -m venv venv
source venv/bin/activate
```

- Windows PowerShell:
```
Copy code
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```
pip3 install -r requirements.txt
```

4. **Run the App**
```
python3 app.py
```
A desktop window will open showing ***SAFE / DANGER / SUPERDANGER*** screens.

## Demo
[Link to our demo here](https://youtu.be/9SD8_7Hnhpc)

## References
This project was developed for the BOSCH CodeRace Challenge 2025 by Team Jolibee.

See research & design details in `documents` folder:
- Research Document
- Software Design Document
