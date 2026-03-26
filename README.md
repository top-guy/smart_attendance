# 🎓 SmartClass — AI Attendance & Behavior Monitoring

An AI-powered smart attendance and classroom behavior monitoring system built with **DeepFace** for face recognition and **MediaPipe** for real-time computer vision.

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time Face Tracking** | 30fps face detection in the browser using MediaPipe JS |
| **Face Recognition** | Identifies registered students using DeepFace (Facenet) |
| **Two-Phase Sessions** | Phase 1: Take attendance → Phase 2: Monitor behavior |
| **Behavior Detection** | Detects sleeping (eye closure) and distraction (head turned) |
| **Multi-Photo Registration** | Capture 3 photos per student for better accuracy |
| **Video Upload** | Upload recorded class videos for batch attendance processing |
| **Annotated Output Video** | Download processed videos with green bounding boxes + student names |
| **No Duplicate Marking** | Each student is marked present only once per session |
| **Reports & CSV Export** | View and download attendance records |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                    Browser                        │
│  ┌─────────────────┐  ┌────────────────────────┐ │
│  │ MediaPipe JS     │  │ Canvas Overlay         │ │
│  │ • Face Detection │  │ • Bounding boxes (30fps)│ │
│  │ • Face Landmarks │  │ • Names + confidence   │ │
│  └────────┬────────┘  └────────────────────────┘ │
│           │ Cropped face every ~2.5s              │
└───────────┼──────────────────────────────────────┘
            ▼
┌──────────────────────────────────────────────────┐
│                 Flask Server                      │
│  ┌─────────────────┐  ┌────────────────────────┐ │
│  │ DeepFace         │  │ Video Processing       │ │
│  │ • Facenet model  │  │ • Haar cascade detect  │ │
│  │ • Cosine matching│  │ • Annotated output MP4 │ │
│  └─────────────────┘  └────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ Data: students.csv • attendance.csv         │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for live sessions)

### Installation

```bash
git clone https://github.com/top-guy/smart_attendance.git
cd smart_attendance
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## 📖 Usage

### 1. Register Students
- Navigate to **Register Students**
- Enter name, year, department
- Capture **3 face photos** from slightly different angles
- Click Register

### 2. Live Session
- Go to **Live Session** → click **Start Attendance**
- **Phase 1 (Green):** Faces are tracked at 30fps. Names appear after AI recognition (~2-3 seconds). Each student is marked present once.
- **Phase 2 (Blue):** Click **Monitor Behavior** to switch. System detects sleeping and head-turned in real-time.

### 3. Upload Video
- Go to **Upload Video**
- Select a recorded class session (MP4, AVI, MOV)
- System processes the video in the background, identifies students, and logs attendance
- **Download the annotated video** with green boxes and student names

### 4. Reports
- View attendance records in the **Reports** section
- Download as CSV

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Face Detection (Client) | MediaPipe Face Detector JS |
| Behavior Detection (Client) | MediaPipe Face Landmarker JS |
| Face Recognition (Server) | DeepFace with Facenet model |
| Video Processing | OpenCV (Haar Cascade + VideoWriter) |
| Backend | Flask (Python) |
| Data Storage | CSV (pandas) |

## 📁 Project Structure

```
smart_attendance/
├── app.py              # Flask backend (recognition, video processing, API)
├── index.html          # Frontend (MediaPipe JS, UI)
├── requirements.txt    # Python dependencies
├── students.csv        # Registered students database
├── attendance.csv      # Attendance records
├── faces/              # Registered face photos (per student)
├── uploads/            # Temporary uploaded videos
└── outputs/            # Annotated output videos
```

## 📄 License

This project was built as a **Data Science Advanced** class project.
