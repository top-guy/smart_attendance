# SmartClass: AI Attendance & Behavior Monitoring
## Project Summary

**SmartClass** is a hybrid AI system designed to automate classroom attendance and monitor student behavior using advanced computer vision techniques. It combines high-speed client-side face tracking with robust server-side face recognition.

---

## Architecture Overview

The system uses a divided architecture to optimize performance and usability:

### 1. Client-Side (Frontend / Browser)
*   **Technologies:** HTML, CSS, JavaScript (Vanilla).
*   **Face Tracking & Detection:** Runs **MediaPipe Face Detector (JS)** to achieve real-time 30-fps face tracking directly in the browser via the user's webcam.
*   **Behavior Detection:** Uses **MediaPipe Face Landmarker (JS)** to detect 3D facial landmarks and analyze behavior (e.g., sleeping, head turned) without hammering the server.
*   **UI/UX:** A robust, dark-themed Single Page Application (SPA) dashboard that provides real-time visualizations (bounding boxes, green/red overlays, confidence percentages).

### 2. Server-Side (Backend)
*   **Technologies:** Python, Flask, Pandas, OpenCV.
*   **Face Recognition:** Uses **DeepFace** with the **Facenet** model. It computes facial embeddings (vectors) and compares them using Cosine Distance. Only faces that pass a < 0.55 distance threshold are considered a match.
*   **Video Processing:** Uses **OpenCV** with Haar Cascades to find faces in uploaded video files (batch processing).
*   **Data Storage:** 
    *   `students.csv`: Registered student database.
    *   `attendance.csv`: Ongoing session and historical attendance logs.
    *   `faces/`: Directory storing raw registration photos.

---

## Core Features & Workflows

### 1. Student Registration (Multi-Photo)
*   The admin enters the student's name, year, and department.
*   The system uses the webcam to capture **three different face photos**.
*   The photos are sent to the backend, saved to disk, and processed by DeepFace to extract the embeddings.
*   These embeddings are cached in memory for incredibly fast comparisons during live sessions.

### 2. Live Session (Two Phases)
The live session uses a performance-conscious polling strategy:
*   **Phase 1 — Attendance:** MediaPipe tracks faces at 30 frames per second. Every 2.5 seconds, the application crops the active faces from the video feed and sends them to the backend API (`/identify_face`). The backend identifies the faces and marks them as present in `attendance.csv` (avoiding duplicate markings).
*   **Phase 2 — Behavior Monitoring:** The admin can switch the class to monitoring mode. The application switches to using MediaPipe Face Landmarker. It computes geometric rules to flag students who might be distracted:
    *   **Sleeping:** Calculated by analyzing the Eye Aspect Ratio (EAR).
    *   **Head Turned:** Calculated by analyzing the distance between the nose tip and the left/right cheeks.

### 3. Video Upload & Batch Processing
*   The admin can upload a prerecorded class video (`.mp4`, `.avi`, `.mov`) instead of using a live webcam.
*   The backend processes the video asynchronously in a background thread.
*   Because MediaPipe JS is client-side, the backend relies on **OpenCV Haar Cascades** to detect faces every ~2 seconds of video time.
*   Detected faces are identified via DeepFace, logged in `attendance.csv`, and an annotated video with drawn bounding boxes and labels is generated in the `outputs/` folder for download.

### 4. Simulating Large Classes
*   There's a simulation button included to insert fake data simulating a 1000-student lecture hall. It randomly assigns "present", "sleeping", and "head_turned" statuses to stress-test the reports generation.

---

## Data Flow Diagram

1. **Webcam** -> (frames) -> **MediaPipe JS** -> finds Face Bounding Boxes locally.
2. **Browser** -> extracts Face Crops -> sends Base64 imagery to **Flask API**.
3. **Flask API** -> sizes image -> **DeepFace (Facenet)** -> computes 128D Embedding.
4. **Flask API** -> compares against Memory Cache -> returns Match Name & Confidence.
5. **Browser** -> displays localized UI updates (green borders, name tags).
