from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
from datetime import datetime
import random
import base64
import cv2
import numpy as np
import threading
import json
from deepface import DeepFace

app = Flask(__name__)

STUDENTS_CSV = "students.csv"
ATTENDANCE_CSV = "attendance.csv"
FACES_FOLDER = "faces"

os.makedirs(FACES_FOLDER, exist_ok=True)

if not os.path.exists(STUDENTS_CSV):
    pd.DataFrame(columns=["name", "reg_number", "year_entry", "department", "reg_time"]).to_csv(STUDENTS_CSV, index=False)

if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=["timestamp", "name", "status", "behavior"]).to_csv(ATTENDANCE_CSV, index=False)

# ===== Known Faces Registry =====
known_embeddings = {}  # { safe_name: [embedding1, embedding2, ...] }

def load_known_faces():
    """Load all registered face embeddings from disk."""
    print("Loading known faces...")
    for student_name in os.listdir(FACES_FOLDER):
        student_dir = os.path.join(FACES_FOLDER, student_name)
        if os.path.isdir(student_dir):
            embeddings = []
            for fname in sorted(os.listdir(student_dir)):
                if fname.endswith((".jpg", ".png")):
                    face_path = os.path.join(student_dir, fname)
                    try:
                        res = DeepFace.represent(img_path=face_path, model_name="Facenet", enforce_detection=False)
                        embeddings.append(res[0]["embedding"])
                    except Exception as e:
                        print(f"  ⚠️ Skipping {face_path}: {e}")
            if embeddings:
                known_embeddings[student_name] = embeddings
                print(f"  ✅ {student_name}: {len(embeddings)} embedding(s)")
    print(f"Loaded {len(known_embeddings)} students.")

load_known_faces()

# Session tracking
marked_today_cache = set()
session_phase = "idle"

UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# Video processing jobs: { job_id: { status, progress, total_frames, found, errors } }
video_jobs = {}

# Helper: identify a face from an image array
def identify_from_image(face_img):
    """Given a BGR face image (numpy array), return (name, confidence) or ('Unknown', 0)."""
    if face_img is None or face_img.size == 0 or not known_embeddings:
        return "Unknown", 0
    try:
        face_resized = cv2.resize(face_img, (160, 160))
        roi_emb = DeepFace.represent(img_path=face_resized, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        best_name, best_dist = "Unknown", float("inf")
        for student_name, emb_list in known_embeddings.items():
            for known_emb in emb_list:
                sim = np.dot(roi_emb, known_emb) / (np.linalg.norm(roi_emb) * np.linalg.norm(known_emb))
                cosine_dist = 1 - sim
                if cosine_dist < best_dist:
                    best_dist = cosine_dist
                    best_name = student_name
        if best_dist < 0.55:
            return best_name, int((1 - best_dist) * 100)
    except:
        pass
    return "Unknown", 0

# ===== Routes =====

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    year = request.form.get("year")
    dept_code = request.form.get("dept_code")
    # Accept multiple photos: image_0, image_1, image_2
    photos = []
    for i in range(5):
        img = request.form.get(f"image_{i}")
        if img:
            photos.append(img)
    # Fallback: single "image" field
    if not photos:
        img = request.form.get("image")
        if img:
            photos.append(img)

    if not name or not year or not dept_code:
        return jsonify({"success": False, "message": "All fields required"})
    if not photos:
        return jsonify({"success": False, "message": "At least one face photo required"})

    df = pd.read_csv(STUDENTS_CSV)

    # Generate unique reg number
    year_short = str(year)[-2:]
    base = f"{year_short}{dept_code}"
    reg_number = None
    for _ in range(50):
        candidate = base + f"{random.randint(1000, 9999)}"
        if candidate not in df["reg_number"].values:
            reg_number = candidate
            break
    if not reg_number:
        return jsonify({"success": False, "message": "Could not generate unique reg number"})

    safe_name = name.replace(' ', '_').replace("'", "").replace('"', "")
    student_folder = os.path.join(FACES_FOLDER, safe_name)
    os.makedirs(student_folder, exist_ok=True)

    embeddings = []
    for idx, photo_b64 in enumerate(photos):
        if "," in photo_b64:
            image_data = photo_b64.split(",")[1]
        else:
            image_data = photo_b64
        image_bytes = base64.b64decode(image_data)
        face_path = os.path.join(student_folder, f"face_{idx}.jpg")
        with open(face_path, "wb") as f:
            f.write(image_bytes)

        # Compute embedding
        try:
            res = DeepFace.represent(img_path=face_path, model_name="Facenet", enforce_detection=False)
            embeddings.append(res[0]["embedding"])
        except Exception as e:
            print(f"⚠️ Embedding error for photo {idx}: {e}")

    if embeddings:
        known_embeddings[safe_name] = embeddings
        print(f"✅ {name}: {len(embeddings)} embeddings registered")
    else:
        print(f"⚠️ {name}: registered but no embeddings computed")

    new_row = {
        "name": name, "reg_number": reg_number,
        "year_entry": year, "department": dept_code,
        "reg_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(STUDENTS_CSV, index=False)

    return jsonify({
        "success": True,
        "message": f"{name} registered with {len(embeddings)} face photo(s)",
        "reg_number": reg_number,
        "photos_saved": len(photos),
        "embeddings_computed": len(embeddings)
    })

@app.route("/identify_face", methods=["POST"])
def identify_face():
    """Accept a cropped face image, return the best matching student name."""
    try:
        data = request.json
        face_b64 = data.get("face")
        mode = data.get("mode", "attendance")
        if not face_b64 or not known_embeddings:
            return jsonify({"name": "Unknown", "confidence": 0})

        # Decode the face crop
        if "," in face_b64:
            face_b64 = face_b64.split(",")[1]
        img_bytes = base64.b64decode(face_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None or face_img.size == 0:
            return jsonify({"name": "Unknown", "confidence": 0})

        # Preprocess: resize to a standard size for consistent embeddings
        face_img = cv2.resize(face_img, (160, 160))

        # Compute embedding
        roi_emb = DeepFace.represent(
            img_path=face_img, model_name="Facenet", enforce_detection=False
        )[0]["embedding"]

        # Compare against all known embeddings (multi-photo: use best match)
        best_name = "Unknown"
        best_dist = float("inf")

        for student_name, emb_list in known_embeddings.items():
            for known_emb in emb_list:
                sim = np.dot(roi_emb, known_emb) / (
                    np.linalg.norm(roi_emb) * np.linalg.norm(known_emb)
                )
                cosine_dist = 1 - sim
                if cosine_dist < best_dist:
                    best_dist = cosine_dist
                    best_name = student_name

        print(f"🔍 Best: {best_name}, dist: {best_dist:.4f}")

        if best_dist < 0.55:
            confidence = int((1 - best_dist) * 100)
            display_name = best_name.replace('_', ' ')

            # Mark attendance if in attendance mode
            if mode == "attendance":
                today = datetime.now().strftime("%Y-%m-%d")
                cache_key = f"{today}_{best_name}"
                already_marked = cache_key in marked_today_cache
                if not already_marked:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    att_df = pd.read_csv(ATTENDANCE_CSV)
                    new_row = {"timestamp": timestamp, "name": display_name, "status": "present", "behavior": "normal"}
                    att_df = pd.concat([att_df, pd.DataFrame([new_row])], ignore_index=True)
                    att_df.to_csv(ATTENDANCE_CSV, index=False)
                    marked_today_cache.add(cache_key)
                    print(f"✅ Marked: {display_name}")
                return jsonify({
                    "name": display_name, "confidence": confidence,
                    "already_marked": already_marked
                })
            else:
                return jsonify({"name": display_name, "confidence": confidence})
        else:
            return jsonify({"name": "Unknown", "confidence": 0})

    except Exception as e:
        print(f"❌ identify error: {e}")
        return jsonify({"name": "Unknown", "confidence": 0})

@app.route("/reset_session", methods=["POST"])
def reset_session():
    global session_phase
    marked_today_cache.clear()
    session_phase = "attendance"
    return jsonify({"success": True, "message": "Session started."})

@app.route("/switch_to_monitoring", methods=["POST"])
def switch_to_monitoring():
    global session_phase
    session_phase = "monitoring"
    return jsonify({"success": True, "marked_count": len(marked_today_cache)})

@app.route("/get_session_stats")
def get_session_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    today_marks = [k for k in marked_today_cache if k.startswith(today)]
    df_students = pd.read_csv(STUDENTS_CSV)
    return jsonify({
        "total_registered": len(df_students),
        "marked_today": len(today_marks),
        "marked_names": [k.split("_", 1)[1].replace("_", " ") for k in today_marks],
        "phase": session_phase
    })

@app.route("/get_students")
def get_students():
    df = pd.read_csv(STUDENTS_CSV)
    return jsonify({"students": df.to_dict("records")})

@app.route("/get_reports")
def get_reports():
    df = pd.read_csv(ATTENDANCE_CSV)
    return jsonify({"records": df.to_dict("records")})

@app.route("/download_csv")
def download_csv():
    return send_file(ATTENDANCE_CSV, as_attachment=True,
                     download_name=f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")

@app.route("/simulate_large_class", methods=["POST"])
def simulate_large_class():
    df_students = pd.read_csv(STUDENTS_CSV)
    att_df = pd.read_csv(ATTENDANCE_CSV)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i in range(1000):
        name = random.choice(df_students["name"].tolist()) if len(df_students) > 0 and random.random() < 0.7 else f"Sim_Student_{i+1}"
        behavior = random.choices(["normal", "sleeping", "head_turned"], weights=[80, 10, 10])[0]
        rows.append({"timestamp": timestamp, "name": name, "status": "present", "behavior": behavior})
    att_df = pd.concat([att_df, pd.DataFrame(rows)], ignore_index=True)
    att_df.to_csv(ATTENDANCE_CSV, index=False)
    return jsonify({"marked": 1000, "message": "1000-student hall simulated"})

# ===== Video Upload Processing =====

OUTPUTS_FOLDER = "outputs"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_video_thread(job_id, video_path):
    """Background thread: process video, annotate faces, write output video."""
    job = video_jobs[job_id]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        job["status"] = "error"
        job["error"] = "Could not open video"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    sample_interval = int(fps * 2)  # run recognition every 2 seconds

    job["total_frames"] = total_frames
    job["duration"] = round(duration, 1)
    job["status"] = "processing"

    # Setup output video writer
    output_path = os.path.join(OUTPUTS_FOLDER, f"{job_id}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    job["output_video"] = f"{job_id}_annotated.mp4"

    found_students = set()
    # Cache of current face labels: [(x, y, w, h, name, confidence), ...]
    current_labels = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            job["progress"] = round((frame_idx / max(total_frames, 1)) * 100, 1)

            # Detect faces and identify
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            new_labels = []

            for (fx, fy, fw, fh) in faces:
                pad = int(max(fw, fh) * 0.2)
                x1 = max(0, fx - pad)
                y1 = max(0, fy - pad)
                x2 = min(frame_w, fx + fw + pad)
                y2 = min(frame_h, fy + fh + pad)
                face_crop = frame[y1:y2, x1:x2]

                name, confidence = identify_from_image(face_crop)
                display_name = name.replace('_', ' ') if name != "Unknown" else "Unknown"
                new_labels.append((fx, fy, fw, fh, display_name, confidence))

                if name != "Unknown" and name not in found_students:
                    found_students.add(name)
                    job["found"].append({"name": display_name, "confidence": confidence})
                    # Log to CSV
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    att_df = pd.read_csv(ATTENDANCE_CSV)
                    new_row = {"timestamp": timestamp, "name": display_name, "status": "present", "behavior": "normal"}
                    att_df = pd.concat([att_df, pd.DataFrame([new_row])], ignore_index=True)
                    att_df.to_csv(ATTENDANCE_CSV, index=False)
                    print(f"📹 Video: marked {display_name} ({confidence}%)")

            current_labels = new_labels

        # Draw cached annotations on EVERY frame for smooth video
        for (fx, fy, fw, fh, name, conf) in current_labels:
            is_known = name != "Unknown"
            color = (0, 230, 118) if is_known else (80, 80, 255)  # green or red (BGR)

            # Box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

            # Label background
            label = f"{name} ({conf}%)" if is_known else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (fx, fy - th - 10), (fx + tw + 10, fy), color, -1)
            text_color = (15, 15, 23) if is_known else (255, 255, 255)
            cv2.putText(frame, label, (fx + 5, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Marked badge for known students
            if is_known:
                badge = "PRESENT"
                cv2.putText(frame, badge, (fx + 5, fy + fh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    job["status"] = "done"
    job["progress"] = 100
    print(f"📹 Video complete: {len(found_students)} students, output: {output_path}")

    # Clean up input video (keep output)
    try:
        os.remove(video_path)
    except:
        pass

@app.route("/process_video", methods=["POST"])
def process_video():
    """Upload a video file for batch attendance processing."""
    if 'video' not in request.files:
        return jsonify({"success": False, "message": "No video file uploaded"})
    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({"success": False, "message": "Empty filename"})

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(random.randint(1000, 9999))
    ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    video_path = os.path.join(UPLOADS_FOLDER, f"{job_id}{ext}")
    video_file.save(video_path)

    video_jobs[job_id] = {
        "status": "starting", "progress": 0,
        "total_frames": 0, "duration": 0,
        "found": [], "error": None, "output_video": None
    }

    thread = threading.Thread(target=process_video_thread, args=(job_id, video_path), daemon=True)
    thread.start()
    return jsonify({"success": True, "job_id": job_id})

@app.route("/video_status/<job_id>")
def video_status(job_id):
    job = video_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"})
    return jsonify(job)

@app.route("/download_output_video/<filename>")
def download_output_video(filename):
    """Download the annotated output video."""
    path = os.path.join(OUTPUTS_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"smartclass_{filename}")
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    print("🚀 SmartClass v7 – Hybrid AI Attendance & Monitoring")
    print("Open → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)