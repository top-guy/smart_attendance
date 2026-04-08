# Flask: The core web framework for handling HTTP requests and serving the dashboard
from flask import Flask, request, jsonify, send_file
# Pandas: Used for managing our CSV-based 'database' for students and attendance
import pandas as pd
# OS: Native library for file directory management and path handling
import os
# Datetime: used to timestamp each attendance record
from datetime import datetime
# Random: generates unique registration numbers (IDs) for students
import random
# Base64: decodes image data sent from the web browser's camera
import base64
# OpenCV: THE computer vision library used for image filters, resizing, and video drawing
import cv2
# Numpy: used for high-performance vector math (Cosine Similarity calculations)
import numpy as np
# Threading: allows video processing to run in the background without freezing the UI
import threading
# JSON: formats the data sent back to the dashboard
import json
# DeepFace: The AI engine that converts face images into 512-dimensional vectors
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

# ===== Global Face Registry (RAM Cache) =====
# This dictionary stores face signatures so the AI doesn't have to re-scan files every time
known_embeddings = {}  # Format: { 'jonah_ishaku': [embedding1, embedding2], ... }

def load_known_faces():
    """
    Scans the 'faces' folder, reads student photos, and calculates their 512-dim AI signatures.
    This runs once at server startup to prime the recognition engine.
    """
    print("Loading known faces...")
    # Loop through every student's folder inside 'faces/'
    for student_name in os.listdir(FACES_FOLDER):
        student_dir = os.path.join(FACES_FOLDER, student_name)
        if os.path.isdir(student_dir):
            embeddings = []
            # Find and process every image file found in the student's directory
            for fname in sorted(os.listdir(student_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    face_path = os.path.join(student_dir, fname)
                    try:
                        # DeepFace.represent: Extracts the mathematical 'face signature' (embedding)
                        # We use Facenet512 for superior accuracy in distinguishing similar faces.
                        res = DeepFace.represent(img_path=face_path, model_name="Facenet512", enforce_detection=False)
                        embeddings.append(res[0]["embedding"])
                    except Exception as e:
                        # Skip files that aren't valid faces or are corrupted
                        print(f"  [SKIP] {face_path}: {str(e).encode('ascii', 'replace').decode()}")
            
            # If we found signatures, keep them in our fast RAM registry
            if embeddings:
                known_embeddings[student_name] = embeddings
                print(f"  [OK] {student_name}: {len(embeddings)} embedding(s)")
    print(f"Loaded {len(known_embeddings)} students into the AI engine.")

load_known_faces()

# Session tracking
marked_today_cache = set()
session_phase = "idle"

UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# Video processing jobs: { job_id: { status, progress, total_frames, found, errors } }
video_jobs = {}

# Helper: Core recognition algorithm
def identify_from_image(face_img, threshold=0.45):
    """
    Takes an image (numpy array), normalizes it, calculates its vector, 
    and checks it against the registry using Cosine Similarity vector math.
    """
    # 1. Validation: ensure the image exists and we have students to compare to
    if face_img is None or face_img.size == 0 or not known_embeddings:
        return "Unknown", 0
    try:
        # 2. Lighting Normalization (CLAHE):
        # This levels out the contrast to handle shadows and glares across the classroom.
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB) # Convert color space to LAB (Luminance, A, B)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l) # Normalize the L channel (brightness)
        face_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR) # Merge back to normal Colors

        # 3. Resize: Standardize the image to 160x160 as expected by the AI model's architecture
        face_resized = cv2.resize(face_img, (160, 160))
        
        # 4. Feature Extraction: Get the 512-dimensional mathematical vector for this specific crop
        roi_emb = DeepFace.represent(img_path=face_resized, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
        
        best_name, best_dist = "Unknown", float("inf")
        # 5. Matching Loop: Check this new vector against EVERY vector in our database
        for student_name, emb_list in known_embeddings.items():
            for known_emb in emb_list:
                # Math Phase: Calculate Cosine Similarity (the angle between vectors)
                sim = np.dot(roi_emb, known_emb) / (np.linalg.norm(roi_emb) * np.linalg.norm(known_emb))
                cosine_dist = 1 - sim # Lower distance = Closer match (0.0 is a perfect match)
                
                if cosine_dist < best_dist:
                    best_dist = cosine_dist
                    best_name = student_name
                    
        # Debugging log: prints to console so you can see the math behind the match
        print(f"[MATCH] Best: {best_name}, dist: {best_dist:.4f}, threshold: {threshold}")
        
        # 6. Final Decision: Is the closest match close ENOUGH to be sure?
        if best_dist < threshold:
            # We convert the math distance into a 'Confidence %' for the user display
            return best_name, int((1 - (best_dist / 0.8)) * 100)
    except Exception as e:
        print(f"[ERROR] identify_from_image: {e}")
    return "Unknown", 0

# ===== Routes =====

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/register", methods=["POST"])
def register():
    """
    API endpoint to register a new student. 
    It saves metadata to a CSV and stores multiple face photos for the AI.
    """
    # Get form data from the web dashboard
    name = request.form.get("name")
    year = request.form.get("year")
    dept_code = request.form.get("dept_code")
    
    # Extract up to 5 photos from the multifile upload
    photos = []
    for i in range(5):
        img = request.form.get(f"image_{i}")
        if img:
            photos.append(img)
            
    # Error checking for missing inputs
    if not name or not year or not dept_code:
        return jsonify({"success": False, "message": "All fields required"})
    if not photos:
        return jsonify({"success": False, "message": "At least one face photo required"})

    df = pd.read_csv(STUDENTS_CSV)

    # ALGORITHM: Generate a unique Registration Number
    # Combines Year Code + Dept Code + 4 random digits (e.g., 23CSC8921)
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

    # Create a safe folder name for the student (removes potential breaking characters like quotes)
    safe_name = name.replace(' ', '_').replace("'", "").replace('"', "")
    student_folder = os.path.join(FACES_FOLDER, safe_name)
    os.makedirs(student_folder, exist_ok=True)

    embeddings = []
    # Loop through each photo provided during registration
    for idx, photo_b64 in enumerate(photos):
        # Decodes the Base64 image data sent from the browser
        if "," in photo_b64:
            image_data = photo_b64.split(",")[1]
        else:
            image_data = photo_b64
        image_bytes = base64.b64decode(image_data)
        
        # Save the physical photo to the 'faces/' directory
        face_path = os.path.join(student_folder, f"face_{idx}.jpg")
        with open(face_path, "wb") as f:
            f.write(image_bytes)

        # AI TRAINING: Extract the vector signature for this newly saved photo
        try:
            # We use Facenet512 to ensure the registration matches the identification engine
            res = DeepFace.represent(img_path=face_path, model_name="Facenet512", enforce_detection=False)
            embeddings.append(res[0]["embedding"])
        except Exception as e:
            print(f"[WARN] Embedding error for photo {idx}: {e}")

    # If successful, add the student record to the master CSV file
    if embeddings:
        known_embeddings[safe_name] = embeddings
        print(f"[OK] {name}: {len(embeddings)} embeddings registered")
    else:
        print(f"[WARN] {name}: registered but no embeddings computed")

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
    """
    Main identification engine for the Live Feed.
    Receives a Base64 image crop from the browser and finds the matching student.
    """
    try:
        data = request.json
        face_b64 = data.get("face") # The raw image data (Base64)
        mode = data.get("mode", "attendance")
        
        # 1. Validation: ensure data exists
        if not face_b64 or not known_embeddings:
            return jsonify({"name": "Unknown", "confidence": 0})

        # 2. Decoding: Convert the Base64 string into a raw OpenCV image (numpy array)
        if "," in face_b64:
            face_b64 = face_b64.split(",")[1]
        img_bytes = base64.b64decode(face_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None or face_img.size == 0:
            return jsonify({"name": "Unknown", "confidence": 0})

        # 3. Preprocessing: Standarize scale for the AI model
        face_img = cv2.resize(face_img, (160, 160))

        # 4. Feature Extraction: Use Facenet512 (512-dimensional vector)
        # This MUST match the 512 dimensions used in registration.
        roi_emb = DeepFace.represent(
            img_path=face_img, model_name="Facenet512", enforce_detection=False
        )[0]["embedding"]

        # 5. Math Comparison: Find the student whose registration signature is closest
        best_name = "Unknown"
        best_dist = float("inf")

        for student_name, emb_list in known_embeddings.items():
            for known_emb in emb_list:
                # Cosine Distance: Math angle between the live vector and registered vector
                sim = np.dot(roi_emb, known_emb) / (
                    np.linalg.norm(roi_emb) * np.linalg.norm(known_emb)
                )
                cosine_dist = 1 - sim
                if cosine_dist < best_dist:
                    best_dist = cosine_dist
                    best_name = student_name

        print(f"[MATCH] Best: {best_name}, dist: {best_dist:.4f}")

        # 6. Decision Threshold: Standardized to 0.45 for balanced accuracy
        if best_dist < 0.45:
            confidence = int((1 - (best_dist / 0.8)) * 100)
            display_name = best_name.replace('_', ' ')

            # 7. Persistence: If in Attendance Mode, save to CSV log
            if mode == "attendance":
                today = datetime.now().strftime("%Y-%m-%d")
                cache_key = f"{today}_{best_name}"
                # Prevent duplicate marking for the same student on the same day
                already_marked = cache_key in marked_today_cache
                if not already_marked:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    att_df = pd.read_csv(ATTENDANCE_CSV)
                    new_row = {"timestamp": timestamp, "name": display_name, "status": "present", "behavior": "normal"}
                    att_df = pd.concat([att_df, pd.DataFrame([new_row])], ignore_index=True)
                    att_df.to_csv(ATTENDANCE_CSV, index=False)
                    marked_today_cache.add(cache_key) # Track in RAM cache
                    print(f"[MARKED] {display_name}")
                return jsonify({
                    "name": display_name, "confidence": confidence,
                    "already_marked": already_marked
                })
            else:
                return jsonify({"name": display_name, "confidence": confidence})
        else:
            return jsonify({"name": "Unknown", "confidence": 0})

    except Exception as e:
        print(f"[ERROR] identify error: {e}")
        return jsonify({"name": "Unknown", "confidence": 0})

@app.route("/reset_session", methods=["POST"])
def reset_session():
    """Resets the day's attendance cache to start a fresh tracking session."""
    global session_phase
    marked_today_cache.clear()
    session_phase = "attendance"
    return jsonify({"success": True, "message": "Session started."})

@app.route("/switch_to_monitoring", methods=["POST"])
def switch_to_monitoring():
    """Transitions the UI from Attendance phase to Behavior Monitoring phase."""
    global session_phase
    session_phase = "monitoring"
    return jsonify({"success": True, "marked_count": len(marked_today_cache)})

@app.route("/get_session_stats")
def get_session_stats():
    """Calculates and returns the Current Session statistics for the sidebar display."""
    today = datetime.now().strftime("%Y-%m-%d")
    today_marks = [k for k in marked_today_cache if k.startswith(today)]
    df_students = pd.read_csv(STUDENTS_CSV)
    # Returns total count, attendance count, and the list of student names present
    return jsonify({
        "total_registered": len(df_students),
        "marked_today": len(today_marks),
        "marked_names": [k.split("_", 1)[1].replace("_", " ") for k in today_marks],
        "phase": session_phase
    })

@app.route("/get_students")
def get_students():
    """Retrieves high-level metadata for all students to display in the management table."""
    df = pd.read_csv(STUDENTS_CSV)
    return jsonify({"students": df.to_dict("records")})

@app.route("/delete_student", methods=["POST"])
def delete_student():
    """
    Utility to remove a student from the system. 
    Cleans up their database record, image files, and AI RAM cache.
    """
    data = request.json
    reg_number = data.get("reg_number")
    if not reg_number:
        return jsonify({"success": False, "message": "No reg_number provided"})

    df = pd.read_csv(STUDENTS_CSV)
    student_rows = df[df["reg_number"] == reg_number]
    if student_rows.empty:
        return jsonify({"success": False, "message": "Student not found"})

    student_name = student_rows.iloc[0]["name"]
    # We use safe_name to reference the directory without breaking on spaces
    safe_name = student_name.replace(' ', '_').replace("'", "").replace('"', "")

    # 1. Remove the record from the master CSV file
    df = df[df["reg_number"] != reg_number]
    df.to_csv(STUDENTS_CSV, index=False)

    # 2. Delete the physical folder containing the face photos
    student_folder = os.path.join(FACES_FOLDER, safe_name)
    if os.path.isdir(student_folder):
        import shutil
        shutil.rmtree(student_folder)

    # 3. Purge from our active RAM cache so recognition stops immediately
    if safe_name in known_embeddings:
        del known_embeddings[safe_name]

    print(f"[DELETE] {student_name} ({reg_number}) removed from registry.")
    return jsonify({"success": True, "message": f"{student_name} deleted successfully"})

@app.route("/get_reports")
def get_reports():
    """Returns the entire history of attendance logs as JSON for the dashboard table."""
    df = pd.read_csv(ATTENDANCE_CSV)
    return jsonify({"records": df.to_dict("records")})

@app.route("/download_csv")
def download_csv():
    """Allows the user to download the attendance report as an Excel-ready .csv file."""
    return send_file(ATTENDANCE_CSV, as_attachment=True,
                     download_name=f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")

@app.route("/simulate_large_class", methods=["POST"])
def simulate_large_class():
    """
    Developer Feature: Simulates a filled lecture hall with 1,000 students.
    Used for stress-testing UI rendering and performance.
    """
    df_students = pd.read_csv(STUDENTS_CSV)
    att_df = pd.read_csv(ATTENDANCE_CSV)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i in range(1000):
        # Assign a random registered student or a simulated name
        name = random.choice(df_students["name"].tolist()) if len(df_students) > 0 and random.random() < 0.7 else f"Sim_Student_{i+1}"
        behavior = random.choices(["normal", "sleeping", "head_turned"], weights=[80, 10, 10])[0]
        rows.append({"timestamp": timestamp, "name": name, "status": "present", "behavior": behavior})
    att_df = pd.concat([att_df, pd.DataFrame(rows)], ignore_index=True)
    att_df.to_csv(ATTENDANCE_CSV, index=False)
    return jsonify({"marked": 1000, "message": "1000-student hall simulated"})

# ------------------------------------------------------------------------------
# THE VIDEO AI ENGINE: Handles high-performance face detection in offline video files
# ------------------------------------------------------------------------------

OUTPUTS_FOLDER = "outputs"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# These files (deploy.prototxt and res10...caffemodel) define the DNN Face Detector
_DNN_PROTO = os.path.join(os.path.dirname(__file__), "models", "deploy.prototxt")
_DNN_MODEL = os.path.join(os.path.dirname(__file__), "models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def _get_dnn_detector():
    """
    Loads the Caffe-based SSD (Single Shot Detector) face model.
    SSD is significantly more robust than traditional 'Haar Cascades' for video.
    """
    if os.path.exists(_DNN_PROTO) and os.path.exists(_DNN_MODEL):
        try:
            net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
            print("[INIT] Using high-performance OpenCV DNN face detector")
            return net
        except Exception as e:
            print(f"[WARN] DNN load failed: {e}")
    print("[INIT] Falling back to standard Haar Cascade (Lesser Accuracy)")
    return None

dnn_net = _get_dnn_detector()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_frame(frame):
    """
    Performs face detection using the DNN network. 
    It scans the image and returns a list of bounding boxes (x, y, w, h).
    """
    h, w = frame.shape[:2]
    if dnn_net is not None:
        # 1. Image to Blob: Prepares the image for the Neural Network (resizes to 300x300)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        dnn_net.setInput(blob)
        # 2. Forward Pass: Runs the AI calculation to find faces
        detections = dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            # Only count detections above 35% confidence (tuned for fast detection)
            if conf < 0.35: continue
            
            # Map relative coordinates (0-1) to actual pixel values (0-width/height)
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Bounds clipping (ensure boxes don't go outside the image)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            
            # Size Filtering: Eliminate small background noise that looks like a face
            if fw > 45 and fh > 45:
                faces.append((x1, y1, fw, fh))
        return faces
    else:
        # Fallback to the older CV2 detection logic if the DNN models are missing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        return list(faces) if len(faces) > 0 else []

def process_video_thread(job_id, video_path):
    """
    HEAVY LIFTING: Background thread that processes an entire lecture video.
    It performs detection, identification, and draws overlays frame-by-frame.
    """
    job = video_jobs[job_id]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        job["status"] = "error"
        job["error"] = "Media error: Could not open video file."
        return

    # 1. Extract Video Properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_interval = int(fps * 2)  # Optimization: Only run expensive face recognition every 2 seconds

    job["total_frames"] = total_frames
    job["status"] = "processing"

    # 2. Setup Output Stream (Writes the annotated video to the 'outputs/' folder)
    output_path = os.path.join(OUTPUTS_FOLDER, f"{job_id}_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    job["output_video"] = f"{job_id}_annotated.mp4"

    found_students = set()
    current_labels = [] # Prevents flickering by caching labels between AI samples
    frame_idx = 0
    VIDEO_THRESHOLD = 0.45 # Standardized threshold for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 3. Sampling: Every 2 seconds, run the full AI identification pipeline
        if frame_idx % sample_interval == 0:
            job["progress"] = round((frame_idx / max(total_frames, 1)) * 100, 1)
            faces = detect_faces_in_frame(frame)
            new_labels = []

            for (fx, fy, fw, fh) in faces:
                # Add 15% padding so the identification engine has more facial context
                pad = int(max(fw, fh) * 0.15)
                x1, y1 = max(0, fx-pad), max(0, fy-pad)
                x2, y2 = min(frame_w, fx+fw+pad), min(frame_h, fy+fh+pad)
                face_crop = frame[y1:y2, x1:x2]

                # Run the 512-dim AI recognition helper
                name, confidence = identify_from_image(face_crop, threshold=VIDEO_THRESHOLD)
                display_name = name.replace('_', ' ')
                new_labels.append((fx, fy, fw, fh, display_name, confidence))

                # If we recognize a student, mark them present in the permanent records
                if name != "Unknown" and name not in found_students:
                    found_students.add(name)
                    job["found"].append({"name": display_name, "confidence": confidence})
                    att_df = pd.read_csv(ATTENDANCE_CSV)
                    new_row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "name": display_name, "status": "present", "behavior": "normal"}
                    pd.concat([att_df, pd.DataFrame([new_row])], ignore_index=True).to_csv(ATTENDANCE_CSV, index=False)
            
            current_labels = new_labels

        # 4. Rendering: Draw the AI boxes and text on EVERY frame for a smooth result
        for (fx, fy, fw, fh, name, conf) in current_labels:
            color = (0, 230, 118) if name != "Unknown" else (80, 80, 255) # Green=Found, Red=Unknown
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
            label = f"{name} ({conf}%)"
            cv2.putText(frame, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    # Cleanup resources once video is done
    cap.release()
    out.release()
    job["status"] = "done"
    job["progress"] = 100
    try: os.remove(video_path) # Delete raw upload to save disk space
    except: pass

@app.route("/process_video", methods=["POST"])
def process_video():
    """Triggers the background video processing thread."""
    if 'video' not in request.files: return jsonify({"success": False})
    video_file = request.files['video']
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(UPLOADS_FOLDER, f"{job_id}.mp4")
    video_file.save(video_path)

    # Initialize the job tracker entry
    video_jobs[job_id] = {"status": "starting", "progress": 0, "found": []}
    
    # Launch in a background thread so the user doesn't have to wait for the page to load
    threading.Thread(target=process_video_thread, args=(job_id, video_path), daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})

@app.route("/video_status/<job_id>")
def video_status(job_id):
    """Returns the current processing % and found students for a video job."""
    return jsonify(video_jobs.get(job_id, {"status": "not_found"}))

@app.route("/download_output_video/<filename>")
def download_output_video(filename):
    """Sends the final annotated .mp4 file to the user's browser."""
    path = os.path.join(OUTPUTS_FOLDER, filename)
    if os.path.exists(path): return send_file(path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

# ==============================================================================
# ENTRY POINT: Start the AI Attendance Backend
# ==============================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print(" SMARTCLASS v7 AI ATTENDANCE ENGINE LOADED.       ")
    print(" Dashboard: http://127.0.0.1:5000                 ")
    print("--------------------------------------------------")
    app.run(debug=True, port=5000)