import os
import cv2
import numpy as np
import qrcode
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector

app = Flask(__name__)
app.secret_key = 'oral_cancer_secret_key_prod'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------------------------------------
# 1. DATABASE CONNECTION HELPER
# -------------------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="",
        password="",           # Add your MySQL password if set
        database="oral_cancer_db"
    )

# -------------------------------------------------------------
# 2. MODEL DEFINITION & LOADING (MobileNetV2)
# -------------------------------------------------------------
def load_pytorch_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model_path = 'oral_cancer_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_pytorch_model()

# -------------------------------------------------------------
# 3. OPENCV BLUR DETECTION & IMAGE VALIDATION
# -------------------------------------------------------------
def is_image_blurry(image_path, threshold=15.0): # Lowered from 80.0 to 15.0
    """
    Detects motion blur or out-of-focus images using Laplacian Variance.
    Returns True if score < threshold (blurry/shaky).
    """
    img = cv2.imread(image_path)
    if img is None:
        return True, 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, round(score, 2)
def validate_oral_cavity_image(image_path):
    """
    Stricter HSV validation to verify the image contains inner oral cavity tissue.
    Filters out non-oral objects, clothing, faces, or random scenes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Oral mucosal tissue HSV ranges (Pinkish / Crimson / Deep Red)
    lower_red1 = np.array([0, 60, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([155, 60, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    tissue_mask = mask1 | mask2

    total_pixels = img.shape[0] * img.shape[1]
    tissue_ratio = np.sum(tissue_mask > 0) / total_pixels

    # Oral cavity photos must have at least 25% mucosal red/pink coverage
    if tissue_ratio < 0.25:
        return False

    # Check color saturation (prevents flat red clothing or solid red objects)
    mean_saturation = np.mean(hsv[:, :, 1])
    if mean_saturation < 40 or mean_saturation > 230:
        return False

    return True

def predict_image(image_path):
    # 🔍 STEP A: Check for Blur / Camera Shake (Threshold set to 15.0 for oral tissue)
    blurry, score = is_image_blurry(image_path, threshold=15.0)
    print(f"\n[DEBUG] Image Sharpness Score: {score}\n")
    
    if blurry:
        return "Blurry Image"

    # ... rest of your validation logic
   

    # 🔍 STEP B: Check for Oral Cavity Color/Tissue Criteria
    if not validate_oral_cavity_image(image_path):
        return "Invalid Image"

    # 🚀 STEP C: PyTorch AI Model Inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # 🎯 CONFIDENCE GATEKEEPER:
    # If confidence is below 85% (0.85), treat as non-oral / invalid image
    if confidence.item() < 0.85:
        return "Invalid Image"

    # Index 0: cancer, Index 1: non_cancer
    return "Cancer Detected" if predicted.item() == 0 else "Clear"
# -------------------------------------------------------------
# 4. FLASK ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        hashed = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed))
            conn.commit()
            cursor.close()
            conn.close()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"\n[!] MYSQL REGISTRATION ERROR: {e}\n")
            flash('Username already exists or database error.', 'danger')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Welcome back!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.', 'danger')
        except Exception as e:
            print(f"\n[!] MYSQL LOGIN ERROR: {e}\n")
            flash('Database connection failed. Check your terminal for details.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM scans WHERE user_id = %s ORDER BY scan_date DESC", (session['user_id'],))
        scans = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"\n[!] MYSQL DASHBOARD ERROR: {e}\n")
        scans = []

    recent_scan = scans[0] if scans else None
    return render_template('dashboard.html', scans=scans, recent_scan=recent_scan)
@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = request.form['patient_name']
        email = request.form['email']
        file = request.files.get('file')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform pre-screening checks & prediction
            result = predict_image(filepath)

            # -------------------------------------------------------------
            # 1. HANDLE BLURRY / SHAKY IMAGE REJECTION
            # -------------------------------------------------------------
            if result == "Blurry Image":
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash('⚠️ Image is too blurry or shaky. Please keep your hand steady and upload a focused photo.', 'danger')
                return redirect(url_for('scan'))

            # -------------------------------------------------------------
            # 2. HANDLE NON-MOUTH / INVALID IMAGE REJECTION
            # -------------------------------------------------------------
            if result == "Invalid Image":
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash('⚠️ Invalid image uploaded. Please ensure you upload a clear photo focused inside the oral cavity.', 'warning')
                return redirect(url_for('scan'))

            # -------------------------------------------------------------
            # 3. GENERATE QR CODE FOR VALID SCANS
            # -------------------------------------------------------------
            qr_content = f"OralCare Report\nPatient: {patient_name}\nResult: {result}"
            qr_filename = f"qr_{filename}.png"
            qr_path = os.path.join(app.config['UPLOAD_FOLDER'], qr_filename)
            qr_img = qrcode.make(qr_content)
            qr_img.save(qr_path)

            # -------------------------------------------------------------
            # 4. SAVE VALID RECORD TO DATABASE
            # -------------------------------------------------------------
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO scans (user_id, patient_name, email, result, image, qr) VALUES (%s, %s, %s, %s, %s, %s)",
                    (session['user_id'], patient_name, email, result, filename, qr_filename)
                )
                conn.commit()
                scan_id = cursor.lastrowid
                cursor.close()
                conn.close()

                return redirect(url_for('result', scan_id=scan_id))
            except Exception as e:
                print(f"\n[!] MYSQL SCAN ERROR: {e}\n")
                return redirect(url_for('error'))

    return render_template('scan.html')

@app.route('/result/<int:scan_id>')
def result(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM scans WHERE id = %s AND user_id = %s", (scan_id, session['user_id']))
        scan_record = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"\n[!] MYSQL RESULT ERROR: {e}\n")
        scan_record = None

    if not scan_record:
        return redirect(url_for('error'))

    return render_template('result.html', scan=scan_record)

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    # host='0.0.0.0' allows connections from smartphones/other PCs on the local Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True)