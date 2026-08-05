
import os
import cv2
import numpy as np
import qrcode
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector

# Modular Helper Imports
from blur_detector import is_image_blurry
from gradcam import generate_gradcam_overlay

app = Flask(__name__)
app.secret_key = 'oral_cancer_secret_key_prod'

# -------------------------------------------------------------
# DISABLE TEMPLATE & STATIC FILE CACHING
# Prevents browser/Flask from serving cached form fields
# -------------------------------------------------------------
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------------------------------------
# GLOBAL PUBLIC URL CONFIGURATION
# Update this with your active VS Code DevTunnel or ngrok URL
# -------------------------------------------------------------
PUBLIC_URL = "https://qt2p8d8b-5000.inc1.devtunnels.ms"

# -------------------------------------------------------------
# 1. DATABASE CONNECTION HELPER (WITH AUTOCOMMIT)
# -------------------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="1239",           # MySQL password
        database="oral_cancer_db",
        autocommit=True            # Guarantees user registration persists immediately
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
# 3. OPENCV ORAL CAVITY TISSUE VALIDATION
# -------------------------------------------------------------
def validate_oral_cavity_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([12, 240, 240])
    lower_red2 = np.array([160, 50, 40])
    upper_red2 = np.array([180, 240, 240])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    tissue_mask = mask1 | mask2

    total_pixels = img.shape[0] * img.shape[1]
    tissue_ratio = np.sum(tissue_mask > 0) / total_pixels

    if tissue_ratio < 0.20 or tissue_ratio > 0.88:
        return False

    mean_saturation = np.mean(hsv[:, :, 1])
    if mean_saturation > 190 or mean_saturation < 45:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / total_pixels
    
    if edge_density > 0.11:
        return False

    return True

# -------------------------------------------------------------
# 4. PREDICTION PIPELINE
# -------------------------------------------------------------
def predict_image(image_path):
    blurry, score = is_image_blurry(image_path, threshold=15.0)
    if blurry:
        return "Blurry Image"

    if not validate_oral_cavity_image(image_path):
        return "Invalid Image"

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

    if confidence.item() < 0.85:
        return "Invalid Image"

    return "Cancer Detected" if predicted.item() == 0 else "Clear"

# -------------------------------------------------------------
# 5. FLASK ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/db-debug')
def db_debug():
    """Diagnostic route to test direct database insertion without email."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 1. Fetch server metadata
        cursor.execute("SELECT @@port AS port, @@hostname AS host, @@datadir AS datadir")
        server_info = cursor.fetchone()
        
        # 2. Insert a test user
        test_user = f"debug_user_{os.urandom(2).hex()}"
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                       (test_user, 'test1234'))
        new_id = cursor.lastrowid
        
        # 3. Read total users count
        cursor.execute("SELECT COUNT(*) AS total FROM users")
        user_count = cursor.fetchone()['total']
        
        cursor.close()
        conn.close()
        
        return f"""
        <div style="font-family: monospace; padding: 20px;">
            <h2>✅ Database Write Test Succeeded!</h2>
            <p><b>Inserted User ID:</b> {new_id} ({test_user})</p>
            <p><b>Total Users in DB:</b> {user_count}</p>
            <hr>
            <h3>MySQL Connection Details Used by Flask:</h3>
            <ul>
                <li><b>Port:</b> {server_info['port']}</li>
                <li><b>Host:</b> {server_info['host']}</li>
                <li><b>Data Directory:</b> {server_info['datadir']}</li>
            </ul>
        </div>
        """
    except Exception as e:
        return f"<h2 style='color:red;'>❌ DB Connection Error:</h2><p>{e}</p>"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('register.html')

        hashed = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            # 1. Explicitly check if username exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash('Username is already taken. Please choose another.', 'warning')
                cursor.close()
                conn.close()
                return render_template('register.html')

            # 2. Insert new user record (Without email)
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed)
            )
            conn.commit()
            cursor.close()
            conn.close()

            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            print(f"\n[!] MYSQL REGISTER ERROR: {err}\n")
            flash(f'Database error: {err.msg}', 'danger')
        except Exception as e:
            print(f"\n[!] UNEXPECTED REGISTER ERROR: {e}\n")
            flash('An unexpected error occurred during registration.', 'danger')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

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
            print(f"\n[!] LOGIN ERROR: {e}\n")
            flash('Database connection failed. Check terminal logs.', 'danger')

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
        scans = []

    recent_scan = scans[0] if scans else None
    return render_template('dashboard.html', scans=scans, recent_scan=recent_scan)

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    patient_name = session.get('username', 'Patient')
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE id = %s", (session['user_id'],))
        row = cursor.fetchone()
        if row:
            patient_name = row.get('username') or patient_name
        cursor.close()
        conn.close()
    except Exception as e:
        pass

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('⚠️ Please select an image file.', 'warning')
            return redirect(url_for('scan'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_image(filepath)

        if result == "Blurry Image":
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('⚠️ Image is too blurry or shaky. Please keep steady and upload a focused photo.', 'danger')
            return redirect(url_for('scan'))

        if result == "Invalid Image":
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('⚠️ Invalid image uploaded. Please upload a clear photo focused inside the oral cavity.', 'danger')
            return redirect(url_for('scan'))

        # Grad-CAM Heatmap Generation
        gradcam_filename = None
        if result == "Cancer Detected":
            gradcam_filename = f"cam_{filename}"
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            cam_success = generate_gradcam_overlay(model, filepath, gradcam_path)
            if not cam_success:
                gradcam_filename = None

        # Save Scan Record directly (Without email column)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scans (user_id, patient_name, result, image, gradcam_image, qr) VALUES (%s, %s, %s, %s, %s, %s)",
                (session['user_id'], patient_name, result, filename, gradcam_filename, "")
            )
            conn.commit()
            scan_id = cursor.lastrowid

            # Build public report URL
            base_url = PUBLIC_URL.rstrip('/')
            report_url = f"{base_url}/report/{scan_id}"
            
            qr_filename = f"qr_{scan_id}.png"
            qr_path = os.path.join(app.config['UPLOAD_FOLDER'], qr_filename)
            
            qr_img = qrcode.make(report_url)
            qr_img.save(qr_path)

            cursor.execute("UPDATE scans SET qr = %s WHERE id = %s", (qr_filename, scan_id))
            conn.commit()

            cursor.close()
            conn.close()

            return redirect(url_for('result', scan_id=scan_id))
        except Exception as e:
            print(f"\n[!] MYSQL SCAN ERROR: {e}\n")
            return redirect(url_for('error', code=500, message='Failed to save scan record to database.'))

    return render_template('scan.html', patient_name=patient_name)

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
        scan_record = None

    if not scan_record:
        return redirect(url_for('error', code='404', message='Scan record not found.'))

    return render_template('result.html', scan=scan_record)

# -------------------------------------------------------------
# QR CODE VIEW ROUTE
# -------------------------------------------------------------
@app.route('/qr/<int:scan_id>')
def view_qr(scan_id):
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
        scan_record = None

    if not scan_record:
        return redirect(url_for('error', code='404', message='Scan record not found.'))

    return render_template('qr_page.html', scan=scan_record)

# -------------------------------------------------------------
# PUBLIC REPORT ROUTE
# -------------------------------------------------------------
@app.route('/report/<int:scan_id>')
def public_report(scan_id):
    scan_record = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM scans WHERE id = %s", (scan_id,))
        scan_record = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        return f"<div style='padding:20px; font-family:sans-serif;'><h2>Database Connection Error</h2><p>{e}</p></div>", 500

    if not scan_record:
        return f"<div style='padding:20px; font-family:sans-serif;'><h2>Report Not Found</h2><p>Diagnostic record #{scan_id} does not exist in the database.</p></div>", 404

    return render_template('result.html', scan=scan_record)

@app.route('/error')
def error():
    code = request.args.get('code', '500')
    message = request.args.get('message', 'An unexpected error occurred.')
    
    try:
        status_code = int(code)
    except ValueError:
        status_code = 500

    return render_template('error.html', error_code=status_code, error_message=message), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)