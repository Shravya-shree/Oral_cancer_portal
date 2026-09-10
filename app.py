import os
import cv2
import time
import socket
import sqlite3
import numpy as np
import qrcode
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageFile
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
app.secret_key = 'oral_cancer_secret_key_prod'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

DB_PATH = 'oral_cancer.db'

# Automatically use Render's live URL or fallback to your live domain
PUBLIC_PRODUCTION_URL = os.environ.get('RENDER_EXTERNAL_URL', 'https://oral-cancer-portal.onrender.com')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------------------------------------
# 1. DATABASE CONNECTION & AUTO-MIGRATION (SQLITE)
# -------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def auto_migrate_database():
    """Initializes local SQLite schema automatically."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_name TEXT,
            email TEXT DEFAULT NULL,
            result TEXT,
            image TEXT,
            qr TEXT DEFAULT '',
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("[+] SQLite schema initialized successfully.")

auto_migrate_database()

# -------------------------------------------------------------
# 2. REACHABLE URL RESOLVER FOR QR CODES
# -------------------------------------------------------------
def get_accessible_base_url():
    # 1. Check Render's injected live URL or configured domain
    if PUBLIC_PRODUCTION_URL and PUBLIC_PRODUCTION_URL.strip():
        return PUBLIC_PRODUCTION_URL.rstrip('/')

    # 2. Fall back to forwarded reverse proxy headers (HTTPS support)
    forwarded_host = request.headers.get('X-Forwarded-Host')
    forwarded_proto = request.headers.get('X-Forwarded-Proto', 'https')
    if forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}"

    # 3. Fall back to local LAN IP if testing offline
    raw_host = request.host_url.rstrip('/')
    if "127.0.0.1" in raw_host or "localhost" in raw_host:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
            port = request.environ.get('SERVER_PORT', '5000')
            return f"http://{lan_ip}:{port}"
        except Exception:
            return raw_host

    return raw_host

# -------------------------------------------------------------
# 3. MODEL DEFINITION & LOADING (MobileNetV2)
# -------------------------------------------------------------
def load_pytorch_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[0] = nn.Dropout(p=0.35)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model_path = 'oral_cancer_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("[+] PyTorch MobileNetV2 weights loaded successfully.")
    else:
        print(f"[!] Warning: Model weights '{model_path}' not found.")
    model.eval()
    return model

model = load_pytorch_model()

# -------------------------------------------------------------
# 4. GRAD-CAM
# -------------------------------------------------------------
class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=0):
        try:
            input_tensor.requires_grad_(True)
            self.model.zero_grad()
            
            output = self.model(input_tensor)
            loss = output[0, target_class]
            loss.backward()

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations.detach().clone()
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            max_val = torch.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            
            return heatmap.cpu().numpy()
        finally:
            self.remove_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# -------------------------------------------------------------
# 5. OPENCV IMAGE GATEKEEPERS
# -------------------------------------------------------------
def load_and_normalize_image(image_path):
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except Exception as e:
        print(f"[DEBUG] Image load error: {e}")
        return None, None

def is_image_blurry(cv_img, laplacian_thresh=4.0):
    if cv_img is None:
        return True, 0.0
    resized = cv2.resize(cv_img, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blur = laplacian_score < laplacian_thresh
    return is_blur, round(laplacian_score, 2)

def validate_oral_cavity_image(cv_img):
    if cv_img is None:
        return False

    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    total_pixels = cv_img.shape[0] * cv_img.shape[1]

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / total_pixels
    if edge_density > 0.22:
        return False

    mean_saturation = np.mean(hsv[:, :, 1])
    if mean_saturation < 15.0:
        return False

    lower_red1, upper_red1 = np.array([0, 30, 30]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([150, 30, 30]), np.array([180, 255, 255])
    tissue_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    tissue_ratio = np.sum(tissue_mask > 0) / total_pixels
    if tissue_ratio < 0.08:
        return False

    lower_forbidden, upper_forbidden = np.array([35, 40, 40]), np.array([140, 255, 255])
    forbidden_mask = cv2.inRange(hsv, lower_forbidden, upper_forbidden)
    forbidden_ratio = np.sum(forbidden_mask > 0) / total_pixels
    if forbidden_ratio > 0.15:
        return False

    return True

def predict_image(image_path):
    pil_img, cv_img = load_and_normalize_image(image_path)
    if pil_img is None or cv_img is None:
        return "Invalid Image"

    blurry, score = is_image_blurry(cv_img, laplacian_thresh=4.0)
    if blurry:
        return "Blurry Image"

    if not validate_oral_cavity_image(cv_img):
        return "Invalid Image"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = next(model.parameters()).device
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    conf_score = round(confidence.item() * 100, 2)
    pred_class = predicted.item()

    if confidence.item() < 0.50:
        return "Invalid Image"

    if pred_class == 0:
        try:
            target_layer = model.features[-1]
            grad_cam = GradCAM(model, target_layer)
            heatmap = grad_cam.generate(tensor.clone(), target_class=0)

            heatmap_resized = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv_img, 0.6, colormap, 0.4, 0)

            dir_name = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            gradcam_path = os.path.join(dir_name, f"gradcam_{file_name}")
            cv2.imwrite(gradcam_path, overlay)
        except Exception as e:
            print(f"[!] Grad-CAM error: {e}")

        return "Cancer Detected"
    else:
        return "Clear"

# -------------------------------------------------------------
# 6. FLASK ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('register.html')

        hashed = generate_password_hash(password)
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'danger')
            return render_template('register.html')
        except Exception as e:
            flash(f'Registration error: {e}', 'danger')
            return render_template('register.html')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['user'] = user['username']
                flash('Welcome back!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.', 'danger')
        finally:
            conn.close()

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY scan_date DESC", (session['user_id'],))
    rows = cursor.fetchall()
    scans = [dict(r) for r in rows]
    for s in scans:
        s['date'] = str(s.get('scan_date', 'Recorded'))[:16]
    conn.close()

    recent_scan = scans[0] if scans else None
    return render_template('dashboard.html', scans=scans, recent_scan=recent_scan, user=session.get('username'))

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = request.form.get('patient_name') or request.form.get('name') or session.get('username', 'Patient')
        email = request.form.get('email', '')
        file = request.files.get('file') or request.files.get('image')

        if not file or file.filename == '':
            flash('⚠️ Please select an image file to analyze.', 'warning')
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
            flash('⚠️ Invalid image uploaded. Please ensure photo is centered inside the oral cavity.', 'warning')
            return redirect(url_for('scan'))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scans (user_id, patient_name, email, result, image, qr) VALUES (?, ?, ?, ?, ?, ?)",
            (session['user_id'], patient_name, email, result, filename, "")
        )
        conn.commit()
        scan_id = cursor.lastrowid
        conn.close()

        return redirect(url_for('result', scan_id=scan_id))

    return render_template('scan.html', patient_name=session.get('username', 'Patient'))

@app.route('/result/<int:scan_id>')
def result(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE id = ? AND user_id = ?", (scan_id, session['user_id']))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return redirect(url_for('error'))

    scan_record = dict(row)
    scan_record['date'] = str(scan_record.get('scan_date', 'Recorded'))[:16]

    return render_template('result.html', scan=scan_record, record=scan_record)

@app.route('/qr/<int:scan_id>')
def view_qr(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE id = ? AND user_id = ?", (scan_id, session['user_id']))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return redirect(url_for('error'))

    scan_record = dict(row)
    scan_record['date'] = str(scan_record.get('scan_date', 'Recorded'))[:16]

    qr_filename = f"qr_{scan_id}.png"
    qr_path = os.path.join(app.config['UPLOAD_FOLDER'], qr_filename)

    base_url = get_accessible_base_url()
    report_url = f"{base_url}/report/{scan_id}"
    qr_img = qrcode.make(report_url)
    qr_img.save(qr_path)

    cursor.execute("UPDATE scans SET qr = ? WHERE id = ?", (qr_filename, scan_id))
    conn.commit()
    conn.close()

    scan_record['qr'] = qr_filename
    cache_id = int(time.time())

    return render_template('qr_page.html', scan=scan_record, record=scan_record, qr_filename=qr_filename, cache_id=cache_id)

@app.route('/report/<int:scan_id>')
def public_report(scan_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "<h3 style='padding:20px; font-family:sans-serif;'>Diagnostic Report Not Found.</h3>", 404

    scan_record = dict(row)
    scan_record['date'] = str(scan_record.get('scan_date', 'Recorded'))[:16]

    return render_template('result.html', scan=scan_record, record=scan_record)

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)