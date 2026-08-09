import os
import cv2
import numpy as np
import qrcode
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        user="root",           # Added 'root'
        password="1234",       # Added your password so it doesn't crash!
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
# GRAD-CAM IMPLEMENTATION CLASS
# -------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations during the forward/backward pass
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
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
        heatmap /= (torch.max(heatmap) + 1e-8)
        
        return heatmap.cpu().numpy()

# -------------------------------------------------------------
# 3. OPENCV BLUR DETECTION & STRICT IMAGE VALIDATION
# -------------------------------------------------------------
def is_image_blurry(image_path, laplacian_thresh=12.0):
    img = cv2.imread(image_path)
    if img is None:
        return True, 0.0

    resized = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"\n[DEBUG] Standardized Sharpness Score: {round(laplacian_score, 2)}")
    is_blurry = laplacian_score < laplacian_thresh
    return is_blurry, round(laplacian_score, 2)


def validate_oral_cavity_image(image_path):
    """
    UPGRADED VALIDATION: Checks for red tissue, but REJECTS if it finds 
    'forbidden colors' like green or blue (e.g., fruit bowls, landscapes).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]

    # 1. TISSUE CHECK: Must contain red/pink/crimson hues
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([155, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    
    tissue_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    tissue_ratio = np.sum(tissue_mask > 0) / total_pixels

    # 2. FORBIDDEN COLOR CHECK: Mouths do NOT contain green, blue, cyan, or bright purple
    lower_forbidden = np.array([30, 40, 40])
    upper_forbidden = np.array([145, 255, 255])
    
    forbidden_mask = cv2.inRange(hsv, lower_forbidden, upper_forbidden)
    forbidden_ratio = np.sum(forbidden_mask > 0) / total_pixels

    # Rule A: If it doesn't have enough red/pink (less than 15%) -> Reject
    if tissue_ratio < 0.15:
        print(f"[REJECTED] Not enough tissue color. Ratio: {round(tissue_ratio, 3)}")
        return False

    # Rule B: If it has obvious non-oral colors (more than 5% green/blue) -> Reject
    if forbidden_ratio > 0.05:
        print(f"[REJECTED] Non-oral colors detected (Green/Blue/etc). Ratio: {round(forbidden_ratio, 3)}")
        return False

    return True


def predict_image(image_path):
    # 🛑 1. STRICT BLUR GATEKEEPER
    blurry, score = is_image_blurry(image_path, laplacian_thresh=12.0)
    if blurry:
        print(f"[REJECTED] Image flagged as Blurry with score: {score}")
        return "Blurry Image" 

    # 🛑 2. UPGRADED ORAL CAVITY TISSUE CHECK
    if not validate_oral_cavity_image(image_path):
        return "Invalid Image"

    # 🚀 3. RUN AI MODEL INFERENCE (Only reached if image is a clear mouth)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    # NOTE: torch.no_grad() is removed so Grad-CAM can trace back the gradients
    outputs = model(tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

    # 🎯 CONFIDENCE GATEKEEPER:
    if confidence.item() < 0.85:
        return "Invalid Image"

    is_cancer = (predicted.item() == 0)

    if is_cancer:
        # Generate Grad-CAM Heatmap for Cancer Cases
        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam.generate(tensor, target_class=0)
        
        orig_img = cv2.imread(image_path)
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Superimpose heatmap over original image
        overlay = cv2.addWeighted(orig_img, 0.6, colormap, 0.4, 0)
        
        # Save heatmap file alongside the original image with a "gradcam_" prefix
        dir_name = os.path.dirname(image_path)
        file_name = os.path.basename(image_path)
        gradcam_path = os.path.join(dir_name, f"gradcam_{file_name}")
        cv2.imwrite(gradcam_path, overlay)
        
        return "Cancer Detected"
    else:
        return "Clear"


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
    app.run(host='0.0.0.0', port=5000, debug=True)