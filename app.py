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

# -------------------------------------------------------------
# PYTORCH CPU OPTIMIZATION
# -------------------------------------------------------------
try:
    cpu_count = os.cpu_count() or 2
    torch.set_num_threads(min(2, cpu_count))
    torch.set_num_interop_threads(1)
except Exception:
    pass

app = Flask(__name__)

app.secret_key = os.environ.get(
    'SECRET_KEY',
    'oral_cancer_secret_key_dev'
)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

DB_PATH = 'oral_cancer.db'

PUBLIC_PRODUCTION_URL = os.environ.get(
    'RENDER_EXTERNAL_URL',
    os.environ.get('RAILWAY_PUBLIC_DOMAIN', '')
)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# -------------------------------------------------------------
# 1. DATABASE CONNECTION & AUTO-MIGRATION
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
    if PUBLIC_PRODUCTION_URL and PUBLIC_PRODUCTION_URL.strip():
        domain = PUBLIC_PRODUCTION_URL.strip().rstrip('/')
        if domain.startswith('http://') or domain.startswith('https://'):
            return domain
        return f"https://{domain}"

    forwarded_host = request.headers.get('X-Forwarded-Host')
    forwarded_proto = request.headers.get('X-Forwarded-Proto', 'https')

    if forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}"

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
# 3. MODEL DEFINITION & LOADING
# -------------------------------------------------------------
def load_pytorch_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[0] = nn.Dropout(p=0.35)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model_path = 'oral_cancer_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
        print("[+] PyTorch MobileNetV2 weights loaded successfully.")
    else:
        print(f"[!] Warning: Model weights '{model_path}' not found.")

    model.eval()
    return model


model = load_pytorch_model()


# -------------------------------------------------------------
# 4. IMAGE TRANSFORMATION
# -------------------------------------------------------------
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------------------
# 5. GRAD-CAM EXPLAINABILITY
# -------------------------------------------------------------
class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self.hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_from_output(self, output, target_class=0):
        try:
            self.model.zero_grad(set_to_none=True)
            target_score = output[0, target_class]
            target_score.backward()

            if self.gradients is None or self.activations is None:
                return None

            pooled_gradients = torch.mean(
                self.gradients, dim=(2, 3), keepdim=True
            )
            activations = self.activations.detach()
            weighted_activations = activations * pooled_gradients

            heatmap = torch.sum(weighted_activations, dim=1)
            heatmap = F.relu(heatmap)
            heatmap = heatmap.squeeze(0)

            max_value = torch.max(heatmap)
            if max_value.item() > 0:
                heatmap = heatmap / max_value

            return heatmap.cpu().numpy()
        except Exception as e:
            print(f"[!] Grad-CAM generation error: {e}")
            return None
        finally:
            self.remove_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()


# -------------------------------------------------------------
# 6. IMAGE QUALITY ASSESSMENT & GATEKEEPERS
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


def prepare_gatekeeper_image(cv_img, max_dimension=800):
    if cv_img is None:
        return None

    height, width = cv_img.shape[:2]
    if max(height, width) <= max_dimension:
        return cv_img

    scale = max_dimension / float(max(height, width))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def assess_image_quality(cv_img, blur_thresh=30.0):
    """
    Evaluates physical and optical suitability of the upload.
    Checks:
      1. Minimum resolution (200x200 px)
      2. Underexposure (mean brightness < 35.0)
      3. Severe overexposure (> 35% washed-out pixels)
      4. Optical and motion blur via normalized Laplacian variance (threshold: 30.0)
    Returns:
      (is_acceptable: bool, status_reason: str, metrics: dict)
    """
    if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.size == 0:
        return False, "Unreadable or corrupted image file", {}

    height, width = cv_img.shape[:2]

    # 1. Resolution Check
    if height < 200 or width < 200:
        reason = f"Image resolution too low ({width}x{height} px; minimum is 200x200 px)"
        print(f"[QUALITY] Decision: REJECTED | Reason: {reason}")
        return False, reason, {"width": width, "height": height}

    # Standardize image to 800px maximum dimension for consistent photometric analysis
    std_img = prepare_gatekeeper_image(cv_img, max_dimension=800)
    gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
    total_pixels = std_img.shape[0] * std_img.shape[1]

    # 2. Illumination / Underexposure Check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 35.0:
        reason = f"Image is severely underexposed/dark (mean brightness: {mean_brightness:.1f}/255)"
        print(f"[QUALITY] Decision: REJECTED | Reason: {reason}")
        return False, reason, {"brightness": mean_brightness}

    # 3. Severe Overexposure / Flash Washout Check
    overexposed_pixels = int(np.sum(gray > 248))
    overexposure_ratio = overexposed_pixels / float(total_pixels)
    if overexposure_ratio > 0.35:
        reason = f"Image is severely overexposed/washed out ({overexposure_ratio * 100:.1f}% glare pixels)"
        print(f"[QUALITY] Decision: REJECTED | Reason: {reason}")
        return False, reason, {"overexposure_ratio": overexposure_ratio}

    # 4. Normalized Blur Check (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {
        "width": width,
        "height": height,
        "brightness": round(mean_brightness, 2),
        "overexposure_ratio": round(overexposure_ratio, 3),
        "blur_score": round(laplacian_var, 2),
        "blur_threshold": blur_thresh
    }

    print(
        f"[QUALITY] Dimensions: {width}x{height} | Brightness: {mean_brightness:.1f} | "
        f"Blur Score: {laplacian_var:.2f} (Threshold: {blur_thresh:.1f})"
    )

    if laplacian_var < blur_thresh:
        reason = f"Image is out of focus or motion-blurred (sharpness score: {laplacian_var:.1f} < {blur_thresh:.1f})"
        print(f"[QUALITY] Decision: REJECTED | Reason: {reason}")
        return False, reason, metrics

    print("[QUALITY] Decision: ACCEPTED | Technical quality verified.")
    return True, "Quality Acceptable", metrics


def validate_oral_cavity_image(cv_img):
    """
    Validates that the upload represents genuine intraoral cavity anatomy.
    Rejects external skin (palms, wrists, arms), watches, and non-biological items.
    """
    if (
        cv_img is None
        or not isinstance(cv_img, np.ndarray)
        or cv_img.ndim != 3
        or cv_img.shape[2] != 3
    ):
        print("[GATEKEEPER] [REJECTED] Input is not a valid 3-channel image array.")
        return False

    small_img = prepare_gatekeeper_image(cv_img, max_dimension=800)
    if small_img is None or small_img.size == 0:
        print("[GATEKEEPER] [REJECTED] Image preprocessing buffer is empty.")
        return False

    total_pixels = small_img.shape[0] * small_img.shape[1]
    if total_pixels <= 0:
        return False

    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blue_ch, green_ch, red_ch = cv2.split(small_img)
    b16 = blue_ch.astype(np.int16)
    g16 = green_ch.astype(np.int16)
    r16 = red_ch.astype(np.int16)

    # 1. Texture & Flatness
    gray_std = float(np.std(gray))
    contrast_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    if gray_std < 10.0 or contrast_range < 24.0:
        print(f"[GATEKEEPER] [REJECTED] Flat or uniform image (std: {gray_std:.2f}, contrast: {contrast_range:.2f}).")
        return False

    # 2. Watch / Metallic Surface Detection
    metallic_mask = (
        (saturation < 25)
        & (value > 35)
        & (np.abs(r16 - g16) < 14)
        & (np.abs(g16 - b16) < 14)
    )
    metallic_ratio = float(np.count_nonzero(metallic_mask) / total_pixels)
    if metallic_ratio > 0.42:
        print(f"[GATEKEEPER] [REJECTED] Non-biological metallic/watch surface detected (ratio: {metallic_ratio:.2f}).")
        return False

    # 3. Forbidden Background Hues (Synthetic green/cyan)
    blue_green_mask = cv2.inRange(
        hsv, np.array([38, 45, 35]), np.array([135, 255, 255])
    ) > 0
    blue_green_ratio = float(np.count_nonzero(blue_green_mask) / total_pixels)
    if blue_green_ratio > 0.22:
        print(f"[GATEKEEPER] [REJECTED] Non-oral background hue detected (blue/green ratio: {blue_green_ratio:.2f}).")
        return False

    # 4. Anatomical Tissue Feature Extraction
    crimson_mucosa_mask = (
        ((hue <= 12) | (hue >= 160))
        & (saturation >= 30)
        & (value >= 30)
        & (r16 >= g16 + 6)
        & (r16 >= b16 + 10)
        & (r16 >= 55)
    )
    coral_tongue_mask = (
        (hue >= 13) & (hue <= 24)
        & (saturation >= 35) & (saturation <= 200)
        & (value >= 40)
        & (r16 >= g16 + 8)
        & (r16 >= b16 + 12)
        & (r16 >= 70)
    )
    total_mucosa_mask = crimson_mucosa_mask | coral_tongue_mask
    mucosa_ratio = float(np.count_nonzero(total_mucosa_mask) / total_pixels)

    dental_keratosis_mask = (
        (saturation < 48)
        & (value >= 110)
        & (r16 >= 90) & (g16 >= 90) & (b16 >= 70)
        & (np.abs(r16 - g16) <= 30)
    )
    dental_ratio = float(np.count_nonzero(dental_keratosis_mask) / total_pixels)

    intraoral_depth_mask = (
        (value < 45)
        | ((r16 < 55) & (g16 < 45) & (b16 < 45))
    )
    depth_shadow_ratio = float(np.count_nonzero(intraoral_depth_mask) / total_pixels)

    dermal_skin_mask = (
        (hue >= 11) & (hue <= 25)
        & (saturation >= 18) & (saturation <= 130)
        & (value >= 55)
        & (r16 >= g16) & (r16 < g16 + 18)
        & (g16 >= b16 - 8)
    )
    dermal_skin_ratio = float(np.count_nonzero(dermal_skin_mask) / total_pixels)

    # 5. Domain Rejection / Acceptance
    is_pure_external_skin = (
        dermal_skin_ratio > 0.50
        and mucosa_ratio < 0.05
        and dental_ratio < 0.02
        and depth_shadow_ratio < 0.03
    )
    if is_pure_external_skin:
        print(
            f"[GATEKEEPER] [REJECTED] External skin detected without oral features "
            f"(Dermal: {dermal_skin_ratio:.2f}, Mucosa: {mucosa_ratio:.2f}, "
            f"Dental: {dental_ratio:.2f}, Depth: {depth_shadow_ratio:.2f})."
        )
        return False

    has_oral_landmarks = (
        (mucosa_ratio >= 0.06)
        or (mucosa_ratio >= 0.03 and dental_ratio >= 0.02)
        or (mucosa_ratio >= 0.03 and depth_shadow_ratio >= 0.04)
    )

    if not has_oral_landmarks:
        print(
            f"[GATEKEEPER] [REJECTED] Insufficient oral cavity anatomy "
            f"(Mucosa: {mucosa_ratio:.2f} [min 0.06], Dental: {dental_ratio:.2f}, "
            f"Depth: {depth_shadow_ratio:.2f})."
        )
        return False

    print(
        f"[GATEKEEPER] [PASSED] Intraoral anatomy verified "
        f"(Mucosa: {mucosa_ratio:.2f}, Dental: {dental_ratio:.2f}, "
        f"Depth: {depth_shadow_ratio:.2f}, Dermal: {dermal_skin_ratio:.2f})."
    )
    return True


# -------------------------------------------------------------
# 7. CLINICAL PREDICTION & TRIAGE ENGINE
# -------------------------------------------------------------
def predict_image(image_path):
    start_time = time.time()

    pil_img, cv_img = load_and_normalize_image(image_path)
    if pil_img is None or cv_img is None:
        print("[SCAN] Image decoding failed. Aborting pipeline.")
        return "Invalid Image: Unreadable or corrupted file"

    # ---------------------------------------------------------
    # STAGE 1: DEDICATED IMAGE QUALITY GATE
    # ---------------------------------------------------------
    quality_passed, quality_reason, metrics = assess_image_quality(cv_img, blur_thresh=30.0)
    if not quality_passed:
        print(f"[SCAN] Quality Gate FAILED: {quality_reason}")
        print("[SCAN] Cancer classifier skipped because image quality failed.")
        return f"Image Quality Insufficient: {quality_reason}"

    # ---------------------------------------------------------
    # STAGE 2: ORAL CAVITY GATEKEEPER
    # ---------------------------------------------------------
    if not validate_oral_cavity_image(cv_img):
        print("[SCAN] Image rejected by oral cavity anatomical validation.")
        print("[SCAN] Cancer classifier skipped because image is not intraoral.")
        return "Invalid Image: Photo must be centered inside the oral cavity"

    # ---------------------------------------------------------
    # STAGE 3: CANCER CLASSIFICATION & GRAD-CAM
    # ---------------------------------------------------------
    tensor = IMAGE_TRANSFORM(pil_img).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)

    gradcam = GradCAM(model, model.features[-1])

    try:
        model.eval()

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)

            # Class 0: Cancer Detected (Malignant) | Class 1: Clear (Benign)
            cancer_prob = float(probs[0, 0].item())
            clear_prob = float(probs[0, 1].item())

        cancer_score = round(cancer_prob * 100, 2)
        clear_score = round(clear_prob * 100, 2)

        print(
            f"[DEBUG] Softmax Probs -> Class 0 (Cancer): {cancer_score}%, "
            f"Class 1 (Clear): {clear_score}%"
        )

        # Zone 3: Cancer Detected (P(Cancer) >= 0.50)
        if cancer_prob >= 0.50:
            print(f"[SCAN] Decision: Cancer Detected | Confidence: {cancer_score}%")
            generate_gradcam_overlay(gradcam, model, tensor, cv_img, image_path)
            total_time = time.time() - start_time
            print(f"[SCAN] Total processing time: {total_time:.2f} seconds")
            return "Cancer Detected"

        # Zone 2: Inconclusive / Clinical Review (0.40 <= P(Cancer) < 0.50 or Clear < 60%)
        elif cancer_prob >= 0.40 or clear_prob < 0.60:
            print(
                f"[SCAN] Decision: Inconclusive (Borderline Suspicion) | "
                f"Cancer: {cancer_score}%, Clear: {clear_score}%"
            )
            generate_gradcam_overlay(gradcam, model, tensor, cv_img, image_path)
            total_time = time.time() - start_time
            print(f"[SCAN] Total processing time: {total_time:.2f} seconds")
            return "Inconclusive — Further Evaluation Required"

        # Zone 1: Decisive Clear (P(Cancer) < 0.40 and P(Clear) >= 0.60)
        else:
            gradcam.remove_hooks()
            print(f"[SCAN] Decision: Clear | Confidence: {clear_score}%")
            total_time = time.time() - start_time
            print(f"[SCAN] Total processing time: {total_time:.2f} seconds")
            return "Clear"

    except Exception as e:
        print(f"[!] Prediction error: {e}")
        if gradcam is not None:
            gradcam.remove_hooks()
        return "Processing Error"


def generate_gradcam_overlay(gradcam, model_ref, tensor, cv_img, image_path):
    try:
        with torch.enable_grad():
            gradcam_outputs = model_ref(tensor)
            heatmap = gradcam.generate_from_output(gradcam_outputs, target_class=0)

        if heatmap is not None:
            heatmap_resized = cv2.resize(
                heatmap,
                (cv_img.shape[1], cv_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv_img, 0.6, colormap, 0.4, 0)

            dir_name = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            gradcam_path = os.path.join(dir_name, f"gradcam_{file_name}")

            cv2.imwrite(
                gradcam_path,
                overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            print("[SCAN] Grad-CAM overlay generated successfully.")
        else:
            print("[SCAN] Warning: Grad-CAM heatmap was empty.")
    except Exception as e:
        print(f"[!] Error generating Grad-CAM overlay: {e}")


# -------------------------------------------------------------
# 8. FLASK ROUTES
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
            cursor.execute(
                """
                INSERT INTO users (username, password)
                VALUES (?, ?)
                """,
                (username, hashed)
            )
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
            cursor.execute(
                """
                SELECT * FROM users
                WHERE username = ?
                """,
                (username,)
            )
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
    cursor.execute(
        """
        SELECT * FROM scans
        WHERE user_id = ?
        ORDER BY scan_date DESC
        """,
        (session['user_id'],)
    )
    rows = cursor.fetchall()
    scans = [dict(r) for r in rows]

    for s in scans:
        s['date'] = str(s.get('scan_date', 'Recorded'))[:16]

    conn.close()
    recent_scan = scans[0] if scans else None

    return render_template(
        'dashboard.html',
        scans=scans,
        recent_scan=recent_scan,
        user=session.get('username')
    )


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = (
            request.form.get('patient_name')
            or request.form.get('name')
            or session.get('username', 'Patient')
        )
        email = request.form.get('email', '')
        file = request.files.get('file') or request.files.get('image')

        if not file or file.filename == '':
            flash('⚠️ Please select an image file to analyze.', 'warning')
            return redirect(url_for('scan'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_image(filepath)

        # Fatal decoding error: flash and prompt for clean upload
        if result.startswith("Invalid Image: Unreadable"):
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('⚠️ The uploaded file is corrupted or unreadable. Please upload a valid JPEG/PNG image.', 'danger')
            return redirect(url_for('scan'))

        # Record scan with verified result (Screening status, Quality Insufficient, or Invalid Anatomy)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scans (user_id, patient_name, email, result, image, qr)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session['user_id'], patient_name, email, result, filename, "")
        )
        conn.commit()
        scan_id = cursor.lastrowid
        conn.close()

        return redirect(url_for('result', scan_id=scan_id))

    return render_template(
        'scan.html',
        patient_name=session.get('username', 'Patient')
    )


@app.route('/result/<int:scan_id>')
def result(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM scans
        WHERE id = ? AND user_id = ?
        """,
        (scan_id, session['user_id'])
    )
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
    cursor.execute(
        """
        SELECT * FROM scans
        WHERE id = ? AND user_id = ?
        """,
        (scan_id, session['user_id'])
    )
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

    cursor.execute(
        """
        UPDATE scans SET qr = ?
        WHERE id = ?
        """,
        (qr_filename, scan_id)
    )
    conn.commit()
    conn.close()

    scan_record['qr'] = qr_filename
    cache_id = int(time.time())

    return render_template(
        'qr_page.html',
        scan=scan_record,
        record=scan_record,
        qr_filename=qr_filename,
        cache_id=cache_id
    )


@app.route('/report/<int:scan_id>')
def public_report(scan_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM scans
        WHERE id = ?
        """,
        (scan_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return (
            "<h3 style='padding:20px; font-family:sans-serif;'>"
            "Diagnostic Report Not Found."
            "</h3>",
            404
        )

    scan_record = dict(row)
    scan_record['date'] = str(scan_record.get('scan_date', 'Recorded'))[:16]

    return render_template('result.html', scan=scan_record, record=scan_record)


@app.route('/error')
def error():
    return render_template('error.html')


# -------------------------------------------------------------
# 9. APPLICATION START
# -------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)