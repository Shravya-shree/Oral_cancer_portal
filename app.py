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

# Railway public domain
PUBLIC_PRODUCTION_URL = os.environ.get(
    'RAILWAY_PUBLIC_DOMAIN',
    ''
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
    forwarded_proto = request.headers.get(
        'X-Forwarded-Proto',
        'https'
    )

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
    model.classifier[1] = nn.Linear(
        model.last_channel,
        2
    )

    model_path = 'oral_cancer_model.pth'

    if os.path.exists(model_path):

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device('cpu')
            )
        )

        print(
            "[+] PyTorch MobileNetV2 weights loaded successfully."
        )

    else:

        print(
            f"[!] Warning: Model weights '{model_path}' not found."
        )

    model.eval()

    return model


model = load_pytorch_model()


# -------------------------------------------------------------
# 4. IMAGE TRANSFORMATION
# -------------------------------------------------------------
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224),
        antialias=True
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------------------
# 5. GRAD-CAM
# -------------------------------------------------------------
class GradCAM:

    def __init__(self, target_model, target_layer):

        self.model = target_model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hooks = []

        self.hooks.append(
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.hooks.append(
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        input,
        output
    ):
        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    def generate_from_output(
        self,
        output,
        target_class=0
    ):

        try:

            self.model.zero_grad(set_to_none=True)

            target_score = output[0, target_class]

            target_score.backward()

            if self.gradients is None or self.activations is None:
                return None

            pooled_gradients = torch.mean(
                self.gradients,
                dim=(2, 3),
                keepdim=True
            )

            activations = self.activations.detach()

            weighted_activations = (
                activations * pooled_gradients
            )

            heatmap = torch.sum(
                weighted_activations,
                dim=1
            )

            heatmap = F.relu(heatmap)

            heatmap = heatmap.squeeze(0)

            max_value = torch.max(heatmap)

            if max_value.item() > 0:
                heatmap = heatmap / max_value

            return heatmap.cpu().numpy()

        except Exception as e:

            print(
                f"[!] Grad-CAM generation error: {e}"
            )

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
# 6. OPENCV IMAGE GATEKEEPERS
# -------------------------------------------------------------
def load_and_normalize_image(image_path):

    try:

        pil_img = Image.open(image_path)

        pil_img = ImageOps.exif_transpose(
            pil_img
        ).convert('RGB')

        cv_img = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR
        )

        return pil_img, cv_img

    except Exception as e:

        print(
            f"[DEBUG] Image load error: {e}"
        )

        return None, None


def prepare_gatekeeper_image(cv_img):

    """
    Creates a smaller copy only for blur/oral-cavity checks.
    The original image is preserved for the report and Grad-CAM.
    """

    if cv_img is None:
        return None

    height, width = cv_img.shape[:2]

    max_dimension = 800

    if max(height, width) <= max_dimension:
        return cv_img

    scale = max_dimension / float(
        max(height, width)
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    return cv2.resize(
        cv_img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def is_image_blurry(
    cv_img,
    laplacian_thresh=4.0
):

    if cv_img is None:
        return True, 0.0

    small_img = cv_img

    height, width = cv_img.shape[:2]

    if max(height, width) > 500:

        scale = 500 / float(
            max(height, width)
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        small_img = cv2.resize(
            cv_img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        small_img,
        cv2.COLOR_BGR2GRAY
    )

    laplacian_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_blur = (
        laplacian_score < laplacian_thresh
    )

    return (
        is_blur,
        round(laplacian_score, 2)
    )


def validate_oral_cavity_image(cv_img):

    if cv_img is None:
        return False

    small_img = prepare_gatekeeper_image(
        cv_img
    )

    hsv = cv2.cvtColor(
        small_img,
        cv2.COLOR_BGR2HSV
    )

    total_pixels = (
        small_img.shape[0]
        * small_img.shape[1]
    )

    gray = cv2.cvtColor(
        small_img,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = (
        np.sum(edges > 0)
        / total_pixels
    )

    if edge_density > 0.22:
        return False

    mean_saturation = np.mean(
        hsv[:, :, 1]
    )

    if mean_saturation < 15.0:
        return False

    lower_red1 = np.array(
        [0, 30, 30]
    )

    upper_red1 = np.array(
        [15, 255, 255]
    )

    lower_red2 = np.array(
        [150, 30, 30]
    )

    upper_red2 = np.array(
        [180, 255, 255]
    )

    tissue_mask = (
        cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )
        |
        cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )
    )

    tissue_ratio = (
        np.sum(tissue_mask > 0)
        / total_pixels
    )

    if tissue_ratio < 0.08:
        return False

    lower_forbidden = np.array(
        [35, 40, 40]
    )

    upper_forbidden = np.array(
        [140, 255, 255]
    )

    forbidden_mask = cv2.inRange(
        hsv,
        lower_forbidden,
        upper_forbidden
    )

    forbidden_ratio = (
        np.sum(forbidden_mask > 0)
        / total_pixels
    )

    if forbidden_ratio > 0.15:
        return False

    return True


# -------------------------------------------------------------
# 7. FAST IMAGE PREDICTION + GRAD-CAM
# -------------------------------------------------------------
def predict_image(image_path):

    start_time = time.time()

    pil_img, cv_img = load_and_normalize_image(
        image_path
    )

    if pil_img is None or cv_img is None:
        return "Invalid Image"

    # ---------------------------------------------------------
    # BLUR CHECK
    # ---------------------------------------------------------
    blurry, score = is_image_blurry(
        cv_img,
        laplacian_thresh=4.0
    )

    if blurry:

        print(
            f"[SCAN] Blurry image rejected. Score: {score}"
        )

        return "Blurry Image"

    # ---------------------------------------------------------
    # ORAL CAVITY VALIDATION
    # ---------------------------------------------------------
    if not validate_oral_cavity_image(cv_img):

        print(
            "[SCAN] Image rejected by oral cavity validation."
        )

        return "Invalid Image"

    # ---------------------------------------------------------
    # PREPARE MODEL INPUT
    # ---------------------------------------------------------
    tensor = IMAGE_TRANSFORM(
        pil_img
    ).unsqueeze(0)

    tensor = tensor.to(
        next(model.parameters()).device
    )

    # ---------------------------------------------------------
    # SINGLE MODEL FORWARD PASS
    #
    # IMPORTANT:
    # The same output is reused for Grad-CAM.
    # This avoids running MobileNetV2 twice.
    # ---------------------------------------------------------
    gradcam = None

    if True:

        target_layer = model.features[-1]

        gradcam = GradCAM(
            model,
            target_layer
        )

        try:

            model.zero_grad(
                set_to_none=True
            )

            outputs = model(
                tensor
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probs,
                1
            )

            confidence_value = (
                confidence.item()
            )

            pred_class = (
                predicted.item()
            )

            conf_score = round(
                confidence_value * 100,
                2
            )

            print(
                f"[SCAN] Prediction: {pred_class} | "
                f"Confidence: {conf_score}%"
            )

            # -------------------------------------------------
            # CONFIDENCE CHECK
            # -------------------------------------------------
            if confidence_value < 0.50:

                gradcam.remove_hooks()

                return "Invalid Image"

            # -------------------------------------------------
            # CANCER DETECTED
            # -------------------------------------------------
            if pred_class == 0:

                heatmap = (
                    gradcam.generate_from_output(
                        outputs,
                        target_class=0
                    )
                )

                if heatmap is not None:

                    # Generate Grad-CAM at the original
                    # image dimensions for the report.
                    heatmap_resized = cv2.resize(
                        heatmap,
                        (
                            cv_img.shape[1],
                            cv_img.shape[0]
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )

                    heatmap_uint8 = np.uint8(
                        255 * heatmap_resized
                    )

                    colormap = cv2.applyColorMap(
                        heatmap_uint8,
                        cv2.COLORMAP_JET
                    )

                    overlay = cv2.addWeighted(
                        cv_img,
                        0.6,
                        colormap,
                        0.4,
                        0
                    )

                    dir_name = os.path.dirname(
                        image_path
                    )

                    file_name = os.path.basename(
                        image_path
                    )

                    gradcam_path = os.path.join(
                        dir_name,
                        f"gradcam_{file_name}"
                    )

                    cv2.imwrite(
                        gradcam_path,
                        overlay,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            90
                        ]
                    )

                    print(
                        "[SCAN] Grad-CAM generated successfully."
                    )

                else:

                    print(
                        "[SCAN] Grad-CAM could not be generated."
                    )

                total_time = (
                    time.time()
                    - start_time
                )

                print(
                    f"[SCAN] Total processing time: "
                    f"{total_time:.2f} seconds"
                )

                return "Cancer Detected"

            # -------------------------------------------------
            # CLEAR
            # -------------------------------------------------
            else:

                gradcam.remove_hooks()

                total_time = (
                    time.time()
                    - start_time
                )

                print(
                    f"[SCAN] Total processing time: "
                    f"{total_time:.2f} seconds"
                )

                return "Clear"

        except Exception as e:

            print(
                f"[!] Prediction error: {e}"
            )

            if gradcam is not None:
                gradcam.remove_hooks()

            return "Invalid Image"


# -------------------------------------------------------------
# 8. FLASK ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():

    return render_template(
        'index.html'
    )


@app.route(
    '/register',
    methods=['GET', 'POST']
)
def register():

    if request.method == 'POST':

        username = request.form.get(
            'username',
            ''
        ).strip()

        password = request.form.get(
            'password',
            ''
        ).strip()

        if not username or not password:

            flash(
                'Username and password are required.',
                'danger'
            )

            return render_template(
                'register.html'
            )

        hashed = generate_password_hash(
            password
        )

        conn = get_db_connection()

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO users
                (username, password)
                VALUES (?, ?)
                """,
                (
                    username,
                    hashed
                )
            )

            conn.commit()

            flash(
                'Registration successful. Please log in.',
                'success'
            )

            return redirect(
                url_for('login')
            )

        except sqlite3.IntegrityError:

            flash(
                'Username already exists. Please choose another.',
                'danger'
            )

            return render_template(
                'register.html'
            )

        except Exception as e:

            flash(
                f'Registration error: {e}',
                'danger'
            )

            return render_template(
                'register.html'
            )

        finally:

            conn.close()

    return render_template(
        'register.html'
    )


@app.route(
    '/login',
    methods=['GET', 'POST']
)
def login():

    if request.method == 'POST':

        username = request.form.get(
            'username',
            ''
        ).strip()

        password = request.form.get(
            'password',
            ''
        ).strip()

        conn = get_db_connection()

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM users
                WHERE username = ?
                """,
                (username,)
            )

            user = cursor.fetchone()

            if user and check_password_hash(
                user['password'],
                password
            ):

                session['user_id'] = user['id']
                session['username'] = user['username']
                session['user'] = user['username']

                flash(
                    'Welcome back!',
                    'success'
                )

                return redirect(
                    url_for('dashboard')
                )

            else:

                flash(
                    'Invalid credentials.',
                    'danger'
                )

        finally:

            conn.close()

    return render_template(
        'login.html'
    )


@app.route('/logout')
def logout():

    session.clear()

    flash(
        'Logged out successfully.',
        'info'
    )

    return redirect(
        url_for('index')
    )


@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:
        return redirect(
            url_for('login')
        )

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM scans
        WHERE user_id = ?
        ORDER BY scan_date DESC
        """,
        (session['user_id'],)
    )

    rows = cursor.fetchall()

    scans = [
        dict(r)
        for r in rows
    ]

    for s in scans:

        s['date'] = str(
            s.get(
                'scan_date',
                'Recorded'
            )
        )[:16]

    conn.close()

    recent_scan = (
        scans[0]
        if scans
        else None
    )

    return render_template(
        'dashboard.html',
        scans=scans,
        recent_scan=recent_scan,
        user=session.get('username')
    )


@app.route(
    '/scan',
    methods=['GET', 'POST']
)
def scan():

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )

    if request.method == 'POST':

        patient_name = (
            request.form.get(
                'patient_name'
            )
            or request.form.get(
                'name'
            )
            or session.get(
                'username',
                'Patient'
            )
        )

        email = request.form.get(
            'email',
            ''
        )

        file = (
            request.files.get('file')
            or request.files.get('image')
        )

        if not file or file.filename == '':

            flash(
                '⚠️ Please select an image file to analyze.',
                'warning'
            )

            return redirect(
                url_for('scan')
            )

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        result = predict_image(
            filepath
        )

        if result == "Blurry Image":

            if os.path.exists(filepath):

                os.remove(filepath)

            flash(
                '⚠️ Image is too blurry or shaky. Please keep steady and upload a focused photo.',
                'danger'
            )

            return redirect(
                url_for('scan')
            )

        if result == "Invalid Image":

            if os.path.exists(filepath):

                os.remove(filepath)

            flash(
                '⚠️ Invalid image uploaded. Please ensure photo is centered inside the oral cavity.',
                'warning'
            )

            return redirect(
                url_for('scan')
            )

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO scans
            (
                user_id,
                patient_name,
                email,
                result,
                image,
                qr
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session['user_id'],
                patient_name,
                email,
                result,
                filename,
                ""
            )
        )

        conn.commit()

        scan_id = cursor.lastrowid

        conn.close()

        return redirect(
            url_for(
                'result',
                scan_id=scan_id
            )
        )

    return render_template(
        'scan.html',
        patient_name=session.get(
            'username',
            'Patient'
        )
    )


@app.route(
    '/result/<int:scan_id>'
)
def result(scan_id):

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM scans
        WHERE id = ?
        AND user_id = ?
        """,
        (
            scan_id,
            session['user_id']
        )
    )

    row = cursor.fetchone()

    conn.close()

    if not row:

        return redirect(
            url_for('error')
        )

    scan_record = dict(row)

    scan_record['date'] = str(
        scan_record.get(
            'scan_date',
            'Recorded'
        )
    )[:16]

    return render_template(
        'result.html',
        scan=scan_record,
        record=scan_record
    )


@app.route(
    '/qr/<int:scan_id>'
)
def view_qr(scan_id):

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM scans
        WHERE id = ?
        AND user_id = ?
        """,
        (
            scan_id,
            session['user_id']
        )
    )

    row = cursor.fetchone()

    if not row:

        conn.close()

        return redirect(
            url_for('error')
        )

    scan_record = dict(row)

    scan_record['date'] = str(
        scan_record.get(
            'scan_date',
            'Recorded'
        )
    )[:16]

    qr_filename = (
        f"qr_{scan_id}.png"
    )

    qr_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        qr_filename
    )

    base_url = get_accessible_base_url()

    report_url = (
        f"{base_url}/report/{scan_id}"
    )

    qr_img = qrcode.make(
        report_url
    )

    qr_img.save(
        qr_path
    )

    cursor.execute(
        """
        UPDATE scans
        SET qr = ?
        WHERE id = ?
        """,
        (
            qr_filename,
            scan_id
        )
    )

    conn.commit()

    conn.close()

    scan_record['qr'] = qr_filename

    cache_id = int(
        time.time()
    )

    return render_template(
        'qr_page.html',
        scan=scan_record,
        record=scan_record,
        qr_filename=qr_filename,
        cache_id=cache_id
    )


@app.route(
    '/report/<int:scan_id>'
)
def public_report(scan_id):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM scans
        WHERE id = ?
        """,
        (scan_id,)
    )

    row = cursor.fetchone()

    conn.close()

    if not row:

        return (
            "<h3 style='padding:20px; "
            "font-family:sans-serif;'>"
            "Diagnostic Report Not Found."
            "</h3>",
            404
        )

    scan_record = dict(row)

    scan_record['date'] = str(
        scan_record.get(
            'scan_date',
            'Recorded'
        )
    )[:16]

    return render_template(
        'result.html',
        scan=scan_record,
        record=scan_record
    )


@app.route('/error')
def error():

    return render_template(
        'error.html'
    )


# -------------------------------------------------------------
# 9. APPLICATION START
# -------------------------------------------------------------
if __name__ == '__main__':

    port = int(
        os.environ.get(
            'PORT',
            5000
        )
    )

    app.run(
        host='0.0.0.0',
        port=port
    )