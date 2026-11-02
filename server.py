import os
import sys
import time
import json
import threading
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename

# Ensure local module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import VideoTo3DProcessor

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = os.path.expanduser("~/Desktop/VideoTo3D_Outputs/uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global Job State
job_state = {
    "status": "idle",   # idle, running, completed, error
    "percent": 0,
    "stage": "Ready",
    "logs": [],
    "result": None,
    "error": None,
    "start_time": None
}
state_lock = threading.Lock()

def add_log(msg, pct=None, stage=None):
    with state_lock:
        if pct is not None:
            job_state["percent"] = pct
        if stage is not None:
            job_state["stage"] = stage
        job_state["logs"].append({
            "time": time.strftime("%H:%M:%S"),
            "msg": str(msg)
        })
        if len(job_state["logs"]) > 200:
            job_state["logs"].pop(0)

def worker_thread(video_path, mode, quality):
    with state_lock:
        job_state["status"] = "running"
        job_state["percent"] = 2
        job_state["stage"] = "Initializing"
        job_state["logs"] = []
        job_state["result"] = None
        job_state["error"] = None
        job_state["start_time"] = time.time()

    try:
        processor = VideoTo3DProcessor(log_callback=add_log)
        res = processor.process(video_path, mode=mode, quality=quality)
        with state_lock:
            job_state["status"] = "completed"
            job_state["percent"] = 100
            job_state["stage"] = "Done"
            job_state["result"] = res
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        with state_lock:
            job_state["status"] = "error"
            job_state["error"] = str(e)
            job_state["stage"] = "Failed"
            job_state["logs"].append({
                "time": time.strftime("%H:%M:%S"),
                "msg": f"ERROR: {err_msg}"
            })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    with state_lock:
        return jsonify(job_state)

@app.route("/api/browse_native", methods=["POST"])
def browse_native():
    """Triggers macOS native Finder file dialog without terminal"""
    script = '''
    set theFile to choose file with prompt "Select a Video for 3D Conversion:" of type {"public.movie", "mov", "mp4", "m4v", "avi", "mkv"}
    return POSIX path of theFile
    '''
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
        selected_path = result.stdout.strip()
        if selected_path and os.path.exists(selected_path):
            file_size_mb = os.path.getsize(selected_path) / (1024 * 1024)
            return jsonify({
                "success": True,
                "path": selected_path,
                "filename": os.path.basename(selected_path),
                "size_mb": round(file_size_mb, 2)
            })
        return jsonify({"success": False, "message": "No file chosen"})
    except subprocess.CalledProcessError:
        return jsonify({"success": False, "message": "User cancelled selection"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Supports direct browser drag-and-drop video upload"""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400

    filename = secure_filename(file.filename)
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(dest_path)
    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)

    return jsonify({
        "success": True,
        "path": dest_path,
        "filename": filename,
        "size_mb": round(file_size_mb, 2)
    })

@app.route("/api/start", methods=["POST"])
def start_conversion():
    data = request.json or {}
    video_path = data.get("video_path")
    mode = data.get("mode", "solid")
    quality = data.get("quality", "balanced")

    if not video_path or not os.path.exists(video_path):
        return jsonify({"success": False, "message": f"Invalid video file path: {video_path}"}), 400

    with state_lock:
        if job_state["status"] == "running":
            return jsonify({"success": False, "message": "A 3D conversion job is already running!"}), 409

    t = threading.Thread(target=worker_thread, args=(video_path, mode, quality), daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Conversion started"})

@app.route("/api/reveal_finder", methods=["POST"])
def reveal_finder():
    with state_lock:
        res = job_state.get("result")
    if res and "project_dir" in res and os.path.exists(res["project_dir"]):
        subprocess.run(["open", res["project_dir"]])
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "No output folder available"}), 400

@app.route("/api/open_blender", methods=["POST"])
def open_blender():
    with state_lock:
        res = job_state.get("result")
    if not res or "models" not in res:
        return jsonify({"success": False, "message": "No models available"}), 400

    models = res["models"]
    target_file = models.get("solid_glb") or models.get("solid_obj") or models.get("gaussian_splat_ply")
    if target_file and os.path.exists(target_file):
        try:
            subprocess.run(["open", "-a", "Blender", target_file])
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    return jsonify({"success": False, "message": "Model file not found"}), 404

@app.route("/api/download_model")
def download_model():
    file_path = request.args.get("path")
    if not file_path or not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

@app.route("/api/preview_glb")
def preview_glb():
    with state_lock:
        res = job_state.get("result")
    if res and "models" in res and "solid_glb" in res["models"]:
        glb_path = res["models"]["solid_glb"]
        if os.path.exists(glb_path):
            return send_file(glb_path, mimetype="model/gltf-binary")
    return "No GLB available", 404

def run_server(port=5050):
    print(f"🚀 Video-to-3D Local Server running at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    run_server(port)
