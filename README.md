# Video to 3D World Studio (Apple Silicon macOS) 🚀

Convert real-world videos (`.MOV`, `.MP4`, `.M4V`, `.AVI`, `.MKV`) into **Solid 3D Objects** (`.GLB`, `.OBJ`, `.FBX`) and **3D Gaussian Splats** (`.PLY`) directly on Apple Silicon Macs (M1/M2/M3/M4) without using the terminal.

---

## 🌟 Highlights

- **Zero Terminal Required**: Includes a native double-clickable macOS application bundle (`VideoTo3D.app`) and local background server.
- **Solid 3D Mesh Generation (No Black Dots!)**:
  - Uses AI neural depth estimation (**Depth-Anything-V2**) accelerated via Apple Silicon Metal (`mps`).
  - Assembles real continuous polygonal geometry with photographic texture atlases.
  - Native export to `.GLB` (self-contained binary glTF), `.OBJ` + `.MTL`, and `.FBX` ready for **Blender**, **Unity**, **Unreal Engine**, and 3D printing.
- **3D Gaussian Splatting (3DGS)**:
  - Structure-from-Motion (SfM) camera tracking via COLMAP.
  - Outputs production-ready `.PLY` radiance fields compatible with **SuperSplat** and the **Blender KIRI 3DGS Addon**.
- **Embedded WebGL 3D Viewer**:
  - Built-in Three.js viewport to orbit, pan, zoom, inspect wireframes, and verify textures instantly inside the browser.
- **Native macOS Integration**:
  - One-click native Finder file picker (`osascript`).
  - One-click **"Reveal in Finder"** and **"Open in Blender"** buttons.
  - Fully offline capable.

---

## 📁 Repository Structure

```
VideoTo3D/
├── VideoTo3D.app/                 # Native double-clickable macOS Application Bundle
├── Launch VideoTo3D.command       # Double-clickable Desktop shell launcher
├── server.py                      # Flask background server & REST API
├── engine.py                      # Apple Silicon 3D mesh & 3DGS pipeline processor
├── templates/
│   └── index.html                 # Apple-style Glassmorphic Dark Mode Web Studio UI
├── static/
│   └── js/
│       ├── three.min.js           # Offline 3D WebGL renderer
│       ├── OrbitControls.js       # Camera navigation controls
│       └── GLTFLoader.js          # Binary glTF model loader
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### Option 1: Double-Click (No Terminal)
1. Double-click `VideoTo3D.app` (or `Launch VideoTo3D.command`).
2. Your default browser opens automatically to `http://127.0.0.1:5050`.
3. Choose your video file, select **Solid 3D Mesh** or **3D Gaussian Splat**, and click **Convert Video to 3D World**!

### Option 2: Command Line
```bash
# Clone the repository
git clone https://github.com/mohamadjlo/VideoTo3D.git
cd VideoTo3D

# Install dependencies
pip install -r requirements.txt

# Launch server
python server.py 5050
```

---

## 🛠️ Requirements

- **Operating System**: macOS 12 Monterey or later (Apple Silicon M1/M2/M3/M4 recommended)
- **Dependencies**: `ffmpeg`, `colmap`
- **Python**: 3.10+ (PyTorch with Metal MPS support)

---

## 📄 License
MIT License
