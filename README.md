# RoadGuard AI

RoadGuard AI detects potholes and speed breakers in images, live camera frames, and videos using a local YOLO model. Image and video results are served by the FastAPI application.

## Requirements

- Python 3.10 or newer
- The model file at `model/best.pt` (included in this repository)

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Run

Start the development server from the project root:

```bash
python -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. The API health check is available at `/api/health`.

The app selects CUDA when a compatible GPU is available and otherwise runs on CPU. Uploads and generated results are stored locally in `uploads/` and `outputs/`; these runtime files are excluded from Git.