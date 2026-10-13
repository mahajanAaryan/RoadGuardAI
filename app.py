from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

from pathlib import Path
import shutil
import uuid
import time
import cv2
import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("ROADGUARD AI")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")
print(f"Device: {GPU_NAME}")

model = YOLO(str(MODEL_PATH))

print(f"Classes: {model.names}")
print("Model loaded successfully.")

print("=" * 60)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RoadGuard AI",
    description="Road hazard detection using YOLO26n",
    version="3.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "online",
        "model": "YOLO26n",
        "device": GPU_NAME,
        "classes": model.names
    }


# ============================================================
# ============================================================
# LIVE CAMERA FRAME DETECTION
# ============================================================

@app.post("/api/predict/frame")
async def predict_camera_frame(
    file: UploadFile = File(...),
    confidence: float = Form(0.50)
):
    """Analyze one in-memory webcam frame without saving it to disk."""
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1.")

    frame_bytes = await file.read()
    if not frame_bytes or len(frame_bytes) > 2_000_000:
        raise HTTPException(status_code=413, detail="Camera frame is empty or too large.")

    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode camera frame.")

    started = time.perf_counter()
    result = model.predict(
        source=frame,
        conf=confidence,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )[0]
    inference_time_ms = (time.perf_counter() - started) * 1000

    detections = []
    pothole_count = 0
    speed_breaker_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(score, 3),
                "box": [round(x1), round(y1), round(x2), round(y2)]
            })
            if class_name == "potholes":
                pothole_count += 1
            elif class_name == "speed-breakers":
                speed_breaker_count += 1

    return {
        "success": True,
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "inference_time_ms": round(inference_time_ms, 2),
        "confidence_threshold": confidence,
        "counts": {
            "potholes": pothole_count,
            "speed_breakers": speed_breaker_count,
            "total": len(detections)
        },
        "detections": detections
    }


# IMAGE DETECTION
# ============================================================

@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.50)
):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        return {
            "success": False,
            "error": "Unsupported image format."
        }


    unique_id = uuid.uuid4().hex

    input_filename = (
        f"{unique_id}{extension}"
    )

    output_filename = (
        f"{unique_id}_result.jpg"
    )

    input_path = (
        UPLOAD_DIR / input_filename
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )


    # Save uploaded image

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # Run YOLO

    start_time = time.perf_counter()

    results = model.predict(
        source=str(input_path),
        conf=confidence,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    inference_time = (
        time.perf_counter() -
        start_time
    ) * 1000

    result = results[0]


    # Detections

    detections = []

    pothole_count = 0
    speed_breaker_count = 0


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence_score = float(
                box.conf[0]
            )

            class_name = (
                model.names[class_id]
            )


            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )


            detections.append({

                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "confidence":
                    round(
                        confidence_score,
                        3
                    ),

                "box": [
                    round(x1),
                    round(y1),
                    round(x2),
                    round(y2)
                ]
            })


            if class_name == "potholes":

                pothole_count += 1


            elif class_name == "speed-breakers":

                speed_breaker_count += 1


    # Draw boxes

    annotated_image = result.plot()

    cv2.imwrite(
        str(output_path),
        annotated_image
    )


    return {

        "success": True,

        "image_url":
            f"/outputs/{output_filename}",

        "model":
            "YOLO26n",

        "device":
            GPU_NAME,

        "inference_time_ms":
            round(
                inference_time,
                2
            ),

        "confidence_threshold":
            confidence,

        "counts": {

            "potholes":
                pothole_count,

            "speed_breakers":
                speed_breaker_count,

            "total":
                len(detections)
        },

        "detections":
            detections
    }


# ============================================================
# VIDEO TRACKING
# ============================================================

@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.50)
):

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }


    extension = Path(
        file.filename
    ).suffix.lower()


    if extension not in allowed_extensions:

        return {
            "success": False,
            "error": "Unsupported video format."
        }


    # --------------------------------------------------------
    # FILE NAMES
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_filename = (
        f"{unique_id}{extension}"
    )

    output_filename = (
        f"{unique_id}_tracked.mp4"
    )


    input_path = (
        UPLOAD_DIR / input_filename
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )


    # --------------------------------------------------------
    # SAVE VIDEO
    # --------------------------------------------------------

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )


    if not cap.isOpened():

        return {
            "success": False,
            "error": "Could not open video."
        }


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if fps <= 0:

        fps = 25


    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )


    # --------------------------------------------------------
    # TRACKING STATE
    # --------------------------------------------------------

    frame_count = 0


    # Unique track IDs we've seen

    seen_pothole_ids = set()

    seen_speed_breaker_ids = set()

    seen_all_ids = set()


    start_time = time.perf_counter()


    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()


        if not success:

            break


        frame_count += 1


        # ----------------------------------------------------
        # TRACK OBJECTS
        # ----------------------------------------------------

        results = model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=confidence,

            imgsz=640,

            device=DEVICE,

            verbose=False

        )


        result = results[0]


        # ----------------------------------------------------
        # PROCESS TRACKS
        # ----------------------------------------------------

        frame_potholes = 0

        frame_speed_breakers = 0


        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):


            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )


            classes = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )


            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )


            for track_id, class_id, conf in zip(
                track_ids,
                classes,
                confidences
            ):


                class_name = (
                    model.names[class_id]
                )


                # --------------------------------------------
                # Track IDs
                # --------------------------------------------

                if class_name == "potholes":

                    seen_pothole_ids.add(
                        track_id
                    )

                    frame_potholes += 1


                elif class_name == "speed-breakers":

                    seen_speed_breaker_ids.add(
                        track_id
                    )

                    frame_speed_breakers += 1


                seen_all_ids.add(
                    track_id
                )


        # ----------------------------------------------------
        # DRAW TRACKING RESULTS
        # ----------------------------------------------------

        annotated_frame = (
            result.plot(
                labels=True,
                boxes=True
            )
        )


        # ----------------------------------------------------
        # INFORMATION OVERLAY
        # ----------------------------------------------------

        cv2.putText(

            annotated_frame,

            (
                f"RoadGuard AI | "
                f"Frame {frame_count}/{total_frames}"
            ),

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2

        )


        cv2.putText(

            annotated_frame,

            (
                f"Unique Potholes: "
                f"{len(seen_pothole_ids)}"
            ),

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            annotated_frame,

            (
                f"Unique Speed-breakers: "
                f"{len(seen_speed_breaker_ids)}"
            ),

            (20, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )


    # --------------------------------------------------------
    # CLOSE VIDEO
    # --------------------------------------------------------

    cap.release()

    writer.release()


    # --------------------------------------------------------
    # PROCESSING STATISTICS
    # --------------------------------------------------------

    processing_time = (
        time.perf_counter()
        - start_time
    )


    processing_fps = (

        frame_count /
        processing_time

        if processing_time > 0

        else 0

    )


    # --------------------------------------------------------
    # FINAL COUNTS
    # --------------------------------------------------------

    unique_potholes = (
        len(seen_pothole_ids)
    )


    unique_speed_breakers = (
        len(seen_speed_breaker_ids)
    )


    unique_total = (
        len(seen_all_ids)
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "success": True,

        "video_url":
            f"/outputs/{output_filename}",

        "model":
            "YOLO26n",

        "device":
            GPU_NAME,

        "frames_processed":
            frame_count,

        "total_frames":
            total_frames,

        "video_fps":
            round(
                fps,
                2
            ),

        "processing_fps":
            round(
                processing_fps,
                2
            ),

        "processing_time_seconds":
            round(
                processing_time,
                2
            ),

        "tracking":

            "ByteTrack",

        "counts": {

            "potholes":
                unique_potholes,

            "speed_breakers":
                unique_speed_breakers,

            "total":
                unique_total
        }
    }