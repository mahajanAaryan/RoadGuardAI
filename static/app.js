// ============================================================
// ELEMENTS
// ============================================================

const imageInput =
    document.getElementById("imageInput");

const videoInput =
    document.getElementById("videoInput");

const uploadButton =
    document.getElementById("uploadButton");

const uploadVideoButton =
    document.getElementById("uploadVideoButton");

const imageModeButton =
    document.getElementById("imageModeButton");

const videoModeButton =
    document.getElementById("videoModeButton");

const cameraModeButton =
    document.getElementById("cameraModeButton");

const startCameraButton =
    document.getElementById("startCameraButton");

const stopCameraButton =
    document.getElementById("stopCameraButton");

const cameraStage =
    document.getElementById("cameraStage");

const cameraPreview =
    document.getElementById("cameraPreview");

const cameraOverlay =
    document.getElementById("cameraOverlay");

const cameraContext =
    cameraOverlay.getContext("2d");

const resultImage =
    document.getElementById("resultImage");

const resultVideo =
    document.getElementById("resultVideo");

const placeholder =
    document.getElementById("placeholder");

const confidence =
    document.getElementById("confidence");

const confidenceValue =
    document.getElementById("confidenceValue");

const potholeCount =
    document.getElementById("potholeCount");

const speedBreakerCount =
    document.getElementById("speedBreakerCount");

const totalCount =
    document.getElementById("totalCount");

const inferenceTime =
    document.getElementById("inferenceTime");

const detectionList =
    document.getElementById("detectionList");

const processingStatus =
    document.getElementById("processingStatus");

const device =
    document.getElementById("device");

const statusText =
    document.getElementById("statusText");

const statusDot =
    document.getElementById("statusDot");


let currentMode = "image";
let cameraStream = null;
let liveCameraActive = false;
let cameraAnimationId = null;
let cameraDetections = [];
let cameraFrameWidth = 1;
let cameraFrameHeight = 1;


// ============================================================
// CONFIDENCE SLIDER
// ============================================================

confidence.addEventListener(
    "input",
    () => {

        const value =
            Math.round(
                parseFloat(
                    confidence.value
                ) * 100
            );

        confidenceValue.textContent =
            `${value}%`;
    }
);


// ============================================================
// MODE SELECTION
// ============================================================

function selectDetectionMode(mode) {
    currentMode = mode;
    stopLiveCamera();

    imageModeButton.classList.toggle("active", mode === "image");
    videoModeButton.classList.toggle("active", mode === "video");
    cameraModeButton.classList.toggle("active", mode === "camera");

    uploadButton.classList.toggle("hidden", mode !== "image");
    uploadVideoButton.classList.toggle("hidden", mode !== "video");
    startCameraButton.classList.toggle("hidden", mode !== "camera");
    stopCameraButton.classList.add("hidden");

    resultImage.classList.add("hidden");
    resultVideo.classList.add("hidden");
    cameraStage.classList.add("hidden");
    placeholder.classList.remove("hidden");

    const placeholderText = placeholder.querySelector("p");
    placeholderText.textContent = mode === "camera"
        ? "Press Start Camera and allow camera access"
        : mode === "video"
            ? "Upload a road video to analyze"
            : "Upload a road image to analyze";

    processingStatus.textContent = "Ready";
    clearDetectionStats();
}

imageModeButton.addEventListener("click", () => selectDetectionMode("image"));
videoModeButton.addEventListener("click", () => selectDetectionMode("video"));
cameraModeButton.addEventListener("click", () => selectDetectionMode("camera"));
startCameraButton.addEventListener("click", startLiveCamera);
stopCameraButton.addEventListener("click", stopLiveCamera);

function clearDetectionStats() {
    potholeCount.textContent = "0";
    speedBreakerCount.textContent = "0";
    totalCount.textContent = "0";
    inferenceTime.textContent = "-";
    detectionList.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty-message";
    empty.textContent = "No detections yet.";
    detectionList.appendChild(empty);
}

function updateCameraResults(data, cycleMs) {
    potholeCount.textContent = data.counts.potholes;
    speedBreakerCount.textContent = data.counts.speed_breakers;
    totalCount.textContent = data.counts.total;
    const cameraFps = cycleMs > 0 ? (1000 / cycleMs).toFixed(1) : "0.0";
    inferenceTime.textContent = `${data.inference_time_ms} ms / ${cameraFps} FPS`;

    detectionList.replaceChildren();
    if (!data.detections.length) {
        const empty = document.createElement("p");
        empty.className = "empty-message";
        empty.textContent = "No hazards detected in the latest camera frame.";
        detectionList.appendChild(empty);
        return;
    }

    for (const detection of data.detections) {
        const item = document.createElement("div");
        item.className = "detection-item";
        const name = document.createElement("span");
        name.className = "detection-name";
        name.textContent = detection.class_name;
        const score = document.createElement("span");
        score.className = "detection-confidence";
        score.textContent = `${Math.round(detection.confidence * 100)}%`;
        item.append(name, score);
        detectionList.appendChild(item);
    }
}

async function startLiveCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        processingStatus.textContent = "Camera requires localhost or a secure connection";
        return;
    }

    startCameraButton.disabled = true;
    processingStatus.textContent = "Requesting camera access...";

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false
        });
        cameraPreview.srcObject = cameraStream;
        await cameraPreview.play();

        if (!cameraPreview.videoWidth || !cameraPreview.videoHeight) {
            await new Promise(resolve => {
                cameraPreview.addEventListener("loadedmetadata", resolve, { once: true });
            });
        }

        cameraOverlay.width = cameraPreview.videoWidth;
        cameraOverlay.height = cameraPreview.videoHeight;
        cameraFrameWidth = cameraPreview.videoWidth;
        cameraFrameHeight = cameraPreview.videoHeight;
        cameraStage.classList.remove("hidden");
        placeholder.classList.add("hidden");
        startCameraButton.classList.add("hidden");
        startCameraButton.disabled = false;
        stopCameraButton.classList.remove("hidden");
        liveCameraActive = true;
        cameraDetections = [];
        processingStatus.textContent = "Camera live - analyzing frames";
        drawCameraOverlay();
        processLiveCameraFrames();
    } catch (error) {
        console.error(error);
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        startCameraButton.disabled = false;
        processingStatus.textContent = `Camera unavailable: ${error.message}`;
    }
}

function stopLiveCamera() {
    liveCameraActive = false;
    if (cameraAnimationId !== null) {
        cancelAnimationFrame(cameraAnimationId);
        cameraAnimationId = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraPreview.srcObject = null;
    cameraStage.classList.add("hidden");
    cameraContext.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
    startCameraButton.disabled = false;
    startCameraButton.classList.toggle("hidden", currentMode !== "camera");
    stopCameraButton.classList.add("hidden");
    if (currentMode === "camera") {
        placeholder.classList.remove("hidden");
        processingStatus.textContent = "Camera stopped";
    }
}

function drawCameraOverlay() {
    if (!liveCameraActive) return;

    cameraContext.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
    const scaleX = cameraOverlay.width / cameraFrameWidth;
    const scaleY = cameraOverlay.height / cameraFrameHeight;
    const lineWidth = Math.max(2, cameraOverlay.width / 500);
    cameraContext.lineWidth = lineWidth;
    cameraContext.font = `bold ${Math.max(14, cameraOverlay.width / 45)}px Arial`;

    for (const detection of cameraDetections) {
        const [x1, y1, x2, y2] = detection.box;
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;
        const color = detection.class_name === "potholes" ? "#fb7185" : "#fbbf24";
        const label = `${detection.class_name} ${Math.round(detection.confidence * 100)}%`;

        cameraContext.strokeStyle = color;
        cameraContext.strokeRect(left, top, width, height);
        const labelWidth = cameraContext.measureText(label).width + 12;
        const labelTop = Math.max(0, top - 26);
        cameraContext.fillStyle = color;
        cameraContext.fillRect(left, labelTop, labelWidth, 26);
        cameraContext.fillStyle = "#101522";
        cameraContext.fillText(label, left + 6, labelTop + 18);
    }

    cameraAnimationId = requestAnimationFrame(drawCameraOverlay);
}

async function processLiveCameraFrames() {
    const frameCanvas = document.createElement("canvas");
    const frameContext = frameCanvas.getContext("2d");

    try {
        while (liveCameraActive) {
            if (cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                await new Promise(resolve => setTimeout(resolve, 100));
                continue;
            }

            const scale = Math.min(1, 640 / cameraPreview.videoWidth);
            frameCanvas.width = Math.round(cameraPreview.videoWidth * scale);
            frameCanvas.height = Math.round(cameraPreview.videoHeight * scale);
            frameContext.drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);

            const frameBlob = await new Promise(resolve =>
                frameCanvas.toBlob(resolve, "image/jpeg", 0.78)
            );
            if (!frameBlob) throw new Error("Could not capture a camera frame.");

            const formData = new FormData();
            formData.append("file", frameBlob, "camera-frame.jpg");
            formData.append("confidence", confidence.value);
            const cycleStarted = performance.now();
            const response = await fetch("/api/predict/frame", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.detail || data.error || "Live detection failed.");
            }
            if (!liveCameraActive) break;

            cameraFrameWidth = data.frame_width;
            cameraFrameHeight = data.frame_height;
            cameraDetections = data.detections || [];
            updateCameraResults(data, performance.now() - cycleStarted);
            processingStatus.textContent = `Camera live - ${data.counts.total} hazard(s) in latest frame`;
            await new Promise(resolve => setTimeout(resolve, 120));
        }
    } catch (error) {
        console.error(error);
        if (liveCameraActive) {
            stopLiveCamera();
            processingStatus.textContent = `Camera detection stopped: ${error.message}`;
        }
    }
}

window.addEventListener("beforeunload", stopLiveCamera);


// IMAGE UPLOAD BUTTON
// ============================================================

uploadButton.addEventListener(
    "click",
    () => {

        imageInput.click();

    }
);


// ============================================================
// IMAGE SELECTED
// ============================================================

imageInput.addEventListener(
    "change",
    async () => {

        if (!imageInput.files.length) {
            return;
        }

        await detectImage(
            imageInput.files[0]
        );

    }
);


// ============================================================
// VIDEO UPLOAD BUTTON
// ============================================================

uploadVideoButton.addEventListener(
    "click",
    () => {

        videoInput.click();

    }
);


// ============================================================
// VIDEO SELECTED
// ============================================================

videoInput.addEventListener(
    "change",
    async () => {

        if (!videoInput.files.length) {
            return;
        }

        await detectVideo(
            videoInput.files[0]
        );

    }
);


// ============================================================
// IMAGE DETECTION
// ============================================================

async function detectImage(file) {

    uploadButton.disabled = true;

    processingStatus.textContent =
        "Running YOLO...";


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    formData.append(
        "confidence",
        confidence.value
    );


    try {

        const response =
            await fetch(
                "/api/predict/image",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Image detection failed."
            );
        }


        // Show result

        resultImage.src =
            data.image_url +
            "?t=" +
            Date.now();


        resultImage.classList.remove(
            "hidden"
        );


        resultVideo.classList.add(
            "hidden"
        );


        placeholder.classList.add(
            "hidden"
        );


        // Statistics

        potholeCount.textContent =
            data.counts.potholes;


        speedBreakerCount.textContent =
            data.counts.speed_breakers;


        totalCount.textContent =
            data.counts.total;


        inferenceTime.textContent =
            `${data.inference_time_ms} ms`;


        // Detection details

        detectionList.innerHTML = "";


        if (
            data.detections.length === 0
        ) {

            detectionList.innerHTML = `
                <p class="empty-message">
                    No hazards detected.
                </p>
            `;

        } else {

            data.detections.forEach(
                detection => {

                    const item =
                        document.createElement(
                            "div"
                        );


                    item.className =
                        "detection-item";


                    const conf =
                        Math.round(
                            detection.confidence *
                            100
                        );


                    item.innerHTML = `

                        <span
                            class="detection-name"
                        >
                            ${detection.class_name}
                        </span>

                        <span
                            class="detection-confidence"
                        >
                            ${conf}%
                        </span>

                    `;


                    detectionList.appendChild(
                        item
                    );

                }
            );
        }


        processingStatus.textContent =
            "Detection complete";


    } catch (error) {

        console.error(error);

        alert(error.message);

        processingStatus.textContent =
            "Error";


    } finally {

        uploadButton.disabled = false;

    }
}


// ============================================================
// VIDEO DETECTION
// ============================================================

async function detectVideo(file) {

    uploadVideoButton.disabled = true;

    processingStatus.textContent =
        "Processing video...";


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    formData.append(
        "confidence",
        confidence.value
    );


    try {

        const response =
            await fetch(
                "/api/predict/video",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Video processing failed."
            );
        }


        // Show video

        resultVideo.src =
            data.video_url +
            "?t=" +
            Date.now();


        resultVideo.classList.remove(
            "hidden"
        );


        resultImage.classList.add(
            "hidden"
        );


        placeholder.classList.add(
            "hidden"
        );


        // Statistics

        potholeCount.textContent =
            data.counts.potholes;


        speedBreakerCount.textContent =
            data.counts.speed_breakers;


        totalCount.textContent =
            data.counts.total;


        inferenceTime.textContent =
            `${data.processing_fps} FPS`;


        // Details

        detectionList.innerHTML = `

            <div class="detection-item">

                <span class="detection-name">
                    Frames Processed
                </span>

                <span class="detection-confidence">
                    ${data.frames_processed}
                </span>

            </div>


            <div class="detection-item">

                <span class="detection-name">
                    Video FPS
                </span>

                <span class="detection-confidence">
                    ${data.video_fps}
                </span>

            </div>


            <div class="detection-item">

                <span class="detection-name">
                    Processing FPS
                </span>

                <span class="detection-confidence">
                    ${data.processing_fps}
                </span>

            </div>


            <div class="detection-item">

                <span class="detection-name">
                    Processing Time
                </span>

                <span class="detection-confidence">
                    ${data.processing_time_seconds} sec
                </span>

            </div>

        `;


        processingStatus.textContent =
            "Video processing complete";


    } catch (error) {

        console.error(error);

        alert(error.message);

        processingStatus.textContent =
            "Error";


    } finally {

        uploadVideoButton.disabled =
            false;

    }
}


// ============================================================
// HEALTH CHECK
// ============================================================

async function checkHealth() {

    try {

        const response =
            await fetch(
                "/api/health"
            );


        const data =
            await response.json();


        if (data.status === "online") {

            statusDot.classList.add(
                "online"
            );


            statusText.textContent =
                "System Online";


            device.textContent =
                data.device;

        }

    } catch (error) {

        statusDot.classList.remove(
            "online"
        );


        statusText.textContent =
            "System Offline";


        device.textContent =
            "Unavailable";
    }
}


checkHealth();