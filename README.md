# FLASH-POINT: Distributed Gaze-Contingent Content Moderation

**FLASH-POINT** is a real-time Big Data pipeline designed to analyze human visual saliency and provide immediate behavioral intervention. By utilizing distributed stream processing, the system maps human gaze telemetry against dynamic object detection to trigger a visual "flash" when a user focuses on restricted or sensitive regions of interest (SROI).

## 🚀 The Big Data Architecture
The system utilizes a **Lambda-style architecture** to handle high-velocity telemetry:

1. **Ingestion Layer:** OpenCV and MediaPipe estimate gaze coordinates ($x, y$) at 60Hz. YOLOv8 identifies SROI bounding boxes.
2. **Message Broker:** Apache Kafka decouples the vision producers from the analytical consumers.
3. **Processing Layer:** Apache Spark Structured Streaming performs a spatial join to detect intersections between gaze points and object polygons.
4. **Intervention Layer:** A low-latency consumer triggers a full-screen visual override (The Flash) upon violation detection.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Vision:** OpenCV, MediaPipe (Iris Tracking), Ultralytics (YOLOv8)
* **Data Stream:** Apache Kafka
* **Analytics:** PySpark (Spark Streaming)
* **Infrastructure:** Docker & Docker Compose

## 📊 Logic & Spatial Join
The system evaluates the violation condition using the following logic:
$$Gaze(x, y) \in [x_{min}, y_{min}, x_{max}, y_{max}]$$
Where the intervention is triggered if the gaze coordinate resides within the dynamically updated bounding box of a detected sensitive object.

## 🏃 Setup Instructions
1. **Start Infrastructure:**
   `docker-compose -f docker/docker-compose.yml up -d`
2. **Install Dependencies:**
   `pip install -r requirements.txt`
3. **Run Pipeline:**
   * Start the Spark Processor: `python src/processors/spark_analytics.py`
   * Start the Flash Consumer: `python src/consumers/flash_trigger.py`
   * Start the Vision Producer: `python src/producers/vision_node.py`

## Core Logic for your Coding Agent
If you are using a coding agent, give it this specific prompt for the Spark Spatial Join (the most important part for your grade):

"Write a PySpark Structured Streaming script that reads a JSON stream from a Kafka topic named gaze_events. The JSON contains gaze_x, gaze_y, and a list of detected_objects with bbox coordinates. Filter the stream to only show events where the gaze point is inside any bounding box labeled 'sensitive'. Sink the output to a local console and a second Kafka topic named intervention_trigger."

## How to Create the Public Repo (CLI)
Once you have the folder ready, run these commands in your terminal:

cd flash-point

git init

git add .

git commit -m "Initial architecture for Flash-Point Big Data project"

Go to GitHub.com, create a new repo named flash-point, and then:

git remote add origin https://github.com/YOUR_USERNAME/flash-point.git

git branch -M main

git push -u origin main
