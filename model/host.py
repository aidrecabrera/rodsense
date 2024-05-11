import json
import os
import csv
from datetime import datetime
import cv2
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame
import numpy as np
from supabase import create_client, Client
import supervision as sv

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ANNOTATED_IMAGE_FILEPATH = "annotated.jpg"

class DatabaseHandler:
    def __init__(self, url: str, key: str):
        self.supabase = create_client(url, key)

    def insert_detection(self, detection: dict):
        try:
            self.supabase.table("detections").insert([detection]).execute()
        except Exception as e:
            print(f"Error inserting detection into database: {e}")

class ImageAnnotator:
    def __init__(self):
        self.annotator = sv.BoxAnnotator()

    def annotate_image(self, image: np.ndarray, detections: list, labels: list) -> np.ndarray:
        return self.annotator.annotate(image, detections, labels)

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    labels = [f"Detected: {p['class']}, X: {p['x']}, Y: {p['y']}" for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    image = video_frame.image.copy()
    annotated_image = annotator.annotate_image(image, detections, labels)

    if len(predictions["predictions"]) == 0:
        return

    debounce_timer = datetime.now()
    debounce_interval = 1

    if (datetime.now() - debounce_timer).total_seconds() >= debounce_interval:
        debounce_timer = datetime.now()
        detections_to_insert = []
        for data in predictions["predictions"]:
            detection = {
                "x": data["x"],
                "y": data["y"],
                "width": data["width"],
                "height": data["height"],
                "confidence": data["confidence"],
                "class": data["class"],
                "time_detected": datetime.now().isoformat(),
            }
            detections_to_insert.append(detection)

        try:
            database_handler.insert_detections(detections_to_insert)
        except Exception as e:
            print(f"Error inserting detections into database: {e}")

    cv2.imwrite(ANNOTATED_IMAGE_FILEPATH, annotated_image)
    cv2.imshow("Predictions", annotated_image)
    cv2.waitKey(1)

database_handler = DatabaseHandler(SUPABASE_URL, SUPABASE_KEY)
annotator = ImageAnnotator()

pipeline = InferencePipeline.init(
    model_id="cuterat-sense/1",
    video_reference=0,
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()