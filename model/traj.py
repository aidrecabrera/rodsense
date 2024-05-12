import json
import os
import csv
from datetime import datetime
import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame
from supabase import create_client, Client
import supervision as sv

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

annotator = sv.BoxAnnotator()

trajectory = {}

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # labels for each prediction
    labels = [f"Detected: {p['class']}, X: {p['x']}, Y: {p['y']}" for p in predictions["predictions"]]
    # load our predictions into the supervision detections object
    detections = sv.Detections.from_inference(predictions)
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    # check if predictions are empty
    if len(predictions["predictions"]) == 0:
        return
    else:
        # save data to supabase database
        for data in predictions["predictions"]:
            # Use a tuple of class and rounded coordinates as a key to distinguish different objects
            key = (data['class'], round(data['x']), round(data['y']))
            if key not in trajectory:
                trajectory[key] = []
            trajectory[key].append((data["x"], data["y"]))
            supabase.table("detections").insert(
                [
                    {
                        "x": data["x"],
                        "y": data["y"],
                        "width": data["width"],
                        "height": data["height"],
                        "confidence": data["confidence"],
                        "class": data["class"],
                        "time_detected": datetime.now().isoformat(),
                    }
                ]
            ).execute()
    
    for traj in trajectory.values():
        for i in range(1, len(traj)):
            cv2.line(image, (int(traj[i-1][0]), int(traj[i-1][1])), 
                     (int(traj[i][0]), int(traj[i][1])), 
                     (0, 255, 0), 2)
    
    # save the annotated image to a file
    cv2.imwrite("annotated.jpg", image)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

# Initialize inference pipeline
pipeline = InferencePipeline.init(
    model_id="cuterat-sense/1",
    video_reference=0,
    on_prediction=my_custom_sink,
)

# Start pipeline
pipeline.start()
pipeline.join()
