import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

model = YOLO("yolo11_best.pt")

def detect_image(image):
    results = model.predict(image, conf=0.25)
    annotated = results[0].plot()
    return Image.fromarray(annotated)

def detect_video(frame):
    results = model.predict(frame, conf=0.15, verbose=False)
    annotated = results[0].plot()
    return annotated

with gr.Blocks(title="Real-Time Object Detection — YOLOv11 + COCO") as demo:
    gr.Markdown(" Real-Time Object Detection — YOLOv11 + COCO")
    gr.Markdown("Detects 80 object classes | mAP@50: 92.7% | Fine-tuned on COCO128")

    with gr.Tabs():
        with gr.Tab("📷 Image Detection"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image")
                img_output = gr.Image(type="pil", label="Detected Objects")
            img_btn = gr.Button("Detect", variant="primary")
            img_btn.click(detect_image, inputs=img_input, outputs=img_output)

        with gr.Tab(" Live Webcam"):
            gr.Interface(
                fn=detect_video,
                inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
                outputs=gr.Image(type="numpy"),
                live=True,
                flagging_mode="never"
            )

demo.launch()
