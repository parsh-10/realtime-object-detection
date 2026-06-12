# 🎯 Real-Time Object Detection with YOLOv8 + COCO

Fine-tuned **YOLOv8s** on the COCO128 dataset to detect **80 object classes** in real time. Built with a Gradio web interface for easy image upload and instant inference.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **mAP@50** | 92.1% |
| **mAP@50-95** | 77.6% |
| **Precision** | 94.0% |
| **Recall** | 85.3% |
| **Inference Speed** | 5.2ms per image (T4 GPU) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/parsh-10/realtime-object-detection
cd realtime-object-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the demo
python app.py
```

Then open `http://localhost:7860` in your browser.

---

> **Sample detections:** person, car, bus, dog, cat, bicycle, airplane, bottle, chair, and 71 more classes.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Dataset** | COCO128 (subset of MS-COCO) |
| **Full Dataset Size** | ~25 GB / 1.5M images |
| **Subset Used** | 128 images, 80 classes |
| **Classes** | Person, vehicle, animal, food, furniture & more |

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| **Model** | YOLOv8s (Ultralytics) |
| **Framework** | PyTorch |
| **Demo UI** | Gradio |
| **Training** | Google Colab (Tesla T4 GPU) |
| **Image Processing** | OpenCV, Pillow |

---

## 📁 Project Structure

```
realtime-object-detection/
├── app.py                  # Gradio web demo
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🔍 Model Details

- **Architecture:** YOLOv8s (small variant — 11.1M parameters, 28.6 GFLOPs)
- **Training:** 50 epochs, image size 640×640, batch size 16
- **Base Weights:** Pretrained on COCO, fine-tuned on COCO128
- **Best Performing Classes:** Airplane (99.5%), Bus (99.5%), Stop Sign (99.5%), Cat (99.5%)

---

## ⚠️ Known Limitations

- Trained on COCO's 80 classes — domain-specific objects (e.g. cricket balls) may be misclassified
- Performance drops on heavily occluded or very small objects
- Planned improvement: fine-tune on full COCO val2017 dataset

---

## 🗺️ Roadmap

- [x] Fine-tune YOLOv8s on COCO128
- [x] Build Gradio web demo
- [ ] Add live webcam / video stream detection
- [ ] Fine-tune on full COCO val2017 dataset
- [ ] Deploy permanently on Hugging Face Spaces

---

## 👤 Author

**Parsh** — [@parsh-10](https://github.com/parsh-10)
