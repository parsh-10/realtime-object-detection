# 🎯 Real-Time Object Detection with YOLOv11 + COCO

Fine-tuned **YOLOv11s** on the COCO128 dataset to detect **80 object classes** in real time. Built with a Gradio web interface supporting both image upload and live webcam detection.

---

## 📊 Model Comparison — YOLOv8s vs YOLOv11s

| Metric | YOLOv8s | YOLOv11s (current) | Winner |
|--------|---------|-------------------|--------|
| **mAP@50** | 92.1% | **92.7%** | 🏆 YOLOv11 |
| **mAP@50-95** | 77.6% | **78.9%** | 🏆 YOLOv11 |
| **Precision** | 94.0% | 92.8% | 🏆 YOLOv8 |
| **Recall** | 85.3% | 85.3% | 🤝 Tie |
| **Parameters** | 11.1M | **9.4M** | 🏆 YOLOv11 |
| **GFLOPs** | 28.6 | **21.5** | 🏆 YOLOv11 |
| **Inference Speed** | 5.2ms | 5.6ms | 🏆 YOLOv8 |

> YOLOv11s achieves **higher accuracy with fewer parameters** — making it more efficient for real-world deployment.

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

## ✨ Features

- 📷 **Image Detection** — Upload any image and detect objects instantly
- 🎥 **Live Webcam** — Real-time detection through your webcam feed
- 80 **Object Classes** — People, vehicles, animals, food, everyday items & more

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
| **Model** | YOLOv11s (Ultralytics) |
| **Framework** | PyTorch |
| **Demo UI** | Gradio |
| **Training** | Google Colab (Tesla T4 GPU) |
| **Image Processing** | OpenCV, Pillow |

---

## 📁 Project Structure

```
realtime-object-detection/
├── app.py                  # Gradio web demo (image + webcam)
├── yolo11_best.pt          # Fine-tuned YOLOv11s weights
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🔍 Model Details

- **Architecture:** YOLOv11s — 101 layers, 9.4M parameters, 21.5 GFLOPs
- **Training:** 50 epochs, image size 640×640, batch size 16
- **Base Weights:** Pretrained on COCO, fine-tuned on COCO128
- **Best Performing Classes:** Airplane (99.5%), Bus (99.5%), Cat (99.5%), Pizza (99.5%)

---

## ⚠️ Known Limitations

- Detects only COCO's 80 classes — domain-specific objects (e.g. pens, erasers, cricket balls) may be misclassified
- Performance drops on heavily occluded or very small objects
- **Planned fix:** Custom dataset fine-tuning for domain-specific objects

---

## 🗺️ Roadmap

- [x] Fine-tune YOLOv8s on COCO128 (mAP@50: 92.1%)
- [x] Fine-tune YOLOv11s on COCO128 (mAP@50: 92.7%)
- [x] Build Gradio web demo with image + webcam support
- [x] Model comparison (YOLOv8 vs YOLOv11)
- [ ] Custom dataset training (pens, erasers & domain-specific objects)
- [ ] Fine-tune on full COCO val2017 dataset
- [ ] Deploy permanently on Hugging Face Spaces

---

## 👤 Author

**Parsh** — [@parsh-10](https://github.com/parsh-10)
