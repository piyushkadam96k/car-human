# 🚀 Real-Time Person & Vehicle Detection using YOLOv8 + OpenCV

This project uses **Ultralytics YOLOv8** and **OpenCV** to perform **real-time object detection** on videos or live webcam feeds. It identifies and counts **people** and **vehicles** (cars, trucks, buses, motorcycles), displaying detection results and FPS directly on the video stream.

---

## 🧠 Features

- ✅ Real-time detection using **YOLOv8 Nano** (fastest variant).  
- 🚗 Detects and counts **people** and **vehicles**.  
- ⚡ Runs on **GPU (CUDA)** for high-speed inference.  
- 💾 Logs all detected labels with timestamps to `detected_objects.txt`.  
- 📊 Displays **FPS**, **people count**, and **vehicle count** on the screen.  

---

## 🧰 Requirements

Make sure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python torch torchvision
```

> 💡 Optional: Install CUDA-compatible PyTorch for GPU acceleration.  
> See: https://pytorch.org/get-started/locally/

---

## 📁 Project Structure

```
📂 yolo-detection/
│
├── detection.py              # Main detection script
├── detected_objects.txt      # Auto-generated log of detections
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ How It Works

1. The YOLOv8 **Nano model (`yolov8n.pt`)** is loaded.  
2. The model is moved to **GPU** (`model.to('cuda')`) for faster inference.  
3. The program opens a **video file** or **webcam feed**.  
4. Each frame is processed and analyzed:
   - Bounding boxes and class labels are drawn.
   - Counts for people and vehicles are displayed.
   - Detected objects are logged with timestamps.
5. Press **`q`** to exit the detection window.

---

## ▶️ Usage

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/piyushkadam96k/yolo-detection.git
   cd yolo-detection
   ```

2. **Run the detection script**
   ```bash
   python detection.py
   ```

3. **Input source**
   - Replace `"The Fate of the Furious ｜ Harpooning Dom's Car.mp4"`  
     with your own video path or set it to `0` for webcam:
     ```python
     cap = cv2.VideoCapture(0)
     ```

4. **Exit**
   - Press `q` anytime to close the video window.

---

## 🧾 Output Example

While running, the script shows:
```
People Count: 3
Vehicle Count: 5
FPS: 32.45
```

And logs detections like this in `detected_objects.txt`:
```
2025-10-30 12:05:43: person, car, truck
2025-10-30 12:05:44: person, motorcycle
```

---

## ⚡ Performance Tips

- For **maximum FPS**, use:
  - `yolov8n.pt` (Nano) or `yolov8s.pt` (Small)
  - Lower resolution frames (e.g., 640x400)
- Use **GPU (CUDA)** if available:
  ```python
  model.to('cuda')
  ```

---

## 🧩 Model Details

| Model | Size | Speed | mAP (50) | Best for |
|:------|:------:|:------:|:------:|:------|
| `yolov8n.pt` | 6.2 MB | ⚡ Fastest | ~37.3 | Real-time |
| `yolov8s.pt` | 21.2 MB | Fast | ~44.9 | Better accuracy |
| `yolov8m.pt` | 49.0 MB | Medium | ~50.2 | Balanced |
| `yolov8l.pt` | 83.7 MB | Slower | ~52.9 | High accuracy |

---

## 🧑‍💻 Author

**Amit Kadam**  
📧 [kadamamit462@gmail.com](mailto:kadamamit462@gmail.com)  
📍 Bhalki  
🔗 [GitHub: piyushkadam96k](https://github.com/piyushkadam96k)

---

## 📜 License

This project is open-source and free to use under the **MIT License**.
