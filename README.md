# 🥬 Phân loại Rau Sạch & Rau Hỏng bằng Deep Learning

> Hệ thống Computer Vision phân loại ảnh rau thành **Fresh** (tươi) và **Rotten** (hỏng), sử dụng **Transfer Learning** với MobileNetV2 và ResNet50.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📋 Mục lục
1. [Tổng quan](#-tổng-quan)
2. [Cấu trúc project](#-cấu-trúc-project)
3. [Yêu cầu phần cứng](#-yêu-cầu-phần-cứng)
4. [Cài đặt](#-cài-đặt)
5. [Hướng dẫn chạy từng bước](#-hướng-dẫn-chạy-từng-bước)
6. [Kết quả](#-kết-quả-mong-đợi)
7. [Tham khảo](#-tham-khảo)

---

## 🎯 Tổng quan

| Hạng mục | Chi tiết |
|---|---|
| **Bài toán** | Image Classification (Binary: fresh / rotten) |
| **Framework** | TensorFlow / Keras 2.15+ |
| **Mô hình** | MobileNetV2 + ResNet50 (Transfer Learning) |
| **Dataset** | ~10.000 ảnh crawl từ Google Images |
| **Mục tiêu** | Accuracy ≥ 90% |
| **Image size** | 224 × 224 × 3 |

## 📁 Cấu trúc project

```
project/
├── dataset/              # Dữ liệu (raw / processed / train / valid / test)
├── crawler/              # Thu thập ảnh tự động
│   └── crawl_images.py
├── preprocessing/        # Tiền xử lý + augmentation
│   ├── preprocess.py
│   └── augmentation.py
├── models/               # Định nghĩa & huấn luyện
│   ├── mobilenet_model.py
│   ├── resnet_model.py
│   └── train.py
├── evaluation/           # Đánh giá kết quả
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   └── plots.py
├── notebook/experiment.ipynb
├── app/predict.py        # Inference
├── report/report_outline.md
├── requirements.txt
├── README.md
└── CLAUDE.md             # Bộ nhớ dự án (đọc trước khi làm việc)
```

## 💻 Yêu cầu phần cứng

| Cấu hình | Khuyến nghị | Tối thiểu |
|---|---|---|
| **CPU** | 4+ nhân | 2 nhân |
| **RAM** | 16 GB | 8 GB |
| **GPU** | NVIDIA 4–6 GB VRAM (RTX 3050+) | Có thể chạy CPU |
| **Disk** | 10 GB trống | 5 GB |
| **Python** | 3.10 | 3.9 – 3.11 |

> 💡 Dự án đã tối ưu cho **laptop GPU yếu**: dùng MobileNetV2, batch nhỏ, mixed precision tùy chọn.

## ⚙️ Cài đặt

```bash
# 1. Clone repo
git clone <your-repo-url>
cd project

# 2. Tạo virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3. Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Tùy chọn) Kiểm tra GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 🚀 Hướng dẫn chạy từng bước

> **Toàn bộ dự án gói gọn trong 2 notebook**. Mở `jupyter lab` (hoặc Jupyter Notebook), chạy lần lượt:

### 📓 Notebook 1 — `notebook/01_prepare.ipynb` (chuẩn bị dữ liệu)

Notebook này thực hiện end-to-end khâu chuẩn bị, **TỰ CRAWL** ảnh từ web theo yêu cầu đề bài:
1. Setup môi trường (kiểm tra TF, GPU)
2. Cài dependencies
3. **Crawl ảnh** từ Google + Bing + Baidu Images (41 keyword Anh + Việt)
4. Sanity check: đếm ảnh theo keyword, kích thước, file lỗi
5. Tiền xử lý: resize 224×224, loại trùng, split 70/15/15
6. Trực quan hoá: ảnh mẫu, phân bố class, augmentation demo

```bash
jupyter lab notebook/01_prepare.ipynb
# Crawl 10k ảnh: ~30-60 phút lần đầu (tuỳ tốc độ mạng)
```

### 📓 Notebook 2 — `notebook/02_train_and_evaluate.ipynb` (thực thi & kết quả)

Notebook này thực hiện huấn luyện và đánh giá đầy đủ:
1. Load dataset đã chuẩn bị
2. **Train MobileNetV2** (transfer learning, 2 pha freeze→fine-tune)
3. **Train ResNet50** (cùng chiến lược)
4. Đánh giá test set: accuracy, precision, recall, F1
5. Sinh **7 nhóm biểu đồ phong phú**:
   - Training curves (accuracy + loss)
   - Confusion matrix (raw + normalized) cho 2 model
   - Classification report heatmap
   - ROC curve + AUC
   - Bar chart so sánh metrics
   - Lưới ảnh bị phân loại sai (phân tích lỗi)
   - Demo predict trên 6 ảnh test
6. Lưu tất cả output vào `results/*.png` để chèn báo cáo

```bash
jupyter lab notebook/02_train_and_evaluate.ipynb
# → Run All Cells, mất ~1-3 giờ tuỳ phần cứng
```

### 💡 Yêu cầu

```bash
pip install -r requirements.txt
```

## 📊 Kết quả mong đợi

| Model | Params | Accuracy | F1 | Train time (RTX 3050) |
|---|---|---|---|---|
| MobileNetV2 | ~2.3M | ~92% | 0.92 | ~25 phút |
| ResNet50    | ~23.6M | ~94% | 0.94 | ~50 phút |

## ☁️ Chạy trên Google Colab (khuyến nghị nếu máy yếu)

Mở `notebook/colab_train.ipynb` trong Colab — đã setup sẵn end-to-end:

1. `Runtime → Change runtime type → GPU (T4)`
2. Chạy lần lượt từng cell. Notebook sẽ:
   - Mount Drive (`/content/drive/MyDrive/veggie_project/`)
   - Symlink `dataset/`, `checkpoints/`, `logs/`, `results/` vào Drive — KHÔNG mất khi disconnect
   - Cài thêm `icrawler imagehash tqdm` (Colab đã có sẵn TF, numpy, sklearn, ...)
   - Train với **mixed precision (FP16) + XLA** → ~1.5–2× nhanh hơn
   - TensorBoard inline để theo dõi

Cờ tối ưu Colab:
```bash
python models/train.py --model mobilenet --epochs 30 \
    --batch-size 64 --mixed-precision --xla
```

| Hardware Colab | Batch | Mixed precision | Train time MobileNet |
|---|---|---|---|
| T4 (Free)     | 64 | ✅ | ~10–15 phút |
| V100 (Pro)    | 128 | ✅ | ~5–8 phút   |
| A100 (Pro+)   | 256 | ✅ | ~3–5 phút   |

## 📚 Tham khảo

- He et al. — *Deep Residual Learning for Image Recognition* (2016)
- Sandler et al. — *MobileNetV2: Inverted Residuals* (2018)
- Keras Applications: <https://keras.io/api/applications/>

---

> 📝 **Lưu ý**: Đọc `CLAUDE.md` để hiểu rõ ngữ cảnh & quyết định kiến trúc của dự án trước khi sửa code.
