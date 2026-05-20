# CLAUDE.md — Project Memory (Bộ nhớ dự án)

> File này giúp AI Assistant nhớ NGỮ CẢNH DỰ ÁN bất kể bao lâu sau quay lại.
> Mỗi khi vào lại dự án, **đọc file này TRƯỚC TIÊN** rồi mới làm việc.

---

## 1. Thông tin tổng quan

- **Tên đề tài**: Xây dựng hệ thống phân loại rau sạch và rau hỏng sử dụng Deep Learning
- **Loại bài toán**: Image Classification (Binary) — `fresh` vs `rotten`
- **Ngôn ngữ**: Python 3.9 – 3.11
- **Framework chính**: TensorFlow / Keras 2.15+
- **Mục tiêu**: Accuracy ≥ 90% trên test set
- **Phần cứng**: Tối ưu cho laptop GPU yếu (4–6GB VRAM, hoặc CPU)
- **Phong cách code**: Clean architecture, production-ready, có comment đầy đủ

## 2. Cấu trúc dự án

```
project/
├── dataset/              # Dữ liệu (không commit lên git)
│   ├── raw/              #   Ảnh thô crawl về (fresh/, rotten/)
│   ├── processed/        #   Ảnh đã làm sạch (loại trùng, lỗi)
│   ├── train/  valid/  test/   # Đã split sẵn
├── crawler/              # Thu thập dữ liệu
│   └── crawl_images.py
├── preprocessing/        # Tiền xử lý + augmentation
│   ├── preprocess.py
│   └── augmentation.py
├── models/               # Định nghĩa & huấn luyện mô hình
│   ├── mobilenet_model.py
│   ├── resnet_model.py
│   └── train.py
├── evaluation/           # Đánh giá & so sánh
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   └── plots.py
├── notebook/             # Notebook thử nghiệm
│   └── experiment.ipynb
├── app/                  # Inference app
│   └── predict.py
├── report/               # Báo cáo
│   └── report_outline.md
├── checkpoints/          # (sinh ra) Trọng số .h5/.keras
├── logs/                 # (sinh ra) TensorBoard logs
├── results/              # (sinh ra) plots, confusion matrices, metrics.json
├── requirements.txt
├── README.md
└── CLAUDE.md             # ← bạn đang đọc
```

## 3. Pipeline tổng thể (luồng chuẩn)

1. **Crawl dữ liệu** → `crawler/crawl_images.py` → tự crawl từ Google + Bing + Baidu (theo yêu cầu đề bài)
   - 41 keyword đa dạng (Anh + Việt) cho fresh/rotten
   - Lưu `<class>/<keyword_slug>_<idx>.jpg` để biết ảnh thuộc keyword nào
   - **Tham khảo (KHÔNG dùng cho đồ án)**: `tools/prepare_freshness44.py` — chỉ giữ làm backup nếu crawl gặp vấn đề
2. **Clean & split** → `preprocessing/preprocess.py` → tạo `dataset/{train,valid,test}/`
   - Tỉ lệ: 70% / 15% / 15%
   - Resize ảnh về **224x224**, normalize [0,1], loại trùng (perceptual hash)
3. **Augmentation** → `preprocessing/augmentation.py` (rotation, flip, zoom, brightness, shift)
4. **Train** → `models/train.py --model {mobilenet|resnet}`
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
   - Optimizer: Adam (lr=1e-3 → 1e-5 fine-tune)
   - Loss: categorical_crossentropy
   - Metrics: accuracy, precision, recall
5. **Evaluate** → `evaluation/evaluate.py --model <path>`
6. **Compare** → `evaluation/plots.py` so sánh MobileNetV2 vs ResNet50
7. **Predict** → `app/predict.py --image <path> --model <path>`

## 3b. Nguồn dữ liệu

- **Chính (theo đề bài)**: tự crawl từ Google + Bing + Baidu Images qua `icrawler`
  - File: `crawler/crawl_images.py` — chứa danh sách 41 keyword đa dạng
  - Lệnh: `python crawler/crawl_images.py --target 10000 --engines google,bing,baidu`
- **Backup (chỉ tham khảo)**: Freshness44 (Kaggle, 53k ảnh, đã clean) — `tools/prepare_freshness44.py`

## 4. Quy ước

| Mục | Quy ước |
|---|---|
| Image size | 224 × 224 × 3 |
| Batch size | 32 (giảm còn 16 nếu OOM) |
| Class index | `0 = fresh`, `1 = rotten` (theo thứ tự alphabet của ImageDataGenerator) |
| Format model | `.keras` (Keras 3) hoặc `.h5` |
| Random seed | 42 |
| Encoding | UTF-8 cho mọi file Python/Markdown |

## 5. Trạng thái hiện tại (cập nhật mỗi lần làm việc)

- [x] **2026-05-18** — Khởi tạo cấu trúc + bộ test pytest (37 tests pass).
- [x] **2026-05-19** — Thử Freshness44 (Kaggle, 53k ảnh) làm nguồn — sau đó **bỏ**, chuyển về tự crawl theo yêu cầu đề bài. Giữ `tools/prepare_freshness44.py` chỉ làm tham khảo backup.
- [x] **2026-05-20** — Refactor notebooks: gộp về 2 file đầy đủ.
  - `notebook/01_prepare.ipynb` — TỰ CRAWL + clean + split + visualize.
  - `notebook/02_train_and_evaluate.ipynb` — train + evaluate + visualize.
  - Đã xoá `experiment.ipynb` và `colab_train.ipynb` cũ.
- [x] **2026-05-21** — Mở rộng `crawler/crawl_images.py` lên 41 keyword đa dạng (Anh + Việt), đặt tên file `<class>/<slug>_<idx>.jpg` để thống kê.
- [ ] Bước kế tiếp: chạy crawl thực + train.

## 5a. Workflow chuẩn (2 notebook)

Toàn bộ dự án **CHỈ** dùng 2 notebook:

1. **`notebook/01_prepare.ipynb`** — Chuẩn bị (TỰ CRAWL theo đề bài)
   - Setup môi trường + cài deps
   - Crawl 10k ảnh từ Google + Bing + Baidu (`crawler/crawl_images.py --target 10000`)
   - Sanity check: file lỗi, kích thước, đa dạng keyword
   - Tiền xử lý: resize 224×224, loại trùng (perceptual hash), split 70/15/15
   - Trực quan hoá: ảnh mẫu, phân bố class, augmentation demo

2. **`notebook/02_train_and_evaluate.ipynb`** — Thực thi & kết quả
   - Train MobileNetV2 (2-phase)
   - Train ResNet50 (2-phase)
   - Đánh giá test set: accuracy/precision/recall/F1
   - 7 nhóm biểu đồ: training curves, confusion matrices, classification report heatmap, ROC curve, comparison bar chart, misclassified images, predict demo
   - Sinh đầy đủ output `results/*.png` + `results/*_metrics.json` cho báo cáo

## 5b. Test (pytest)

```bash
# Cài thêm pytest nếu chưa có
pip install pytest

# Chạy toàn bộ
pytest -v

# Chạy theo nhóm
pytest tests/test_models.py
pytest tests/test_preprocess.py -v

# Bỏ qua test cần Internet
pytest -m "not network"
```

Cấu trúc tests:
- `tests/conftest.py` — fixtures: `raw_dataset`, `split_dataset_dir`, `tiny_image`, `class_names` (synthetic ảnh, không cần Internet).
- `tests/test_preprocess.py` — process_image / preprocess_all / split_dataset / plots.
- `tests/test_augmentation.py` — train/eval generator: shape, normalize [0,1], one-hot.
- `tests/test_models.py` — MobileNetV2, ResNet50: shape, softmax, freeze/unfreeze, 1-step train smoke.
- `tests/test_evaluation.py` — confusion matrix, plots.py, evaluate.py end-to-end với model giả.
- `tests/test_predict.py` — load_and_preprocess, collect_images, predict_one (dùng `_DummyModel`).
- `tests/test_crawler.py` — clean_directory (broken/small/duplicate), collect_stats, plot_distribution.

## 6. Lệnh hay dùng (cheatsheet)

```bash
# Cài đặt
pip install -r requirements.txt

# Crawl ảnh (mặc định 10000 ảnh tổng)
python crawler/crawl_images.py --target 10000

# Tiền xử lý + split
python preprocessing/preprocess.py --src dataset/raw --dst dataset

# Huấn luyện
python models/train.py --model mobilenet --epochs 30
python models/train.py --model resnet    --epochs 30

# Đánh giá
python evaluation/evaluate.py --model checkpoints/mobilenet_best.keras
python evaluation/evaluate.py --model checkpoints/resnet_best.keras

# So sánh 2 model
python evaluation/plots.py

# Dự đoán
python app/predict.py --image path/to/img.jpg --model checkpoints/mobilenet_best.keras
```

## 7. Quyết định kiến trúc (ADR ngắn)

- **Tại sao MobileNetV2 + ResNet50?** MobileNet nhẹ, train nhanh, hợp laptop yếu; ResNet50 mạnh hơn để so sánh.
- **Tại sao Transfer Learning?** Dataset ~10k ảnh là vừa phải, train from scratch sẽ overfit.
- **Tại sao 2 pha (freeze → fine-tune)?** Pha 1 train classifier head, pha 2 unfreeze top layers + lr nhỏ để tinh chỉnh.
- **Tại sao binary nhưng dùng `categorical_crossentropy`?** Theo yêu cầu đề bài, dùng softmax 2 lớp + one-hot để dễ mở rộng đa lớp sau này.

## 8. Lưu ý quan trọng

- KHÔNG commit thư mục `dataset/`, `checkpoints/`, `logs/` lên git (xem `.gitignore`).
- Khi chạy lần đầu, nếu thiếu Chrome driver thì icrawler sẽ tự fallback — KHÔNG cần selenium.
- Nếu OOM: giảm `--batch-size 16`, tắt augmentation nặng.
- Test các thay đổi với `--epochs 2` trước khi chạy full.

## 9. Chạy trên Google Colab

Notebook `notebook/colab_train.ipynb` đã sẵn sàng end-to-end. Tóm tắt:

1. **Mở Colab** → upload notebook `notebook/colab_train.ipynb` (hoặc `File → Open notebook → GitHub`).
2. **Bật GPU**: `Runtime → Change runtime type → GPU (T4)`.
3. **Chạy lần lượt** các cell — notebook tự:
   - Mount Drive vào `/content/drive/MyDrive/veggie_project/`
   - Symlink `dataset/`, `checkpoints/`, `logs/`, `results/` sang Drive (giữ data qua các phiên)
   - Cài chỉ những lib Colab thiếu (`icrawler`, `imagehash`, `tqdm`)
   - Train với `--mixed-precision --xla` (~1.5–2× nhanh hơn)
   - Hiển thị TensorBoard inline
4. **Crawl 1 lần duy nhất** (~30–60 phút), giữ `dataset/raw/` trong Drive cho các lần sau.

Cờ tối ưu Colab cho `models/train.py`:
- `--mixed-precision` → FP16 (T4/V100/A100)
- `--xla` → XLA JIT compile
- `--batch-size 64` cho MobileNet (T4), `--batch-size 32` cho ResNet

Mẹo:
- Mất kết nối khi disconnect → mọi thứ vẫn còn trong Drive, chỉ cần re-mount + chạy lại train từ checkpoint.
- OOM → giảm `--batch-size` hoặc bỏ `--mixed-precision`.
