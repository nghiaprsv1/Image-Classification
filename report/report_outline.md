# BÁO CÁO ĐỀ TÀI
## Xây dựng hệ thống phân loại rau sạch và rau hỏng sử dụng Deep Learning

> **Ước lượng**: ~35–45 trang A4, font Times New Roman 13pt, line spacing 1.5.

---

## TRANG BÌA & MỤC LỤC (≈ 4 trang)

- Trang bìa chính (tên đề tài, GVHD, sinh viên thực hiện, năm học)
- Trang bìa phụ
- Lời cảm ơn
- Lời cam đoan
- Mục lục
- Danh mục hình ảnh
- Danh mục bảng biểu
- Danh mục từ viết tắt (CNN, ReLU, ResNet, MobileNet, GAP, BN, ImageNet, ...)

---

## CHƯƠNG 1 — GIỚI THIỆU (≈ 4 trang)

### 1.1 Đặt vấn đề
- Thực trạng tiêu thụ rau ở Việt Nam, nguy cơ rau hỏng ảnh hưởng sức khoẻ.
- Việc phân loại thủ công tốn công, dễ sai.
- Sự bùng nổ của Deep Learning trong Computer Vision (AlexNet 2012 → nay).

### 1.2 Mục tiêu đề tài
- Xây dựng hệ thống AI phân loại ảnh rau thành **fresh** / **rotten**.
- So sánh ít nhất **2 mô hình** Deep Learning (MobileNetV2, ResNet50).
- Đạt accuracy ≥ 90% trên tập test.

### 1.3 Đối tượng & phạm vi
- Đối tượng: ảnh các loại rau phổ biến (cà chua, bắp cải, cà rốt, dưa leo, ớt, xà lách, súp lơ).
- Phạm vi: chỉ phân loại 2 lớp (fresh / rotten), ảnh tĩnh, RGB.

### 1.4 Phương pháp nghiên cứu
- Thu thập dữ liệu thực tế (web crawl).
- Áp dụng Transfer Learning + Fine-tuning.
- Đánh giá định lượng (accuracy, precision, recall, F1, confusion matrix).

### 1.5 Bố cục báo cáo
- Tóm tắt nội dung 7 chương.

**Hình ảnh cần chèn**: 1.1 Pipeline tổng thể của hệ thống.

---

## CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT (≈ 7 trang)

### 2.1 Tổng quan về Machine Learning & Deep Learning
- Định nghĩa, lịch sử phát triển.
- Phân biệt ML cổ điển vs Deep Learning.
- **Hình 2.1**: So sánh hiệu năng ML vs DL khi tăng data.

### 2.2 Convolutional Neural Network (CNN)
- Kiến trúc tổng quát: Conv → ReLU → Pooling → FC.
- Ý nghĩa từng thành phần:
  - Convolution layer (kernel, stride, padding).
  - Activation (ReLU, LeakyReLU, Softmax).
  - Pooling (Max, Average, Global Average).
  - Batch Normalization.
  - Dropout.
- **Hình 2.2**: Sơ đồ CNN cơ bản.
- **Hình 2.3**: Minh hoạ phép tích chập (kernel 3×3).
- **Bảng 2.1**: Công thức tính kích thước feature map.

### 2.3 Transfer Learning
- Khái niệm, lợi ích (giảm dữ liệu, giảm thời gian train).
- 2 chiến lược: feature extraction vs fine-tuning.
- **Hình 2.4**: Sơ đồ transfer learning.

### 2.4 MobileNetV2
- Inverted Residual + Linear Bottleneck.
- Depthwise Separable Convolution → giảm FLOPs.
- **Bảng 2.2**: Số tham số ~ 2.3M.
- **Hình 2.5**: Khối inverted residual.

### 2.5 ResNet50
- Skip connection / residual block.
- Giải quyết vanishing gradient.
- **Hình 2.6**: Residual block.
- **Bảng 2.3**: Cấu hình ResNet50 (50 layer, ~25.6M params).

### 2.6 Hàm mất mát & Optimizer
- Categorical Cross-Entropy (công thức + ý nghĩa).
- Adam optimizer (kết hợp Momentum + RMSProp).
- **Bảng 2.4**: So sánh SGD / Adam / RMSProp.

### 2.7 Đánh giá mô hình phân loại
- Accuracy, Precision, Recall, F1-score.
- Confusion Matrix.
- ROC-AUC (giới thiệu).

---

## CHƯƠNG 3 — THU THẬP DỮ LIỆU (≈ 4 trang)

### 3.1 Nguồn dữ liệu
- Google Images, Bing Images, Baidu (qua thư viện `icrawler`).
- Lý do chọn web crawl: dataset miễn phí, đa dạng.

### 3.2 Quy trình thu thập
- Danh sách keyword (fresh/rotten + tên rau).
- Số ảnh mục tiêu: ≥ 10.000 ảnh.
- **Hình 3.1**: Quy trình crawl.

### 3.3 Làm sạch dữ liệu
- Loại ảnh hỏng (file lỗi, không decode được).
- Loại ảnh quá nhỏ (< 64×64).
- Loại ảnh trùng bằng **perceptual hash (pHash)**.
- **Bảng 3.1**: Số ảnh trước/sau làm sạch.

### 3.4 Thống kê dataset cuối cùng
- Số ảnh fresh / rotten.
- **Hình 3.2**: Biểu đồ phân bố class.
- **Hình 3.3**: Top kích thước ảnh phổ biến.
- **Bảng 3.2**: Bảng kê chi tiết.

---

## CHƯƠNG 4 — TIỀN XỬ LÝ DỮ LIỆU (≈ 4 trang)

### 4.1 Resize & Chuẩn hóa
- Resize 224×224, convert RGB.
- Normalize pixel [0, 255] → [0, 1].

### 4.2 Chia tập dữ liệu
- Train 70% / Valid 15% / Test 15% (stratified).
- **Hình 4.1**: Biểu đồ phân bố train/valid/test.

### 4.3 Data Augmentation
- Rotation ±25°, horizontal flip, zoom 0.2, brightness 0.8–1.2, shift ±15%, shear 0.1.
- Lý do: tăng đa dạng, chống overfit.
- **Hình 4.2**: 8 phiên bản augmentation từ 1 ảnh gốc.

### 4.4 Trực quan hóa
- **Hình 4.3**: Sample images mỗi class.
- **Bảng 4.1**: Cấu hình tiền xử lý.

---

## CHƯƠNG 5 — XÂY DỰNG MÔ HÌNH (≈ 6 trang)

### 5.1 Kiến trúc tổng quát
- Backbone (pretrained) + custom head (GAP → Dense → Dropout → Softmax).
- **Hình 5.1**: Sơ đồ tổng quát.

### 5.2 Mô hình 1 — MobileNetV2
- Cấu hình: dense=128, dropout=0.3, L2=1e-4.
- **Hình 5.2**: Model summary.

### 5.3 Mô hình 2 — ResNet50
- Cấu hình: dense=256, dropout=0.4, L2=1e-4 + BatchNorm.
- **Hình 5.3**: Model summary.

### 5.4 Chiến lược huấn luyện 2 pha
- Pha 1: freeze base, train head với lr=1e-3, ~30 epoch.
- Pha 2: unfreeze top 30–40 layers, fine-tune với lr=1e-5, ~15 epoch.
- **Bảng 5.1**: Hyperparameters.

### 5.5 Callbacks
- ModelCheckpoint (best val_accuracy).
- EarlyStopping (patience=7).
- ReduceLROnPlateau (factor=0.5, patience=3).
- TensorBoard, CSVLogger.

### 5.6 Cấu hình môi trường
- Phần cứng: CPU/GPU.
- Phần mềm: TensorFlow 2.15, Python 3.10.

---

## CHƯƠNG 6 — ĐÁNH GIÁ KẾT QUẢ (≈ 7 trang)

### 6.1 Kết quả MobileNetV2
- **Hình 6.1**: Accuracy curve.
- **Hình 6.2**: Loss curve.
- **Hình 6.3**: Confusion matrix.
- **Bảng 6.1**: Classification report.

### 6.2 Kết quả ResNet50
- **Hình 6.4**: Accuracy curve.
- **Hình 6.5**: Loss curve.
- **Hình 6.6**: Confusion matrix.
- **Bảng 6.2**: Classification report.

### 6.3 So sánh 2 mô hình
- **Bảng 6.3**: Accuracy / Precision / Recall / F1 / Params / Train time.
- **Hình 6.7**: Bar chart so sánh metrics.
- **Hình 6.8**: Overlay val_accuracy theo epoch.

### 6.4 Phân tích lỗi
- Một số ảnh bị phân loại sai.
- Nguyên nhân: ánh sáng, ảnh nhiễu, biến thể chưa thấy khi train.
- **Hình 6.9**: Lưới ảnh dự đoán sai.

### 6.5 Demo ứng dụng
- Ảnh chụp thực tế → kết quả dự đoán + confidence.
- **Hình 6.10–6.12**: Ảnh demo predict.

---

## CHƯƠNG 7 — KẾT LUẬN & HƯỚNG PHÁT TRIỂN (≈ 3 trang)

### 7.1 Kết quả đạt được
- Hoàn thành pipeline end-to-end.
- Accuracy đạt mục tiêu (≥ 90%).
- So sánh được 2 kiến trúc.

### 7.2 Hạn chế
- Dataset crawl còn nhiễu.
- Mới phân 2 lớp, chưa nhận diện loại rau cụ thể.
- Chưa triển khai web/mobile.

### 7.3 Hướng phát triển
- Đa lớp (theo loại rau + mức độ hỏng).
- Object detection (YOLO) để phát hiện vùng hỏng.
- Triển khai Streamlit / FastAPI / TFLite cho mobile.
- Tăng dataset bằng GAN hoặc data thực tế.

---

## TÀI LIỆU THAM KHẢO (≈ 1 trang)

1. He K. et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
2. Sandler M. et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
3. Krizhevsky A. et al., *ImageNet Classification with Deep CNN*, NeurIPS 2012.
4. Goodfellow I., Bengio Y., Courville A., *Deep Learning*, MIT Press, 2016.
5. Chollet F., *Deep Learning with Python*, Manning, 2nd ed., 2021.
6. Keras Documentation — https://keras.io
7. TensorFlow Documentation — https://www.tensorflow.org

---

## PHỤ LỤC (≈ 2 trang)

- Phụ lục A: Mã nguồn quan trọng (crawl, train, predict).
- Phụ lục B: Hướng dẫn cài đặt & chạy thử.
- Phụ lục C: Bảng tra Hyperparameter.
