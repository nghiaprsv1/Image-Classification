# Questions.md — Bảng vấn đáp đồ án

> File ghi lại các câu hỏi vấn đáp / kiểm tra của thầy giáo qua từng giai đoạn,
> kèm câu trả lời chi tiết. Cập nhật mỗi lần có câu hỏi mới.

---

## GIAI ĐOẠN 1 — Thu thập dữ liệu

### Q1. Những phương pháp nào được sử dụng để đạt được số ảnh > 12k như hiện tại?

**Trả lời (đã đạt 12.786 ảnh / 16 class)**

#### 1. Multi-engine crawling — 4 search engine song song

Sử dụng đồng thời **4 nguồn search engine khác nhau** thay vì chỉ 1 nguồn để tăng diversity và tránh bị block:

| Engine | Thư viện | Vai trò |
|---|---|---|
| **Bing Images** | `icrawler.builtin.BingImageCrawler` | Index rộng, ổn định, ít bị block |
| **Baidu Images** | `icrawler.builtin.BaiduImageCrawler` | Coverage tốt cho ảnh châu Á |
| **Google Images** | `icrawler.builtin.GoogleImageCrawler` | Quality cao, fallback |
| **DuckDuckGo** | `crawler/ddg_crawler.py` (tự viết) | Privacy-friendly, không yêu cầu API key |

DuckDuckGo crawler **tự viết** vì `icrawler` không hỗ trợ native — wrapper dùng thư viện `ddgs` để fetch image URLs, sau đó download song song 8 thread bằng `requests`.

**Tại sao multi-engine?** Mỗi engine có thuật toán index khác nhau → kết quả crawl trùng nhau ít hơn 30% → tăng diversity dataset, model tránh overfit theo style ảnh của 1 nguồn (ví dụ Google có nhiều watermark).

#### 2. Keyword strategy — 41+ keyword đa dạng / class

Mỗi class (fresh / rotten) có **41+ keyword bằng 3 ngôn ngữ**:

- **Tiếng Anh chung**: `"fresh apple fruit"`, `"rotten banana brown"`, `"moldy orange decay"`
- **Tiếng Anh chi tiết**: `"close-up rotten guava"`, `"decaying fruit garbage"`, `"fungus growing on bread"`
- **Tiếng Việt**: `"rau cu tuoi"`, `"trai cay hu thoi"`, `"ca chua tuoi"`, `"qua hong moc meo"`

Cụ thể trong `crawler/crawl_images.py::KEYWORDS`:
- Class `fresh`: 47 keyword (trái cây 20 + rau củ 21 + tiếng Việt 6)
- Class `rotten`: 41 keyword (trái cây 19 + rau hỏng 16 + tiếng Việt 6)

→ Mỗi keyword × 4 engine × ~per_keyword/4 ≈ 50-150 ảnh/keyword × 41 keyword/class ≈ **2-6k ảnh/class** trước khi clean.

#### 3. Naming convention để traceback

File được lưu thành `<class>/<keyword_slug>_<idx>.jpg` thay vì tên random:

```
Apple_Fresh/
├── fresh_apple_fruit_00001.jpg     ← từ keyword "fresh apple fruit"
├── fresh_apple_fruit_00002.jpg
├── tao_tuoi_do_00001.jpg            ← từ keyword "tao tuoi do" (Việt)
└── ...
```

→ Nhờ đó có thể thống kê **keyword nào hiệu quả nhất**, keyword nào trùng nhiều bị loại — tinh chỉnh cho lần crawl sau.

#### 4. Data cleaning trong quá trình crawl (`clean_directory`)

Mỗi class sau khi crawl được lọc qua **3 filter**:

| Filter | Phương pháp | Mục đích |
|---|---|---|
| **Broken image** | `PIL.Image.verify()` | Loại file lỗi không decode được |
| **Tiny image** | `min(W, H) < 64 px` | Loại icon, thumbnail (không đủ thông tin học) |
| **Duplicate** | `imagehash.phash()` 64-bit (8×8 DCT) | Loại ảnh trùng (kể cả resize/recompress) |

→ Tỷ lệ giữ lại sau cleaning thường là **40-60%** so với raw crawl.

#### 5. Perceptual Hash (pHash) thay vì byte hash

Khi 4 engine crawl cùng keyword → nhiều ảnh trùng:
- **MD5/SHA256**: 1 pixel khác cũng không match → không phát hiện ảnh bị resize/recompress
- **Filename match**: vô dụng vì engine đặt tên ngẫu nhiên
- **Perceptual hash (pHash)**: resize ảnh về 32×32 grayscale, áp DCT, lấy 8×8 = 64 bit từ low-frequency coefficients → **2 ảnh giống nội dung sẽ có hash giống nhau** kể cả bị resize / recompress / watermark nhẹ

Hamming distance giữa 2 phash < 5 → coi là duplicate. Đây là chuẩn industry cho deduplication ảnh.

#### 6. Early-stopping per class

Trong `crawl_class()`, mỗi keyword crawl xong sẽ check tổng số ảnh hiện có:

```python
if current >= target_per_class:
    print(f"[done-early] {class_name} đạt {current}/{target_per_class}")
    break
```

→ Tránh crawl thừa khi đã đủ target → tiết kiệm thời gian + bandwidth.

#### 7. Topup từ Kaggle (Freshness44) — backup khi crawl thiếu

Vì crawl 4 engine vẫn có thể không đủ ảnh cho 1 số class hiếm (ví dụ rotten guava, rotten pomegranate), project chuẩn bị **`tools/prepare_freshness44.py`** đọc từ Kaggle Freshness44 dataset (53k ảnh đã clean) để topup các class thiếu.

**Lưu ý**: theo yêu cầu đề bài "TỰ crawl, KHÔNG dùng nguyên dataset có sẵn", topup này chỉ dùng khi **hoàn toàn cần thiết** và phải khai báo rõ trong báo cáo. Số ảnh chính (>10k) phải đến từ tự crawl.

#### 8. Threading + retries

Mỗi crawler instance dùng:
- `feeder_threads=1` (parse query)
- `parser_threads=2` (extract image URLs)
- `downloader_threads=4` (download song song)

→ **8 connection song song** cho mỗi engine → crawl 1 keyword 50 ảnh chỉ mất ~15-30 giây.

DuckDuckGo crawler dùng `ThreadPoolExecutor` 8 workers + retry 2 lần với timeout 8s → ổn định khi mạng chập chờn.

---

#### Tổng hợp con số thực tế

```
Tổng số ảnh thu thập:      12.786 ảnh
Số class:                  16 (8 loại quả × Fresh/Rotten)
Trung bình mỗi class:      ~800 ảnh
Imbalance ratio:            1.13 (max 870 / min 766) — rất cân bằng
Số nguồn crawl:             4 search engine (Bing + Baidu + Google + DuckDuckGo)
Số keyword đa dạng:         41+ / class (3 ngôn ngữ)
Định dạng:                 JPG, PNG, WebP (sau cleaning unify về JPG ở GĐ 2)
Kích thước:                 đa dạng từ 100×100 đến 2000×2000 px
```

#### Bằng chứng (file output)

- `dataset/raw/` — folder chứa 16 class với 12.786 ảnh đã clean
- `results/eda_metadata.csv` — metadata chi tiết từng ảnh (path, class, format, w, h, size_kb)
- `results/eda_dataset_summary.csv` — bảng tổng hợp 16 class × 4 cột (Fresh, Rotten, Total, Train/Val/Test plan)
- `results/eda_balance_*.png` — 4 biểu đồ trực quan phân bố

---

<!-- ===== TEMPLATE CHO CÂU HỎI TIẾP THEO ===== -->

<!--
### Q2. [Câu hỏi tiếp theo của thầy giáo]

**Trả lời**

[Nội dung trả lời]
-->
