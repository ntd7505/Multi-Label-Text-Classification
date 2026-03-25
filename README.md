# HARNN — Phân loại văn bản tiếng Việt theo phân cấp nhãn

Project huấn luyện và suy luận mô hình HARNN cho bài toán phân loại bài viết tiếng Việt theo 3 mức nhãn:

```
L1 (domain) → L2 (sub-domain) → L3 (fine-grained)
```

Hiện tại pipeline trong project dùng dữ liệu VnExpress đã xử lý trong `data/process_data/` và workflow chính bằng notebook.

---

## 1) Yêu cầu môi trường

- Python 3.10+
- Khuyến nghị tạo môi trường ảo trước khi cài

```bash
pip install -r requirements.txt
```

---

## 2) Cấu trúc project

```
NLP_Project/
├── craw/
│   └── crawl_data.py
├── data/
│   ├── dictionary/
│   │   ├── vietnamese-stopwords.txt
│   │   └── vietnamese-stopwords-dash.txt
│   ├── process_data/
│   │   ├── dataset.json
│   │   ├── vocab.json
│   │   └── label_map.json
│   ├── raw/
│   ├── train_data.json          # sinh ra từ notebook train
│   └── test_data.json           # sinh ra từ notebook train
├── notebooks/
│   ├── preprocessing_data.ipynb
│   ├── train_w2v_clean.ipynb
│   ├── evaluation.ipynb
│   └── predict.ipynb
├── output/
│   ├── models/
│   │   ├── checkpoints/
│   │   └── word2vec.model
│   ├── results/
│   ├── figures/
│   └── log/
├── requirements.txt
└── README.md
```

---

## 3) Quy trình chạy chuẩn

### Bước 1 — Tiền xử lý dữ liệu

Chạy notebook:

`notebooks/preprocessing_data.ipynb`

Kết quả đầu ra chính:

- `data/process_data/dataset.json`
- `data/process_data/vocab.json`
- `data/process_data/label_map.json`

### Bước 2 — Train mô hình

Chạy notebook:

`notebooks/train_w2v_clean.ipynb`

Notebook này sẽ:

- Chia train/val/test theo iterative stratification
- Lưu lại split:
  - `data/train_data.json`
  - `data/test_data.json`
- Train Word2Vec + HARNN
- Lưu checkpoint tốt nhất vào:
  - `output/models/checkpoints/best_model.pt`

### Bước 3 — Đánh giá mô hình

Chạy notebook:

`notebooks/evaluation.ipynb`

Notebook đánh giá hiện tại sử dụng trực tiếp `data/test_data.json`.

Kết quả và biểu đồ được lưu ở:

- `output/results/`
- `output/figures/`

### Bước 4 — Dự đoán dữ liệu mới

Chạy notebook:

`notebooks/predict.ipynb`

Notebook đã được tối giản còn các cell quan trọng để:

- Load checkpoint + artifacts
- Chạy hàm predict
- Dự đoán trực tiếp từ text nhập tay

---

## 4) Ghi chú quan trọng

- Một số notebook đang dùng đường dẫn tuyệt đối Windows (`C:\Users\Admin\...`).
- Nếu chạy trên máy khác, cần sửa lại các biến path trong Cell 1 của notebook tương ứng.
- Các thư mục output và dữ liệu sinh ra đã được cấu hình để bỏ qua trong `.gitignore`.

---

## 5) Tham khảo

Van Lam et al. *"Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language"*. NLPIR 2023.

DOI: https://dl.acm.org/doi/10.1145/3639233.3639244
