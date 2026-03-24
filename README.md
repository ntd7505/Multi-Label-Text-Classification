# HARNN — Phân loại văn bản tiếng Việt đa nhãn phân cấp

Tự động phân loại bài viết VnExpress vào hệ thống nhãn 3 tầng dựa trên kiến trúc HARNN (Hierarchical Attention-based Recurrent Neural Network).

```
Thể thao  →  Bóng đá  →  Champions League
   L1           L2              L3
  (13 nhãn)  (40 nhãn)      (43 nhãn)
```

**Kết quả trên test set:**

| Level | F1 | AUPRC |
|-------|----|-------|
| L1 | 0.902 | 0.961 |
| L2 | 0.767 | 0.898 |
| L3 | 0.660 | 0.775 |

---

## Yêu cầu

- Python 3.10
- GPU với CUDA 11.8 (khuyến nghị)

### Cài đặt

```bash
# Tạo virtual environment
py -3.10 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Cài PyTorch CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài các thư viện còn lại
pip install -r requirements.txt
```

---

## Hướng dẫn chạy

Chạy theo đúng thứ tự:

**Bước 1 — Tiền xử lý**
```
notebooks/preprocessing_data.ipynb
```
Đọc `raw_data.json`, chuẩn hóa nhãn, tokenize, xuất ra `dataset.json` · `vocab.json` · `label_map.json`.

**Bước 2 — Train**
```
notebooks/train_w2v_clean.ipynb
```
Train Word2Vec trên corpus, train HARNN model, lưu checkpoint tốt nhất vào `output/models/checkpoints/best_model.pt`.

**Bước 3 — Đánh giá** *(tùy chọn)*
```
notebooks/evaluation.ipynb
```
Vẽ confusion matrix, F1 theo từng nhãn, precision-recall curve.

**Bước 4 — Dự đoán**
```
notebooks/predict.ipynb
```
Load checkpoint, dự đoán nhãn cho bài viết mới.

> Các notebook tự động xác định đường dẫn project (sử dụng `Path.cwd().parent`). Chỉ cần chạy notebook từ đúng thư mục chứa nó.

---

## Dataset

| | |
|--|--|
| Nguồn | VnExpress — crawl bằng Selenium |
| Kích thước | 5541 bài · 13 domain |
| Chia | Train 80% · Val 10% · Test 10% |

---

## Cấu trúc thư mục

```
├── data/
│   ├── raw/
│   │   └── raw_data.json              # dữ liệu crawl thô
│   ├── process_data/
│   │   ├── dataset.json               # tokens + multi-hot vectors
│   │   ├── vocab.json                 # token → index
│   │   └── label_map.json             # nhãn → index
│   ├── dictionary/
│   │   ├── vietnamese-stopwords.txt
│   │   └── vietnamese-stopwords-dash.txt
│   ├── label_cleaning.ipynb           # chuẩn hóa nhãn
│   └── label_stats.ipynb              # thống kê phân phối nhãn
├── notebooks/
│   ├── preprocessing_data.ipynb
│   ├── train_w2v_clean.ipynb
│   ├── evaluation.ipynb
│   └── predict.ipynb
└── output/
    ├── models/
    │   ├── checkpoints/               # best_model.pt
    │   └── word2vec.model
    ├── results/                       # results_w2v_global.json
    └── figures/                       # biểu đồ đánh giá
```

---

## Tham khảo

Van Lam et al. *"Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language"*. NLPIR 2023. [DOI: 10.1145/3639233.3639244](https://dl.acm.org/doi/10.1145/3639233.3639244)
