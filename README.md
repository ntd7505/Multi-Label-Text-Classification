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

- Python 3.10+


```bash
pip install torch gensim underthesea scikit-learn matplotlib pandas
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

> Sửa `DATA_DIR` và `OUTPUT_DIR` trong Cell 1 của mỗi notebook cho phù hợp với máy.

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
│   ├── raw_data.json               # dữ liệu crawl thô
│   └── process_data/
│       ├── dataset.json            # tokens + multi-hot vectors
│       ├── vocab.json              # token → index
│       └── label_map.json          # nhãn → index
├── dictionary/
│   ├── vietnamese-stopwords.txt
│   └── vietnamese-stopwords-dash.txt
├── notebooks/
│   ├── preprocessing_data.ipynb
│   ├── train_w2v_clean.ipynb
│   ├── evaluation.ipynb
│   └── predict.ipynb
└── output/
    ├── models/
    │   ├── checkpoints/            # best_model.pt
    │   └── word2vec.model
    ├── results/                    # results_w2v_global.json
    └── figures/                    # biểu đồ đánh giá
```

---

## Tham khảo

Van Lam et al. *"Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language"*. NLPIR 2023. [DOI: 10.1145/3639233.3639244](https://dl.acm.org/doi/10.1145/3639233.3639244)
