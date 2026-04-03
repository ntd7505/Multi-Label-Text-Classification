# HARNN — Phân loại văn bản đa nhãn phân cấp tiếng Việt

Dự án triển khai pipeline huấn luyện và suy luận (inference) cho mô hình **HARNN (Hierarchical Attention-based Recurrent Neural Network)** nhằm giải bài toán phân loại văn bản tiếng Việt.

Phân loại theo cấu trúc 3 cấp:
**Level 1 (Domain) → Level 2 (Sub-domain) → Level 3 (Fine-grained)**

---

## 🌟 Điểm nổi bật

* Tiền xử lý tiếng Việt với `underthesea` + stopwords
* Word2Vec (Skip-gram, 100d)
* Kiến trúc:

  * BiGRU (context)
  * HARL (attention theo level)
  * HAM - LSTMCell (truyền L1 → L3)
* Xử lý imbalance bằng `BCEWithLogitsLoss` + Level Weights

---

## ⚙️ Cài đặt

```bash
git clone <your-repo-url>
cd NLP_Project
pip install -r requirements.txt
```

---

## 📂 Cấu trúc project

```plaintext
NLP_Project/
├── craw/
├── data/
│   ├── dictionary/
│   ├── process_data/
│   ├── raw/
│   ├── train_data.json
│   └── test_data.json
├── notebooks/
│   ├── preprocessing_data.ipynb
│   ├── train_w2v_clean.ipynb
│   ├── evaluation.ipynb
│   └── predict.ipynb
├── output/
│   ├── models/
│   ├── results/
│   ├── figures/
│   └── log/
├── requirements.txt
└── README.md
```

---

## 🚀 Workflow

### 1. Preprocessing

`notebooks/preprocessing_data.ipynb`
→ Sinh `dataset.json`, `vocab.json`, `label_map.json`

### 2. Training

`notebooks/train_w2v_clean.ipynb`
→ Train Word2Vec + HARNN
→ Output: `best_model.pt`

### 3. Evaluation

`notebooks/evaluation.ipynb`
→ Micro/Macro F1, AUPRC, Confusion Matrix

### 4. Predict

`notebooks/predict.ipynb`
→ Dự đoán văn bản bất kỳ

---

## ⚠️ Lưu ý

* Sửa path (tránh dùng path tuyệt đối Windows)
* `.gitignore` đã loại:

  * data raw
  * models
  * logs
  * figures

---

## 👥 Authors

* Trịnh Đăng Huy
* Vũ Hải Đăng
* Nguyễn Thành Đạt

---

## 📚 Reference

Van Lam et al. (NLPIR 2023)
https://dl.acm.org/doi/10.1145/3639233.3639244
