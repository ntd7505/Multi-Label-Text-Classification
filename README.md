# HARNN — Phân loại văn bản đa nhãn phân cấp tiếng Việt

Dự án này triển khai toàn bộ pipeline huấn luyện và suy luận (inference) cho mô hình **HARNN (Hierarchical Attention-based Recurrent Neural Network)** nhằm giải quyết bài toán phân loại văn bản tiếng Việt.

Hệ thống được thiết kế để phân loại các bài báo (dữ liệu từ VnExpress) theo cấu trúc cây phân cấp 3 mức độ (Hierarchical Multi-label Text Classification): 
👉 **Level 1 (Domain)** → **Level 2 (Sub-domain)** → **Level 3 (Fine-grained)**

## 🌟 Điểm nổi bật của dự án

* **Tiền xử lý chuyên biệt:** Sử dụng thư viện `underthesea` để tách từ tiếng Việt, kết hợp lọc stopwords nhằm bảo toàn ngữ nghĩa của các từ ghép đa âm tiết.
* **Word Embedding:** Tự huấn luyện ma trận Word2Vec (thuật toán Skip-gram, 100 chiều) để biểu diễn tốt các từ hiếm.
* **Kiến trúc HARNN:** Kết hợp mạng **BiGRU** để trích xuất ngữ cảnh toàn cục, cơ chế **Hierarchical Attention Layer (HARL)** riêng biệt cho từng tầng nhãn và khối bộ nhớ **HAM (LSTMCell)** để truyền tải thông tin logic từ L1 xuống L3.
* **Xử lý mất cân bằng dữ liệu (Long-tail):** Tối ưu hóa bằng hàm `BCEWithLogitsLoss` kết hợp kỹ thuật Trọng số cấp độ (Level Weights), phạt nặng hơn đối với các lỗi dự đoán ở nhãn thiểu số (L3).

---

## ⚙️ 1. Yêu cầu hệ thống và Cài đặt

* **Python:** Phiên bản 3.10 trở lên.
* Khuyến nghị tạo môi trường ảo (Virtual Environment) trước khi cài đặt các thư viện phụ thuộc để tránh xung đột.

```bash
# Clone repository
git clone <your-repo-url>
cd NLP_Project

# Cài đặt thư viện
pip install -r requirements.txt
📂 2. Cấu trúc thư mục (Project Structure)
Plaintext
NLP_Project/
├── craw/
│   └── crawl_data.py
├── data/
│   ├── dictionary/
│   │   ├── vietnamese-stopwords.txt
│   │   └── vietnamese-stopwords-dash.txt
│   ├── process_data/
│   │   ├── dataset.json             # Dữ liệu sau tiền xử lý
│   │   ├── vocab.json               # Từ điển xây dựng từ corpus
│   │   └── label_map.json           # Mapping nhãn L1, L2, L3
│   ├── raw/                         # Chứa dữ liệu thô crawl về
│   ├── train_data.json              # Sinh ra từ notebook train (Iterative Stratification)
│   └── test_data.json               # Sinh ra từ notebook train
├── notebooks/
│   ├── preprocessing_data.ipynb     # Pipeline làm sạch và chuẩn bị dữ liệu
│   ├── train_w2v_clean.ipynb        # Huấn luyện Word2Vec & HARNN
│   ├── evaluation.ipynb             # Đánh giá chi tiết (Confusion Matrix, Metrics)
│   └── predict.ipynb                # Inference trên văn bản nhập tay
├── output/
│   ├── models/
│   │   ├── checkpoints/             # Lưu trữ best_model.pt
│   │   └── word2vec.model           # Model nhúng từ
│   ├── results/
│   ├── figures/                     # Lưu các biểu đồ phân tích
│   └── log/
├── requirements.txt
└── README.md
🚀 3. Quy trình chạy chuẩn (Workflow)
Pipeline của dự án được đóng gói trực quan qua các file Jupyter Notebook. Vui lòng chạy theo thứ tự sau:

Bước 1: Tiền xử lý dữ liệu
File thực thi: notebooks/preprocessing_data.ipynb

Mục đích: Làm sạch văn bản, tách từ, loại bỏ stopwords và định dạng lại dữ liệu.

Đầu ra: Các file dataset.json, vocab.json, label_map.json nằm trong thư mục data/process_data/.

Bước 2: Huấn luyện mô hình (Train)
File thực thi: notebooks/train_w2v_clean.ipynb

Mục đích: * Chia tập dữ liệu train/val/test theo phương pháp Iterative Stratification nhằm bảo toàn phân bố của các nhãn hiếm.

Lưu lại bản split ra file data/train_data.json và data/test_data.json.

Huấn luyện thuật toán Word2Vec và mô hình HARNN.

Đầu ra: Checkpoint có F1-score tốt nhất sẽ được lưu tại output/models/checkpoints/best_model.pt.

Bước 3: Đánh giá mô hình (Evaluate)
File thực thi: notebooks/evaluation.ipynb

Mục đích: Sử dụng trực tiếp data/test_data.json để đánh giá độc lập. Tính toán các độ đo Micro/Macro F1, AUPRC và vẽ Ma trận nhầm lẫn (Confusion Matrix).

Đầu ra: Biểu đồ và báo cáo chỉ số được lưu tự động ở output/results/ và output/figures/.

Bước 4: Suy luận và Dự đoán (Predict)
File thực thi: notebooks/predict.ipynb

Mục đích: Notebook đã được tối giản hóa. Bạn chỉ cần chạy các cell để load file checkpoint cùng các artifacts, sau đó sử dụng hàm predict để dự đoán tự động phân cấp nhãn cho bất kỳ đoạn văn bản tiếng Việt nào được nhập tay.

⚠️ 4. Ghi chú quan trọng
Vấn đề đường dẫn (Path Variables): Một số notebook hiện tại đang sử dụng đường dẫn tuyệt đối của môi trường Windows (VD: C:\Users\Admin\...). Khi clone về và chạy trên máy khác (hoặc Linux/macOS), bạn cần sửa lại các biến path trong Cell đầu tiên của mỗi notebook tương ứng thành đường dẫn tương đối hoặc khớp với thư mục máy của bạn.

Git Ignore: Các thư mục chứa dữ liệu thô, output logs, models lớn và figures đã được cấu hình loại bỏ trong .gitignore để tránh phình to kích thước repository. Bạn cần tự tạo lại cấu trúc folder rỗng (nếu thiếu) khi mới clone về.

👥 5. Nhóm tác giả
Trịnh Đăng Huy

Vũ Hải Đăng

Nguyễn Thành Đạt

(Trường Đại học Xây dựng Hà Nội - Lớp học phần: 68CS2)

📚 6. Nguồn tham khảo
Dự án này được lấy cảm hứng và xây dựng dựa trên nền tảng lý thuyết từ nghiên cứu:

Van Lam et al. "Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language". NLPIR 2023. DOI: https://dl.acm.org/doi/10.1145/3639233.3639244
