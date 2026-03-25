"""
label_merge_optimized.py
========================
Logic merge nhãn tối ưu cho dataset VnExpress (HARNN).
Thay thế Cell 2 + Cell 3 trong preprocessing_updated.ipynb.

Các cải tiến so với phiên bản cũ:
  1. Sửa tên nhãn lệch (Giải sao → Giới sao, Nhân khoa → Nhãn khoa, ...)
  2. Cứu toàn bộ 765 bài "Tin tức" L2 bằng TIN_TUC_MAP đầy đủ
  3. Tách rõ Phân tích là L2 độc lập trong Thế giới (không merge vào Tư liệu)
  4. Loại bỏ rule duplicate (Ngoại hạng Anh xuất hiện ở cả L2_TO_L3 lẫn L3_MERGE)
  5. Drop chỉ áp dụng cho nhãn < 5 bài hoặc nhãn format/sự kiện thuần tuý
  6. Thêm rescue map đầy đủ khi L3 bị drop nhưng L2 chưa có
"""

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHY  (index cố định — không thay đổi thứ tự giữa các lần chạy)
# HIERARCHY[l1][l2] = [l3, ...]   |   list rỗng = không có L3
# ─────────────────────────────────────────────────────────────────────────────
HIERARCHY = {
    # ── 1. Thời sự ──────────────────────────────────────────────────────────
    "Thời sự": {
        "Chính trị": [],
        "Dân sinh":  [],
        "Giao thông":[],
        "Diễn đàn":  [],
    },
    # ── 2. Thế giới ─────────────────────────────────────────────────────────
    "Thế giới": {
        "Quân sự":   [],
        "Quốc tế":   [],
        "Tư liệu":   [],
        "Phân tích": [],      # ← tách ra, không merge vào Tư liệu
    },
    # ── 3. Kinh doanh ───────────────────────────────────────────────────────
    "Kinh doanh": {
        "Thị trường":  [],
        "Hàng hoá":    [],    # ← chuẩn: dấu hỏi (hoá)
        "Doanh nghiệp":[],
        "Vĩ mô":       [],
        "Chứng khoán": [],
    },
    # ── 4. Khoa học ─────────────────────────────────────────────────────────
    "Khoa học": {
        "Vũ trụ":            [],
        "Thế giới tự nhiên": [],
    },
    # ── 5. Khoa học công nghệ ───────────────────────────────────────────────
    "Khoa học công nghệ": {
        "AI":            [],
        "Thiết bị":      [],
        "Chuyển đổi số": [],
        "Netzero":       [],
    },
    # ── 6. Giải trí ─────────────────────────────────────────────────────────
    "Giải trí": {
        "Giới sao": ["Sao đẹp - Sao xấu", "Chuyện màn ảnh"],   # ← Giới (không phải Giải)
        "Sân khấu - Mỹ thuật": ["Sân khấu", "Mỹ thuật"],
        "Phim":       [],
        "Sách":       ["Điểm sách"],
        "Thời trang": ["Lăng mốt"],
        "Nhạc":       ["Lăng nhạc"],
    },
    # ── 7. Thể thao ─────────────────────────────────────────────────────────
    "Thể thao": {
        "Bóng đá": [
            "Champions League",
            "Các giải khác",   # ← Ngoại hạng Anh, V-League, La Liga... sẽ merge vào đây
            "Trong nước",
        ],
        "Tennis":       [],
        "Marathon":     [],
        "Các môn khác": [],
    },
    # ── 8. Pháp luật ────────────────────────────────────────────────────────
    "Pháp luật": {
        "Hồ sơ phá án": [],
        "Tư vấn":        [],
    },
    # ── 9. Giáo dục ─────────────────────────────────────────────────────────
    "Giáo dục": {
        "Tuyển sinh": ["Đại học"],
        "Du học":     [],
        "Chân dung":  [],
    },
    # ── 10. Sức khỏe ────────────────────────────────────────────────────────
    "Sức khỏe": {
        "Các bệnh": [
            "Nhãn khoa",          # ← sửa: Nhãn (không phải Nhân)
            "Tim mạch", "Nội tiết", "Thần kinh", "Tiêu hóa",
            "Da liễu", "Hô hấp", "Sản phụ khoa", "Cơ xương khớp",
            "Ung thư", "Tai mũi họng", "Tiết niệu - Nam học",
            "Nhi - Sơ sinh",
        ],
        "Sống khỏe":        ["Dinh dưỡng", "Hỏi đáp"],
        "Vaccine":          ["Vaccine người lớn"],
        "Tin tức sức khỏe": [],
    },
    # ── 11. Đời sống ────────────────────────────────────────────────────────
    "Đời sống": {
        "Nhịp sống":        ["Lăng văn", "Bạn đọc viết", "Nhịp sống số"],
        "Ẩm thực":          [],
        "Tổ ấm":            [],
        "Bài học sống":     [],
        "Cuộc sống đó đây": [],
    },
    # ── 12. Du lịch ─────────────────────────────────────────────────────────
    "Du lịch": {
        "Điểm đến": [],
        "Ẩm thực":  [],
    },
    # ── 13. Xe ──────────────────────────────────────────────────────────────
    "Xe": {
        "Cầm lái":       ["Kỹ năng lái", "Tình huống", "Đánh giá xe"],
        "Thị trường xe": [],
        "Tin tức xe":    [],
    },
}

# Build lookup tables
L1_SET, L2_SET, L3_SET = set(), set(), set()
L2_TO_L1, L3_TO_L2 = {}, {}
for _l1, _subs in HIERARCHY.items():
    L1_SET.add(_l1)
    for _l2, _l3s in _subs.items():
        L2_SET.add(_l2)
        L2_TO_L1[_l2] = _l1
        for _l3 in _l3s:
            L3_SET.add(_l3)
            L3_TO_L2[_l3] = _l2


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZE  — chuẩn hóa tên trước mọi xử lý
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZE = {
    # L1
    "Số hóa":                    "Khoa học công nghệ",
    # L2 — ẩm thực
    "cooking":                   "Ẩm thực",
    "Cooking":                   "Ẩm thực",
    # L2 — chính trị
    "Chính trị & chính sách":    "Chính trị",
    "Chính sách":                "Chính trị",
    # L2 — chuẩn hóa dấu
    "Hàng hóa":                  "Hàng hoá",
    "tennis":                    "Tennis",
    # L2 — Giới sao (data dùng "Giới", code cũ dùng "Giải")
    "Giải sao":                  "Giới sao",
    # L3 typo
    "Nhân khoa":                 "Nhãn khoa",
    "Sân phụ khoa":              "Sản phụ khoa",
    "Hổi - Đáp":                 "Hỏi đáp",
    "Hổi đáp":                   "Hỏi đáp",
    "Sân khấu & Mỹ thuật":       "Sân khấu - Mỹ thuật",
    "Lăng mốt":                  "Lăng mốt",   # giữ (không merge nhầm)
}

# ─────────────────────────────────────────────────────────────────────────────
# TIN_TUC_MAP  — phân loại nhãn L2 = "Tin tức" theo L1 cha
# Đầy đủ cho tất cả 13 L1 → không mất bài nào
# ─────────────────────────────────────────────────────────────────────────────
TIN_TUC_MAP = {
    "Sức khỏe":           "Tin tức sức khỏe",   # → L2 riêng
    "Xe":                 "Tin tức xe",           # → L2 riêng
    "Khoa học công nghệ": "Chuyển đổi số",        # gộp vào CĐS
    "Kinh doanh":         "Thị trường",           # gộp vào Thị trường
    "Thể thao":           "Các môn khác",         # gộp vào Các môn khác
    "Giáo dục":           "Tuyển sinh",           # gộp vào Tuyển sinh
    "Thời sự":            "Dân sinh",             # gộp vào Dân sinh
    "Thế giới":           "Quốc tế",              # gộp vào Quốc tế
    "Du lịch":            "Điểm đến",             # gộp vào Điểm đến
    "Đời sống":           "Nhịp sống",            # gộp vào Nhịp sống
    "Pháp luật":          "Hồ sơ phá án",         # gộp vào HSPA
    "Giải trí":           "Giới sao",             # gộp vào Giới sao
    "Khoa học":           "Thế giới tự nhiên",    # gộp vào TGTN
}

# ─────────────────────────────────────────────────────────────────────────────
# L2_MERGE  — gộp nhãn L2 trùng/tương đồng → nhãn đích trong hierarchy
# ─────────────────────────────────────────────────────────────────────────────
L2_MERGE = {
    # Thế giới
    "Bắc Mỹ":                    "Quốc tế",
    # Kinh doanh
    "Tiêu dùng":                 "Hàng hoá",
    "Ebank":                     "Thị trường",        # 28 bài → Thị trường
    # Pháp luật
    "Dân sự":                    "Hồ sơ phá án",
    # Du lịch
    "Dấu chân":                  "Điểm đến",
    # Khoa học công nghệ
    "Đổi mới sáng tạo":          "Chuyển đổi số",
    "Bộ Khoa học và Công nghệ":  "Chuyển đổi số",
    "Tin tức số hóa":            "Chuyển đổi số",
    # Đời sống
    "Văn hóa & lối sống":        "Nhịp sống",
    "Không gian sống":           "Tổ ấm",
    # Sức khỏe
    "Y tế & sức khỏe":           "Sống khỏe",
    # Thể thao — giải đấu bóng đá nhỏ → Bóng đá (L3 sẽ là Các giải khác)
    "Ngoại hạng Anh":            "Bóng đá",
    "V-League":                  "Bóng đá",
    "Bundesliga":                "Bóng đá",
    "La Liga":                   "Bóng đá",
    "Europa League":             "Bóng đá",
    "Serie A":                   "Bóng đá",
    # Thể thao — Vô thuật / Võ thuật
    "Vô thuật":                  "Các môn khác",
    "Võ thuật":                  "Các môn khác",
    # Giải trí — Hậu trường
    "Hậu trường":                "Giới sao",
    # Thời sự
    "Chính trị & chính sách":    "Chính trị",
    # Marathon (L2 hiện là Marathon nhưng hierarchy đặt ở Thể thao)
    "Sống khỏe":                 "Sống khỏe",   # giữ (đã trong hierarchy)
}

# L2 → L3 mặc định khi L2 bị merge vào "Bóng đá" từ tên giải đấu
# (chỉ dùng khi L3 gốc rỗng)
L2_TO_DEFAULT_L3 = {
    "Ngoại hạng Anh": "Các giải khác",
    "V-League":       "Trong nước",
    "Bundesliga":     "Các giải khác",
    "La Liga":        "Các giải khác",
    "Europa League":  "Các giải khác",
    "Serie A":        "Các giải khác",
}

# ─────────────────────────────────────────────────────────────────────────────
# L2_DROP  — loại bỏ hoàn toàn (nhãn format / sự kiện / < 5 bài)
# ─────────────────────────────────────────────────────────────────────────────
L2_DROP = {
    # Format / media
    "Video", "Ảnh", "Podcast", "Trắc nghiệm",
    # Sự kiện ngắn hạn
    "Mekong", "Kỷ nguyên mới",
    "Bảo vệ trọn vẹn từng khoảnh khắc",
    "Bầu cử Đại biểu Quốc hội khóa 16",
    "Kun Marathon", "Giáo dục 4.0", "Người Việt 5 châu",
    "Cửa sổ tri thức", "Của sổ tri thức",
    "Học tiếng Anh", "Kinh tế vùng", "Tiền của tôi",
    "Khoa học trong nước", "Nội trợ",
    "GameVerse 2026", "GameVerse 2025", "esport", "Bảo hiểm",
    "Sống tinh tế trong thầm lặng", "Doanh nghiệp vươn mình",
    "Sáng kiến khoa học 2025", "Sáng kiến khoa học 2026",
    "Car Awards 2025", "Sea Games 33",
    "VnExpress Youth Basketball 2025",
    "80 năm vinh quang Thể thao Việt Nam",
    "Việc làm", "Quỹ Hy vọng", "Quỹ hy vọng",
    "Dự án", "Xe điện",
    "Trao đổi yêu, giữ sức khỏe",
    "Hệ sinh thái PC AI của HP", "Phòng vệ HPV",
    "Đại hội Đảng XIV", "Chính sách",
    "Câu chuyện ngành",
    "Miễn dịch khỏe, bé lớn khôn",
    "Cooking",  # đã normalize ở trên, chỉ phòng hờ
}

# ─────────────────────────────────────────────────────────────────────────────
# L3_MERGE  — gộp nhãn L3 trùng/tương đồng
# Giá trị rỗng "" = drop (bài vẫn giữ nếu L2 hợp lệ)
# ─────────────────────────────────────────────────────────────────────────────
L3_MERGE = {
    # Giải trí — giữ nguyên các nhãn chính
    "Sao đẹp - Sao xấu":   "Sao đẹp - Sao xấu",
    "Chuyện màn ảnh":       "Chuyện màn ảnh",
    "Lăng nhạc":            "Lăng nhạc",
    "Lăng mốt":             "Lăng mốt",
    "Lăng văn":             "Lăng văn",
    "Nhịp sống số":         "Nhịp sống số",
    "Bạn đọc viết":         "Nhịp sống số",    # gộp vào Nhịp sống số
    "Sân khấu":             "Sân khấu",
    "Mỹ thuật":             "Mỹ thuật",
    "Điểm sách":            "Điểm sách",

    # Sức khỏe — bệnh (các tên cụ thể giữ nguyên)
    "Nhãn khoa":            "Nhãn khoa",   # sau khi normalize từ "Nhân khoa"
    "Tim mạch":             "Tim mạch",
    "Nội tiết":             "Nội tiết",
    "Thần kinh":            "Thần kinh",
    "Tiêu hóa":             "Tiêu hóa",
    "Da liễu":              "Da liễu",
    "Hô hấp":               "Hô hấp",
    "Sản phụ khoa":         "Sản phụ khoa",
    "Cơ xương khớp":        "Cơ xương khớp",
    "Ung thư":              "Ung thư",
    "Tai mũi họng":         "Tai mũi họng",
    "Tiết niệu - Nam học":  "Tiết niệu - Nam học",
    "Nhi - Sơ sinh":        "Nhi - Sơ sinh",

    # Sức khỏe — tư vấn / hỏi đáp
    "Hỏi đáp":              "Hỏi đáp",
    "Ý kiến":               "Hỏi đáp",
    "meo-tu-van":           "Hỏi đáp",
    "Y kiến":               "Hỏi đáp",
    "Khỏe đẹp":             "Dinh dưỡng",
    "Hỏi - Đáp":            "Hỏi đáp",
    "Tư vấn":               "Hỏi đáp",        # L3 Tư vấn trong Sức khỏe → Hỏi đáp

    # Sức khỏe — Vaccine
    "Vaccine người lớn":    "Vaccine người lớn",

    # Thể thao — Bóng đá
    "Trong nước":           "Trong nước",
    "Champions League":     "Champions League",
    "Các giải khác":        "Các giải khác",
    "Ngoại hạng Anh":       "Các giải khác",
    "La Liga":              "Các giải khác",
    "V-League":             "Trong nước",      # V-League = bóng đá Việt Nam
    "Bundesliga":           "Các giải khác",
    "Serie A":              "Các giải khác",
    "Europa League":        "Các giải khác",
    # "Thế giới" ở L3 Bóng đá → Các giải khác (tránh nhầm với L1)
    "Thế giới":             "",                # xử lý bên dưới tuỳ ngữ cảnh L2

    # Xe
    "Tình huống":           "Kỹ năng lái",
    "Luật giao thông":      "Kỹ năng lái",
    "Chăm sóc xe":          "Kỹ năng lái",
    "Kinh nghiệm":          "Kỹ năng lái",
    "Đánh giá xe":          "Đánh giá xe",
    "Car test":             "Đánh giá xe",

    # Giáo dục
    "Đại học":              "Đại học",
    "Lớp 10":               "Đại học",         # gộp vào Tuyển sinh / Đại học

    # Pháp luật
    "Hình sự":              "",                # drop L3, giữ L2 Hồ sơ phá án

    # DROP hoàn toàn (bài vẫn giữ nếu L2 OK)
    "Thị trường":           "",
    "Giao thông":           "",
    "Sản phẩm":             "",
    "Diễn đàn":             "",
    "Bóng đá":              "",
    "Tin tức":              "",
    "Quốc tế":              "",
    "Việt Nam":             "",
    "Chính sách":           "",
    "Bình luận":            "",
    "Tin dự án":            "",
    "Giày & Phụ kiện":      "",
    "Cẩm nang Net Zero":    "",
    "Nhịp sống":            "",   # tên L2, không phải L3
    "Tường thuật":          "",
    "Phân tích":            "",   # tên L2, không phải L3
    "Điền đa":              "",
    "Điền kinh":            "",
    "Điển kinh":            "",
    "Điểm phim":            "",
    "Chân dung":            "",   # tên L2, không phải L3 Sức khỏe
    "Ngân hàng":            "",
    "Cờ vua":               "",
    "Tác giả":              "",
    "Nhân sự":              "",
    "Sáng kiến":            "",
    "Dua xe":               "",
    "Ứng dụng":             "",
    "Doanh nhân":           "",
    "Doanh nghiệp xanh":    "",
    "Game talk":            "",
    "VMC":                  "",
    "Hôn nhân":             "",
    "Quản trị":             "",
    "Nông nghiệp":          "",
    "Ngân hàng sáng tạo":   "",
    "Câu chuyện ngành":     "",
    "Europe League":        "Các giải khác",
    "Các môn thể thao khác":"",
    "Diff":                 "",
    "home A":               "",
    "Im dự án":             "",
    "Sáng kiến":            "",
    "Cẩm nang đầu tư F0":   "",
    "Làm đẹp":              "",
    "Nước sạch đô thị":     "",
    "Vaccine trẻ em":       "",
    "Chăm sóc miễn dịch":   "",
    "Dinh dưỡng miễn dịch": "",
    "Đổi mới sáng tạo":     "",
    "Hành tinh kiếu cứu":   "",
    "Câu chuyện bảo hiểm":  "",
    "Thúc đẩy Khoa học Công nghệ": "",
    "Bộ sưu tập":           "",
    "Sống bền vững":        "",
    "Hiếm muộn":            "",    # < 10 bài, giữ L2 Các bệnh
    "Vốc tâm":              "",
    "Sàn chắm":             "",
    "Doanh nghiệp viết":    "",
    "Chuyện đời số":        "",
    "Dầu ăn thương hiệu":   "",
    "Nhà - Sơ sinh":        "",    # gộp vào L2 Sống khỏe
}

# ─────────────────────────────────────────────────────────────────────────────
# L3_LOW_FREQ_MERGE — tầng merge bổ sung cho các nhãn L3 đuôi thấp
# Không thay thế L3_MERGE cũ, chỉ áp dụng thêm sau bước L3_MERGE.
# ─────────────────────────────────────────────────────────────────────────────
L3_LOW_FREQ_MERGE = {
    # Xe
    "Đánh giá xe":          "Kỹ năng lái",

    # Sức khỏe (đuôi thấp) -> giữ ở mức L2
    "Dinh dưỡng":           "",
    "Vaccine người lớn":    "",
    "Cơ xương khớp":        "",
    "Tai mũi họng":         "",
}

# ─────────────────────────────────────────────────────────────────────────────
# L3_DROP  — loại bỏ hoàn toàn (không tìm L2 rescue)
# ─────────────────────────────────────────────────────────────────────────────
L3_DROP = {
    "Video", "Ảnh", "Trắc nghiệm",
    "Tướng thuật",
    "Kun Marathon Hồ Chí Minh",
}

# ─────────────────────────────────────────────────────────────────────────────
# L3_RESCUE  — khi L3 bị drop, tìm L2 thay thế từ tên L3
# ─────────────────────────────────────────────────────────────────────────────
L3_RESCUE = {
    "Giao thông":    "Giao thông",
    "Bóng đá":       "Bóng đá",
    "Diễn đàn":      "Diễn đàn",
    "Thị trường":    "Thị trường",
    "Hình sự":       "Hồ sơ phá án",
    "Dân sự":        "Hồ sơ phá án",
    "Chính sách":    "Chính trị",
    "Việt Nam":      "Quốc tế",
    "Kinh nghiệm":   "Cầm lái",
    "Chăm sóc xe":   "Cầm lái",
    "Nhà - Sơ sinh": "Sống khỏe",
    "Hiếm muộn":     "Các bệnh",
    "Chân dung":     "Chân dung",
    "Phân tích":     "Phân tích",
    "Ngân hàng":     "Thị trường",
    "Đầu tư":        "Thị trường",
}


# ─────────────────────────────────────────────────────────────────────────────
# resolve_labels  — hàm chính
# ─────────────────────────────────────────────────────────────────────────────
def resolve_labels(l1: str, l2: str, l3: str) -> tuple[str, str, str]:
    """
    Normalize + merge + validate → (l1, l2, l3) sạch.

    Rules (theo thứ tự ưu tiên):
      1. Normalize typo / biến thể tên
      2. L1 phải thuộc L1_SET — nếu không → bỏ bài ('', '', '')
      3. Xử lý L2:
         a. 'Tin tức' → tra TIN_TUC_MAP theo L1
         b. Drop → ''
         c. Là tên giải đấu → merge vào 'Bóng đá', gán L3 default nếu chưa có
         d. Merge → nhãn đích
      4. Xử lý L3:
         a. Drop hoàn toàn (L3_DROP) → ''
         b. Merge (L3_MERGE) → nhãn đích hoặc ''
         c. Không nhận ra → rescue L2 nếu có thể, rồi xóa L3
      5. Enforce parent-child: L3 có → L2 phải có
      6. Validate: L2/L3 phải thuộc set của hierarchy
    """
    # ── Bước 1: Normalize ────────────────────────────────────────────────────
    l1 = NORMALIZE.get(l1.strip(), l1.strip())
    l2 = NORMALIZE.get(l2.strip(), l2.strip())
    l3 = NORMALIZE.get(l3.strip(), l3.strip())

    # ── Bước 2: Validate L1 ──────────────────────────────────────────────────
    if l1 not in L1_SET:
        return "", "", ""

    # ── Bước 3: Xử lý L2 ─────────────────────────────────────────────────────
    _l2_orig = l2  # lưu để tra L2_TO_DEFAULT_L3 sau

    if l2 == "Tin tức":
        l2 = TIN_TUC_MAP.get(l1, "")

    elif l2 in L2_DROP:
        l2 = ""

    elif l2 in L2_TO_DEFAULT_L3:
        # Giải đấu bóng đá (Ngoại hạng Anh, V-League, ...)
        if not l3:
            l3 = L2_TO_DEFAULT_L3[_l2_orig]
        l2 = "Bóng đá"

    else:
        l2 = L2_MERGE.get(l2, l2)

    # Validate L2
    if l2 and l2 not in L2_SET:
        l2 = ""

    # ── Bước 4: Xử lý L3 ─────────────────────────────────────────────────────
    if l3 in L3_DROP:
        l3 = ""

    elif l3 in L3_MERGE:
        merged = L3_MERGE[l3]
        if merged == "" and not l2:
            # Thử rescue L2
            rescue = L3_RESCUE.get(l3, "")
            if rescue in L2_SET:
                l2 = rescue
        l3 = merged

    elif l3:
        # Không nhận ra → thử rescue L2 rồi xóa L3
        rescue = L3_RESCUE.get(l3, "")
        if not l2 and rescue in L2_SET:
            l2 = rescue
        l3 = ""

    # Tầng merge bổ sung cho nhãn L3 đuôi thấp
    if l3 in L3_LOW_FREQ_MERGE:
        merged = L3_LOW_FREQ_MERGE[l3]
        if merged == "" and not l2:
            rescue = L3_RESCUE.get(l3, "")
            if rescue in L2_SET:
                l2 = rescue
        l3 = merged

    # Đặc biệt: L3 = "Thế giới" chỉ hợp lệ nếu L2 = "Bóng đá"
    if l3 == "Thế giới" and l2 != "Bóng đá":
        l3 = ""

    # Validate L3
    if l3 and l3 not in L3_SET:
        l3 = ""

    # ── Bước 5: Enforce parent-child ─────────────────────────────────────────
    if l3 and not l2:
        l2 = L3_TO_L2.get(l3, "")
    if l2 and not l1:
        l1 = L2_TO_L1.get(l2, "")

    # ── Bước 6: Final validate ────────────────────────────────────────────────
    if l2 and l2 not in L2_SET:
        l2 = ""
    if l3 and l3 not in L3_SET:
        l3 = ""

    return l1, l2, l3


# ─────────────────────────────────────────────────────────────────────────────
# LABEL_MAP  — chỉ số cố định cho training
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "l1": {lb: i for i, lb in enumerate(sorted(L1_SET))},
    "l2": {lb: i for i, lb in enumerate(sorted(L2_SET))},
    "l3": {lb: i for i, lb in enumerate(sorted(L3_SET))},
}


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"L1: {len(LABEL_MAP['l1'])}  "
          f"L2: {len(LABEL_MAP['l2'])}  "
          f"L3: {len(LABEL_MAP['l3'])}")
    print()

    cases = [
        # (input_l1,            input_l2,            input_l3,             expected_l2,           expected_l3)
        ("Thời sự",             "Tin tức",            "",                   "Dân sinh",            ""),
        ("Sức khỏe",            "Tin tức",            "",                   "Tin tức sức khỏe",   ""),
        ("Khoa học công nghệ",  "Tin tức",            "",                   "Chuyển đổi số",       ""),
        ("Kinh doanh",          "Tin tức",            "",                   "Thị trường",          ""),
        ("Giải trí",            "Tin tức",            "",                   "Giới sao",            ""),
        ("Thể thao",            "Ngoại hạng Anh",     "",                   "Bóng đá",             "Các giải khác"),
        ("Thể thao",            "V-League",           "",                   "Bóng đá",             "Trong nước"),
        ("Thể thao",            "Bóng đá",            "Champions League",   "Bóng đá",             "Champions League"),
        ("Thể thao",            "Bóng đá",            "La Liga",            "Bóng đá",             "Các giải khác"),
        ("Thể thao",            "Bóng đá",            "Ngoại hạng Anh",     "Bóng đá",             "Các giải khác"),
        ("Giải trí",            "Giới sao",           "Sao đẹp - Sao xấu",  "Giới sao",            "Sao đẹp - Sao xấu"),
        ("Giải trí",            "Giải sao",           "Chuyện màn ảnh",     "Giới sao",            "Chuyện màn ảnh"),  # normalize
        ("Giải trí",            "Hậu trường",         "",                   "Giới sao",            ""),
        ("Sức khỏe",            "Các bệnh",           "Nhân khoa",          "Các bệnh",            "Nhãn khoa"),  # sửa typo
        ("Sức khỏe",            "Vaccine",            "Vaccine người lớn",  "Vaccine",             ""),
        ("Sức khỏe",            "Các bệnh",           "meo-tu-van",         "Các bệnh",            "Hỏi đáp"),  # L2 giữ Các bệnh vì là input hợp lệ
        ("Xe",                  "Cầm lái",            "Tình huống",         "Cầm lái",             "Kỹ năng lái"),
        ("Xe",                  "Cầm lái",            "Luật giao thông",    "Cầm lái",             "Kỹ năng lái"),
        ("Xe",                  "Cầm lái",            "Chăm sóc xe",        "Cầm lái",             "Kỹ năng lái"),
        ("Kinh doanh",          "Tiêu dùng",          "",                   "Hàng hoá",            ""),
        ("Kinh doanh",          "Ebank",              "",                   "Thị trường",          ""),
        ("Khoa học",            "Thế giới tự nhiên",  "",                   "Thế giới tự nhiên",   ""),
        ("Đời sống",            "Nhịp sống",          "Bạn đọc viết",       "Nhịp sống",           "Nhịp sống số"),
        ("Số hóa",              "AI",                 "",                   "AI",                  ""),   # normalize L1
        ("cooking",             "",                   "",                   "",                    ""),   # normalize L2 → L1 mất
        ("Giáo dục",            "Tuyển sinh",         "Lớp 10",             "Tuyển sinh",          "Đại học"),
        ("Thế giới",            "Bắc Mỹ",             "",                   "Quốc tế",             ""),
        ("Du lịch",             "Dấu chân",           "",                   "Điểm đến",            ""),
    ]

    header = f"{'L1 in':<22} {'L2 in':<22} {'L3 in':<22} → {'L1':<22} {'L2':<22} {'L3'}"
    print(header)
    print("─" * 130)

    passed = 0
    for row in cases:
        if len(row) == 5:
            l1i, l2i, l3i, exp_l2, exp_l3 = row
        else:
            l1i, l2i, l3i = row
            exp_l2 = exp_l3 = None

        r_l1, r_l2, r_l3 = resolve_labels(l1i, l2i, l3i)

        ok = True
        if exp_l2 is not None and r_l2 != exp_l2:
            ok = False
        if exp_l3 is not None and r_l3 != exp_l3:
            ok = False

        flag = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"{flag} {l1i:<21} {l2i:<22} {l3i:<22} → {r_l1:<22} {r_l2:<22} {r_l3}")

    print()
    print(f"Passed: {passed}/{len(cases)}")
