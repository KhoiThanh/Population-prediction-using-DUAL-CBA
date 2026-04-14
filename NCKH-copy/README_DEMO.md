# 📊 Ứng dụng Dự báo Dân số Việt Nam (2025-2034)

## Giới thiệu

Ứng dụng demo này cho phép dự báo và phân tích xu hướng dân số của 63 tỉnh/thành phố Việt Nam trong 10 năm tới (2025-2034).

### Tính năng chính:
- 📈 **Biểu đồ xu hướng**: Hiển thị dữ liệu lịch sử (2011-2024) và dự báo (2025-2034)
- 📊 **Bảng dữ liệu chi tiết**: Xem dữ liệu cụ thể từng năm
- 🔄 **So sánh các tỉnh**: Xem tốc độ tăng trưởng dân số giữa các tỉnh
- 💡 **Phân tích tự động**: Cảnh báo và nhận xét về tăng trưởng dân số
- 📋 **Chỉ số chính**: Dân số 2024-2034, tốc độ tăng trưởng, thay đổi tuyệt đối

---

## Cách chạy ứng dụng

### 1. **Cài đặt dependencies**

```bash
pip install streamlit pandas numpy plotly
```

### 2. **Chạy ứng dụng**

```bash
streamlit run demo_vietnam_population.py
```

Ứng dụng sẽ mở trên trình duyệt tại địa chỉ: `http://localhost:8501`

---

## Cấu trúc dữ liệu

### File input:
- **`Vietnam_Final_63_Provinces.csv`**: Dữ liệu lịch sử (2011-2024)
  - Cột: `Area_ID`, `Area_Name`, `2011_VN`, `2012_VN`, ..., `2024_VN`

- **`artifacts/vietnam_transfer_forecast_10y.csv`**: Dự báo (2025-2034)
  - Cột: `Area_ID`, `Area_Name`, `source_last_year`, `pred_2025`, ..., `pred_2034`

---

## Cách sử dụng

1. **Chọn tỉnh/thành phố**: Sử dụng sidebar bên trái để chọn khu vực cần xem
2. **Xem biểu đồ**: Biểu đồ chính sẽ hiển thị xu hướng dân số
3. **Xem bảng dữ liệu**: Scroll xuống để xem bảng chi tiết
4. **So sánh**: Cuộn xuống để xem biểu đồ so sánh 15 tỉnh có tăng trưởng cao nhất

---

## Chỉ số được tính toán

### Metrics được hiển thị:
- **Dân số 2024**: Dân số thực tế năm 2024
- **Dân số 2034**: Dân số dự báo năm 2034
- **Tăng trưởng 10 năm**: Phần trăm thay đổi từ 2024 đến 2034
- **Tăng trưởng trung bình/năm**: Tốc độ tăng trưởng hàng năm

### Phân loại tăng trưởng:
- ✅ **Tăng cao** (>15%): Cần lên kế hoạch cơ sở hạ tầng
- ℹ️ **Tăng vừa** (5-15%): Tăng trưởng bình thường
- ⚠️ **Tăng chậm** (0-5%): Tăng trưởng thấp
- 🔴 **Giảm dân số** (<0%): Dân số suy giảm

---

## Model đằng sau

Ứng dụng sử dụng model **Dual-CBA (Dual-stream CNN-BiLSTM + Attention)**:
- Mạng CNN: Trích xuất đặc trưng từ dữ liệu chuỗi thời gian
- BiLSTM: Học quan hệ phụ thuộc dài hạn
- Attention: Cơ chế chú ý để trọng số hóa các bước thời gian quan trọng

---

## Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### Lỗi: "File not found"
Đảm bảo các file này tồn tại trong thư mục làm việc:
- `Vietnam_Final_63_Provinces.csv`
- `artifacts/vietnam_transfer_forecast_10y.csv`

### Ứng dụng chạy chậm
- Nếu cần tốc độ nhanh hơn, có thể cache dữ liệu hoặc sử dụng phiên bản đơn giản hơn

---

## Liên hệ & Phát triển

Để phát triển thêm, bạn có thể:
- Thêm các chỉ số dân học khác (tỷ lệ tăng tự nhiên, di cư, v.v.)
- Thêm bản đồ địa lý hiển thị dân số theo vùng
- Thêm so sánh với các kịch bản khác nhau
- Xuất dữ liệu thành các định dạng khác nhau (Excel, PDF, v.v.)

---

**Phiên bản**: 1.0
**Ngày tạo**: 2026-03-27
**Người tạo**: Claude Code
