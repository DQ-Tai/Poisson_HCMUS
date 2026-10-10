1. # Thống Kê Phân Rã Phóng Xạ ☢️

## 📁 Cấu trúc thư mục

*   **`data_analysis.py`**: Script xử lý số liệu thực nghiệm. Thực hiện vẽ đồ thị hội tụ, biểu đồ phân bố tần suất thực nghiệm so với lý thuyết (Poisson/Gauss) và tính $\chi^2$.
*   **`monte_carlo.py`**: Script chạy mô phỏng Monte Carlo, tạo 1000 lần thử giả lập dựa trên giá trị trung bình thực tế.
*   **`requirements.txt`**: Danh sách thư viện Python cần thiết.

## ⚙️ Cài đặt

Cài đặt các thư viện toán học và vẽ đồ họa bằng lệnh sau:

```bash
pip install numpy matplotlib scipy
```

## 🚀 Hướng dẫn sử dụng

**1. Phân tích dữ liệu thực nghiệm:**
Chạy lệnh bên dưới để phần mềm tự động phân tích, xuất các file đồ thị (PDF/PNG) và file văn bản báo cáo thông số `Bao_cao_Thong_ke.txt`.
```bash
python data_analysis.py
```

**2. Chạy mô phỏng Monte Carlo:**
Chạy lệnh bên dưới để tạo các đồ thị giả lập 1000 lần đếm.
```bash
python monte_carlo.py
```