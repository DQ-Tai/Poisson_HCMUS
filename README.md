# Thống Kê Phân Rã Phóng Xạ ☢️

Phần mềm tự động tính toán các sai số, vẽ đồ thị hội tụ, phân bố Poisson/Gauss và chạy giả lập Monte Carlo.

---

## ⚙️ System requirement

Để chạy được mã nguồn này, máy tính của bạn cần có:
1. **Python (phiên bản 3.6 trở lên):** Nếu chưa có, bạn có thể tải tại [python.org](https://www.python.org/downloads/) (hoặc có thể dùng VS code)
2. Một phần mềm gõ lệnh: **Terminal** (trên macOS/Linux) hoặc **Command Prompt / PowerShell** (trên Windows).

---

## 🛠 Hướng dẫn Cài đặt (Installation)

**Bước 1: Tải mã nguồn về máy**
Bấm nút **Code -> Download ZIP** ở góc trên, giải nén ra, sau đó mở Terminal và di chuyển vào thư mục vừa giải nén.
Hoặc dùng lệnh Git nếu bạn đã quen:
```bash
git clone [https://github.com/DQ-Tai/Poisson_HCMUS.git](https://github.com/DQ-Tai/Poisson_HCMUS.git)
cd Poisson_HCMUS
```

**Bước 2: Cài đặt các thư viện toán học cần thiết**
Source sử dụng `numpy` để tính toán, `scipy` cho các hàm thống kê và `matplotlib` để vẽ đồ thị. Hãy gõ lệnh sau vào Terminal và nhấn Enter:
```bash
pip install numpy matplotlib scipy
```

---

## 🚀 Hướng dẫn Sử dụng (Usage)

### 1. Phân tích Dữ liệu Thực nghiệm
Chạy file `poisson.py` (hoặc `data_analysis.py` tùy tên bạn lưu) để phần mềm xử lý dữ liệu thực tế:
```bash
python poisson.py
```
- Code sẽ tính toán và tự động sinh ra **6 file đồ thị đuôi `.pdf`**
- Một file văn bản tên là `Bao_cao_Thong_ke.txt` sẽ được tạo ra. File này chứa toàn bộ các con số như: Trị trung bình, Phương sai thực nghiệm, khoảng cách $\chi^2$ và các kết luận 3-sigma để đưa vào báo cáo.

### 2. Chạy Mô phỏng Monte Carlo
Chạy lệnh bên dưới để giả lập hệ thống thử đếm tia phóng xạ 1000 lần:
```bash
python monte_carlo.py
```
- Code sẽ sinh ra thêm **3 file đồ thị `.pdf`** mới. Các chấm đỏ trên đồ thị chính là kết quả của 1000 lần quay xổ số ngẫu nhiên (Monte Carlo) dựa trên giá trị kỳ vọng ban đầu.
---

## 📁 Cấu trúc thư mục chi tiết

- 📄 **`poisson.py`**: Script xử lý số liệu thực nghiệm.
- 📄 **`monte_carlo.py`**: Script mô phỏng Monte Carlo.
- 📄 **`Bao_cao_Thong_ke.txt`**: Kết quả phân tích (Dùng để copy vào báo cáo).
- 📈 **`Fig_1_Conv_*.pdf`**: (3 file) Đồ thị chứng minh sự hội tụ của phép đo.
- 📈 **`Fig_2_Dist_*.pdf`**: (3 file) Đồ thị phân bố tần suất thực nghiệm.
- 📈 **`Fig_3_Sim_*.pdf`**: (3 file) Đồ thị phân bố từ mô phỏng Monte Carlo.

---
