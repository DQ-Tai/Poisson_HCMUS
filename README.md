# Thống kê trong phân rã hạt nhân ☢️

Phần mềm tự động tính toán các sai số, vẽ đồ thị hội tụ, phân bố Poisson/Gauss và chạy giả lập Monte Carlo.

---

## ⚙️ Cấu hình yêu cầu

Để chạy được mã nguồn này, máy tính cần:
1. **Python (phiên bản 3.6 trở lên):** Nếu chưa có, bạn có thể tải tại [python.org](https://www.python.org/downloads/) (hoặc có thể dùng VS code)
2. Một phần mềm gõ lệnh: **Terminal** (trên macOS/Linux) hoặc **Command Prompt / PowerShell** (trên Windows).

---

## 🛠 Hướng dẫn cài đặt 

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

## 🚀 Hướng dẫn sử dụng

### 1. Phân tích dữ liệu thực nghiệm
Chạy file `poisson.py` (hoặc `data_analysis.py` tùy tên bạn lưu) để phần mềm xử lý dữ liệu thực tế:
```bash
python poisson.py
```
- Code sẽ tính toán và tự động sinh ra các file đồ thị đuôi `.pdf`** và `.png`**
- Một file văn bản tên là `Bao_cao_Thong_ke.txt` sẽ được tạo ra. File này chứa toàn bộ các đặc trưng thống kê cần thiết và phân tích chi tiết, kèm với dữ liệu từ mô phỏng Monte Carlo.

python monte_carlo.py
```
- Code sẽ sinh ra thêm **3 file đồ thị `.pdf`** mới. Các chấm đỏ trên đồ thị chính là kết quả của 1000 lần quay xổ số ngẫu nhiên (Monte Carlo) dựa trên giá trị kỳ vọng ban đầu.
---
