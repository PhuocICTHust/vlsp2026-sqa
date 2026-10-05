# Hướng dẫn thiết lập & sử dụng Label Studio cho VLSP 2026 SQA

Hệ thống Label Studio đã được thiết lập sẵn sàng qua Docker container, tích hợp sẵn cấu hình gán nhãn và thẩm định lỗi âm thanh theo đúng kế hoạch triển khai.

---

## 1. Trạng thái & Tài khoản đăng nhập

Container Label Studio đang chạy tại:
* **URL:** [http://localhost:8080](http://localhost:8080)
* **Tài khoản đăng nhập:** Đăng nhập bằng tài khoản Label Studio của bạn.
* **Dữ liệu database:** `./ls_data`
* **Mount thư mục dự án:** `/label-studio/files` (được đọc trực tiếp từ thư mục gốc `vlsp2026-sqa`)

### Các lệnh tiện ích:
* Khởi động: `./label_studio/start.sh`
* Tắt container: `./label_studio/stop.sh`
* Xem logs: `docker logs -f vlsp_label_studio`

---

## 2. Các bước thiết lập Project trên giao diện Web

1. **Mở trình duyệt:** Truy cập [http://localhost:8080](http://localhost:8080)
2. **Đăng nhập:** Đăng nhập bằng tài khoản cá nhân của bạn.
3. **Tạo Project mới:**
   * Nhấn **Create Project**.
   * Đặt tên: `VLSP 2026 Fault Audit & DSP Review`.
4. **Cấu hình giao diện gán nhãn (Labeling Interface):**
   * Trong cửa sổ tạo project, chọn tab **Labeling Setup** $\rightarrow$ chọn **Custom template** (hoặc nút **Code** ở góc trên bên phải).
   * Xoá code mặc định và sao chép toàn bộ nội dung từ file [label_studio/config.xml](config.xml) dán vào.
   * Nhấn **Save**.

---

## 3. Quy trình Thẩm định & Duyệt nhãn

### Cấu hình giao diện hỗ trợ:
1. **Audio Waveform:** Cho phép nghe và quét chọn trực tiếp khoảng thời gian lỗi trên sóng âm.
2. **3 loại lỗi:**
   * `noise` (màu đỏ)
   * `distortion` (màu cam)
   * `unnatural_pause` (màu xanh dương)
3. **4 trạng thái bắt buộc theo kế hoạch:**
   * `absent`: File chắc chắn không có lỗi này.
   * `localized`: Có lỗi và đã khoanh vùng mốc giây chính xác.
   * `present_unknown_time`: Biết chắc có lỗi nhưng không rõ ở mốc giây nào.
   * `unknown`: Không đủ thông tin xác định.
4. **Duyệt ứng viên DSP (DSP Decision):**
   * `accepted`: Đồng ý với khoảng lỗi do DSP phát hiện $\rightarrow$ Kiên sẽ cập nhật `fault_status = localized`.
   * `rejected`: Bác bỏ $\rightarrow$ Giữ nguyên trạng thái trước đó.
   * `pending`: Chưa rõ ràng $\rightarrow$ Tiếp tục giữ lại, loại khỏi baseline hard-label.
5. **Ghi chú thẩm định (Audit Notes):** Ghi rõ lý do ghi đè hoặc nhận định cá nhân.

---

## 4. Import & Export dữ liệu

### A. Chuẩn bị file import từ DSP candidates:
```bash
python label_studio/format_import_tasks.py --input /path/to/dsp_candidates.json --output tasks_to_import.json
```
Sau đó trên giao diện Label Studio, nhấn **Import** và tải file `tasks_to_import.json` lên.

### B. Xuất dữ liệu sau khi duyệt để bàn giao cho Kiên:
1. Trên giao diện Label Studio, nhấn **Export** $\rightarrow$ chọn định dạng **JSON**.
2. Chạy script chuẩn hoá nhãn:
```bash
python label_studio/parse_export_results.py --input project_export.json --output reviewed_labels_for_kien.json
```
File `reviewed_labels_for_kien.json` sẽ chứa các interval đã được chuẩn hoá tự động bằng `normalize_intervals()` và thông tin người duyệt `resolved_by = "manual_correction:Hieu"`.
