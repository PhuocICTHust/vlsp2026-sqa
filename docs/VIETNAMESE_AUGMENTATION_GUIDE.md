# HƯỚNG DẪN CHI TIẾT: TỔNG HỢP & AUGMENT DỮ LIỆU TIẾNG VIỆT CHO VLSP 2026 SQA

> **Mục tiêu**: Chuẩn bị tập dữ liệu Speech Quality Assessment tiếng Việt (~1,000 – 1,500 mẫu) nhằm tăng cường độ mạnh (robustness) cho mô hình trước thềm **Private Test (9/10)**, phòng trường hợp Ban tổ chức đưa audio tiếng Việt vào tập kiểm thử bí mật.

---

## 🎯 1. BẢNG PHÂN CÔNG TRÁCH NHIỆM TOÀN TEAM (RACI)

| Thành viên | Vai trò chính | Nhiệm vụ cụ thể | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Thảo** (`vdmhth`) | **Speech Lead & Audio Reviewer (Gatekeeper)** | • **Kiểm duyệt & nghiệm thu** chất lượng audio sạch của Đạt và kho noise của Hoàng.<br>• Viết engine can thiệp tín hiệu (Noise mixing, Codec VoIP/Clipping distortion, Pause insertion). | `scripts/augment_engine.py`<br>Biên bản duyệt Audio Gatekeeping<br>`data/vietnamese_aug/wav/` (~1,000 files `.wav`) |
| **Đạt** | **Clean Speech & Golden Benchmark** | • Thu thập giọng nói sạch đa dạng 3 miền (Bắc - Trung - Nam) & audio TTS VN.<br>• Chuẩn hóa âm lượng LUFS/RMS.<br>• Bàn giao cho **Thảo kiểm duyệt**.<br>• Xây dựng tập 100 mẫu "Golden Benchmark Test tiếng Việt" nội bộ. | `data/vietnamese_clean/` (~1,000 files `.wav`)<br>`data/vietnamese_golden_test/` (100 files + nhãn kiểm thử) |
| **Hoàng** | **Vietnamese Noise & Asset Bank** | • Thu thập và chuẩn hóa kho tạp âm đặc trưng Việt Nam (xe máy, còi xe, mưa mái tôn, quạt, quán café, babble tiếng Việt).<br>• Cắt lọc clip 10s–15s, chuyển chuẩn 16kHz mono.<br>• Bàn giao cho **Thảo kiểm duyệt**. | Thư mục `data/vietnamese_noise_bank/` (~50–80 files `.wav` tạp âm chuẩn) |
| **Hiếu** (`GMconqoor29`) | **Ground-truth & MOS Annotation** | • Đồng bộ timestamp lỗi (chuẩn xác từng frame).<br>• Xây dựng công thức ước lượng điểm MOS tổng thể ($1.0 - 5.0$). | `data/vietnamese_aug/labels.jsonl`<br>`data/vietnamese_aug/metadata.csv` |
| **Kiên** (`kienlh`) | **Quality Assurance & Verification** | • Dùng `dataset_audit.py` thẩm định tập augment.<br>• Kiểm tra validation schema, benchmark thử nghiệm trên model. | `reports/vietnamese_aug_audit.md`<br>Tích hợp vào `train_full.csv` |

---

## 🎙️ 2. NHIỆM VỤ CỦA ĐẠT (GIỌNG SẠCH 3 MIỀN & GOLDEN BENCHMARK)

### 2.1. Đa dạng hóa nguồn giọng nói sạch 3 miền
* **Vấn đề**: Tập VIVOS chủ yếu là giọng miền Nam trong phòng thu. Private Test có thể gồm giọng miền Bắc, Trung, Nam với độ tuổi và thiết bị thu khác nhau.
* **Nguồn lấy**:
  * Mozilla Common Voice tiếng Việt (lọc các clip được upvote cao, không nhiễu).
  * VIVOS (chọn lọc các câu phát âm tự nhiên).
  * Một số mẫu trích xuất từ TTS tiếng Việt hiện đại (FPT.AI / Zalo AI / Vbee / ViVoice) để mô phỏng nhóm `recent_tts`.
* **Tỷ lệ phân bổ mục tiêu**:
  * **40% giọng Bắc** (Hà Nội, đồng bằng Bắc Bộ).
  * **40% giọng Nam** (TP.HCM, Tây Nam Bộ).
  * **20% giọng Trung** (Đà Nẵng, Huế, Nghệ Tĩnh).
  * Cân bằng Nam/Nữ và người lớn/thanh niên.
* **Chuẩn hóa kỹ thuật**:
  * Format: 16,000 Hz, Single Channel (Mono), 16-bit PCM WAV.
  * Độ dài: từ $2.5\text{s}$ đến $10.0\text{s}$.
  * Chuẩn hóa âm lượng: Đưa về khoảng $-23\text{ LUFS}$ (hoặc RMS $\approx 0.05 - 0.1$) để tránh file quá to hoặc quá bé trước khi thêm nhiễu.
  * **Bàn giao vào thư mục**: `data/vietnamese_clean/` và **gửi link cho Thảo kiểm duyệt**.

### 2.2. Xây dựng Tập 100 Golden Benchmark Test tiếng Việt nội bộ
* Chọn lọc riêng **100 file thử thách** (Edge Cases):
  * Người nói ngắc ngứ, hắng giọng, nuốt âm.
  * Nhiễu mic rè (loose connection), méo tiếng Zalo Call khi mạng giật lag.
* Đạt cùng Hoàng nghe và tự chấm điểm MOS thật (1.0 – 5.0) cho 100 file này.
* Lưu vào `data/vietnamese_golden_test/` để làm bài test độc lập cho toàn team trước khi nộp bài.

---

## 🔊 3. NHIỆM VỤ CỦA HOÀNG (KHO TẠP ÂM ĐẶC THÙ VIỆT NAM)

### 3.1. Thu thập kho tạp âm thực tế
Tạp âm đời sống tại Việt Nam có những đặc trưng rất riêng mà các tập dữ liệu quốc tế không có:
1. **Nhiễu giao thông Việt Nam**: Tiếng còi xe máy, còi ô tô bim bim, tiếng pô xe, tiếng đường phố giờ tan tầm.
2. **Nhiễu thiết bị trong nhà**: Tiếng quạt trần, quạt cây quay vù vù (rất phổ biến khi người Việt ghi âm bằng tai nghe/mic máy tính), tiếng mưa rào gõ mái tôn.
3. **Nhiễu sinh hoạt / Babble**: Tiếng ồn quán cà phê, quán ăn vỉa hè, tiếng nhiều người nói chuyện rì rầm tiếng Việt ở hậu cảnh.
4. **Nhiễu công cộng**: Tiếng loa phát thanh phường, tiếng rao hàng rong đường phố.

### 3.2. Tiêu chuẩn kỹ thuật của file Tạp âm
* Thu thập / cắt gọt từ các nguồn âm thanh môi trường mở (Freesound, YouTube audio Creative Commons, hoặc ghi âm thực tế).
* Cắt thành các đoạn từ **10 giây đến 15 giây**.
* Chuyển đổi định dạng: **16,000 Hz, Mono, 16-bit PCM WAV**.
* Đặt tên rõ ràng theo nhóm:
  * `noise_traffic_01.wav`, `noise_traffic_02.wav`
  * `noise_fan_01.wav`, `noise_rain_01.wav`
  * `noise_cafe_babble_01.wav`, `noise_street_vendor_01.wav`
* **Bàn giao vào thư mục**: `data/vietnamese_noise_bank/` và **gửi link cho Thảo kiểm duyệt**.

---

## 🛡️ 4. QUY TRÌNH KIỂM DUYỆT CỦA THẢO (QUALITY GATEKEEPING) & AUDIO DSP ENGINE

Do **Thảo có chuyên môn sâu về Speech Processing**, Thảo đóng vai trò **Người gác cổng chất lượng âm thanh (Audio Gatekeeper)**:

### 4.1. Tiêu chí Thảo duyệt dữ liệu của ĐẠT (`data/vietnamese_clean/`):
* [ ] **Độ sạch tuyệt đối**: Nghe ngẫu nhiên 30–50 file, đảm bảo nền âm tĩnh lặng, không có tiếng quạt ù ngầm, không có tiếng vọng phòng (reverberation) quá mức.
* [ ] **Độ tự nhiên ngữ âm**: Câu nói rõ ràng, không bị ngắt cụt mất âm tiết đầu hoặc âm tiết cuối (không bị clipping ranh giới do cắt file lỗi).
* [ ] **Âm lượng đồng nhất**: Không có file nào quá bé hoặc bị nén méo tiếng (overdrive/clipping).
* ⚠️ *Nếu phát hiện file chưa đạt, Thảo yêu cầu Đạt lọc bỏ hoặc thay thế trước khi tiến hành bước tiếp theo.*

### 4.2. Tiêu chí Thảo duyệt kho tạp âm của HOÀNG (`data/vietnamese_noise_bank/`):
* [ ] **Tính đại diện**: Đầy đủ các nhóm tiếng còi xe, tiếng mưa, tiếng quạt và tiếng quán cà phê.
* [ ] **Không lấn át giọng nói**: File tạp âm không được chứa tiếng một người nói chuyện quá to và rõ ràng ở cự ly gần (sẽ khiến mô hình nhầm thành 2 người nói cùng lúc thay vì nhận diện là nhiễu nền).
* [ ] **Không dính artifact nén**: Không dùng file âm thanh bị nén bitrate thấp (mp3 64kbps tạo âm thanh kim loại giả tạo).
* ⚠️ *Nếu file noise nào bị lỗi, Thảo yêu cầu Hoàng cắt lại hoặc tăng/giảm âm lượng cho phù hợp.*

### 4.3. Viết Engine Can thiệp Tín hiệu (DSP Augment Engine)
Sau khi đã **Duyệt Đạt (Approved)** cho cả hai kho dữ liệu, Thảo tiến hành chạy engine:
* **Lỗi `noise`**: Trộn tạp âm đã duyệt vào khoảng $[t_{\text{start}}, t_{\text{end}}]$ với SNR ngẫu nhiên từ $0\text{ dB}$ (nhiễu rất nặng) đến $15\text{ dB}$ (nhiễu nhẹ).
* **Lỗi `distortion`**: 
  * *Méo Codec*: Downsample 8kHz (mô phỏng băng hẹp thoại G.711 / AMR) rồi upsample lại 16kHz.
  * *Clipping*: Nhân biên độ sóng với hệ số $1.5 - 3.0$ rồi kẹp ngưỡng `np.clip(wav, -0.95, 0.95)`.
  * *Reverberation*: Convolve tín hiệu với Room Impulse Response (RIR).
* **Lỗi `unnatural_pause`**: Chèn khoảng lặng $\Delta t \in [0.3s, 1.2s]$ vào giữa các từ đơn âm tiết hoặc ngắt cụt audio đột ngột.

---

## 📝 5. NHIỆM VỤ CỦA HIẾU (METADATA & GROUND-TRUTH)

Khi Thảo chạy engine, Hiếu phụ trách tự động ghi nhận metadata chuẩn xác:
* **File nhãn**: `data/vietnamese_aug/labels.jsonl`
  ```json
  {
    "file_id": "vi_aug_000123",
    "faults": {
      "noise": [[0.50, 2.30]],
      "distortion": [],
      "unnatural_pause": [[3.10, 3.80]]
    }
  }
  ```
* **Ước tính điểm MOS** (`Overall quality` $\in [1.0, 5.0]$):
  * Audio gốc sạch: $\text{Base MOS} \sim \mathcal{U}(4.5, 4.9)$.
  * Noise nặng ($0-5\text{ dB}$): Trừ $1.5 - 2.2$ điểm.
  * Noise nhẹ ($6-15\text{ dB}$): Trừ $0.6 - 1.2$ điểm.
  * Distortion (Clipping/Codec): Trừ $1.2 - 2.0$ điểm.
  * Unnatural Pause (> 0.5s): Trừ $0.8 - 1.5$ điểm.
  * $\text{Overall quality} = \max(1.0, \min(5.0, \text{Final MOS}))$.

---

## 🔎 6. NHIỆM VỤ CỦA KIÊN (AUDITING & INTEGRATION)

* Chạy kiểm toán tự động:
  ```bash
  python scripts/dataset_audit.py --csv data/vietnamese_aug/metadata.csv --wav-dir data/vietnamese_aug/wav/
  ```
* Tiêu chí nghiệm thu (Acceptance Criteria):
  1. Không có interval nào có $start \ge end$ hoặc $end > duration$.
  2. Toàn bộ audio đọc được bằng `soundfile`, đúng 16kHz, mono, không có `NaN` hay `Inf`.
* Trích xuất biểu diễn:
  ```bash
  python scripts/cache_wavlm.py --wav-dir data/vietnamese_aug/wav/ --out-dir data/qualispeech/wavlm_cache/
  ```
* Bàn giao tập dữ liệu sạch 100% cho Phước.

---

## ⏱️ 7. TIẾN ĐỘ THỜI GIAN & ĐIỂM KIỂM DUYỆT (08/10)

```
08h00 - 09h30: Đạt & Hoàng thu thập và chuẩn hóa dữ liệu.
09h30 - 10h15: THẢO KIỂM DUYỆT (Review & Gatekeeping Checklist) -> Đạt & Hoàng fix nếu có phản hồi.
10h15 - 12h00: Thảo (chạy DSP Engine) + Hiếu (sinh labels.jsonl & metadata.csv).
13h00 - 14h30: Kiên (Audit & Cache WavLM) + Đạt/Hoàng (chạy test trên tập 100 Golden Benchmark).
15h00        : Phước khởi động Retrain Full Model trên GPU Server RTX 4090.
```
