# Báo cáo phần việc của Thảo: Kiểm duyệt audio & DSP Augmentation Engine tiếng Việt — VLSP 2026 SQA

| | |
|---|---|
| **Người phụ trách** | Thảo (`vdmhth`), Speech Lead & Audio Reviewer (Gatekeeper) |
| **Ngày** | 08–09/10/2026 |
| **Phiên bản engine** | `vi_aug_engine_v2.7` (bản chốt) |
| **Mã nguồn** | `scripts/augment_engine.py`, `tests/test_augment_engine.py` |
| **Input** | `data/vietnamese_audio_clean/` (Đạt, ~1.000 file), `data/vietnamese_noise_bank/` (Hoàng, ~60 file) |
| **Output** | `data/vietnamese_aug/` (wav + labels + params + metadata + report) |
| **Bàn giao cho** | Hiếu (MOS), Kiên (audit, cache WavLM, chia fold), Phước (retrain) |

> **Trạng thái:** đã gen xong 1.000 file trên dữ liệu thật (engine v2.7, seed 2026, mục 8). Còn **[ĐIỀN]**: biên bản nghe duyệt (mục 3.2) và kết quả nghe kiểm tra output (cuối mục 8).

---

## 0. Phạm vi và sản phẩm bàn giao

Báo cáo này chỉ gồm phần việc của Thảo. **Điểm MOS (`quality_score`) là phần của Hiếu** và không nằm trong báo cáo.

**Sản phẩm của Thảo:**

| Sản phẩm | Nội dung | Mục |
|---|---|---|
| `scripts/augment_engine.py` + `tests/test_augment_engine.py` | Engine kiểm duyệt và chèn lỗi, kèm bộ test tự động | 2, 4, 7 |
| Biên bản duyệt Audio Gatekeeping | `reports/gatekeeping_{clean,noise}.md/.csv` (tự động) + biên bản nghe duyệt + `reports/reject.txt` | 3 |
| `data/vietnamese_aug/wav/` | ~1.000 file audio đã chèn lỗi (mỗi audio sạch đúng 1 bản), 16 kHz mono PCM16 | 4, 8 |
| `data/vietnamese_aug/labels.jsonl` | **Nhãn thời gian lỗi** `{file_id, faults}`. Chỉ engine làm được, vì chỉ engine biết chính xác đã chèn lỗi ở đâu | 6 |
| `data/vietnamese_aug/params.jsonl` | Chi tiết mọi lỗi đã chèn: SNR, loại distortion, độ dài pause... Đầu vào để Hiếu tính MOS | 6 |
| `data/vietnamese_aug/metadata.csv` + `generation_report.md` | Bảng thông tin từng file (có `source_group_id` để chia fold) và thống kê cho paper | 6, 8 |

**Phân công cập nhật so với bảng RACI gốc (chốt 09/10):** `labels.jsonl` và `metadata.csv` chuyển từ Hiếu sang Thảo, vì engine sinh trực tiếp khi chèn lỗi. Hiếu chỉ còn phần MOS.

| Thành viên | Phần việc | Sản phẩm |
|---|---|---|
| **Thảo** | Kiểm duyệt đầu vào, engine chèn lỗi, **nhãn thời gian lỗi** | engine, biên bản duyệt, `wav/`, `labels.jsonl` (faults), `params.jsonl`, `metadata.csv` |
| **Hiếu** | **Chỉ MOS:** tính `quality_score` từ `params.jsonl` theo công thức mục 5 của guide | `quality_score` được thêm vào `labels.jsonl` |

---

## 1. Mục tiêu

Tạo tập Speech Quality Assessment tiếng Việt có nhãn để tăng độ bền của mô hình với domain tiếng Việt cho Private Test. Mỗi mẫu có:

- `faults` (phần của Thảo) gồm `noise`, `distortion`, `unnatural_pause`. Mỗi loại là danh sách interval `[start, end]` tính bằng giây, cùng format với file nộp bài, nên dùng lại được evaluator và validator của team;
- `quality_score` ∈ [1,0; 5,0] (phần của Hiếu): Hiếu tính từ `params.jsonl` rồi thêm vào `labels.jsonl`.

Theo review notes mục 2.6, augmentation được coi là **thí nghiệm về domain robustness**, không mặc định đưa vào baseline.

---

## 2. Quy trình và cách chạy

```
Đạt: clean speech ─┐
                   ├─► [1] audit (tự động) ─► [2] nghe duyệt (thủ công) ─► reports/reject.txt
Hoàng: noise bank ─┘                                                          │
                                                                              ▼
           [3] generate: pause → noise → distortion → gain ngẫu nhiên → PCM16
                                                                              │
           wav/ + labels.jsonl + params.jsonl + metadata.csv + generation_report.md
                                                                              │
           Hiếu (MOS) ─► Kiên (dataset_audit + cache WavLM + fold) ─► Phước (retrain)
```

```bash
pip install numpy scipy soundfile
python tests/test_augment_engine.py                     # phải ra "ALL TESTS PASSED"

python scripts/augment_engine.py audit \
  --clean-dir data/vietnamese_audio_clean --noise-dir data/vietnamese_noise_bank

python scripts/augment_engine.py generate --reject-list reports/reject.txt --seed 2026 \
  --clean-dir data/vietnamese_audio_clean --noise-dir data/vietnamese_noise_bank \
  --out-dir data/vietnamese_aug --variants-per-clean 1
```

- **Nghe thử trước:** thêm `--limit 20 --out-dir data/vi_aug_preview`.
- **Mỗi audio sạch chỉ sinh đúng 1 file** (team chốt 09/10): `--variants-per-clean 1`, đây cũng là mặc định. Không tạo nhiều bản từ cùng một audio. 1.000 file sạch cho ra khoảng 1.000 file output. Tham số `--variants-per-clean N` vẫn còn trong code để dùng cho ablation sau này, nhưng **không dùng** cho bản nộp.
- **Tài nguyên:** chỉ dùng CPU, không cần GPU. Benchmark trên máy 2 nhân:

| Cấu hình | Số file | Thời gian | Dung lượng |
|---|---|---|---|
| audit, 1.000 clean + 60 noise | — | ~10 s | — |
| `--variants-per-clean 1` | 1.000 | ~8 s | ~210 MB |
| `--variants-per-clean 3` | 3.000 | ~18 s | ~640 MB |

**Lý do chỉ gen 1 bản:** chỉ có ~1.000 câu nói gốc. Nhiều bản từ cùng một câu không làm tăng số người nói hay số câu, nhưng dễ khiến model học thuộc giọng thay vì học lỗi.

---

## 3. Kiểm duyệt dữ liệu đầu vào (Gatekeeping)

### 3.1. Kiểm tra tự động (`audit`)

| Cờ | Áp dụng | Ý nghĩa |
|---|---|---|
| `SR_*`, `CH_*` | cả hai | Không phải 16 kHz / mono. Engine vẫn tự resample và downmix, nhưng nên sửa ở nguồn |
| `CLIPPED` | cả hai | Hơn 0,01 % số mẫu chạm ±1,0 |
| `NAN_INF`, `UNREADABLE` | cả hai | File hỏng |
| `DUPLICATE` | cả hai | Trùng nội dung byte (SHA-1) |
| `BANDLIMITED_<4k`, `NO_ENERGY_>7k` | cả hai | Nghi nén bitrate thấp hoặc nguồn băng hẹp |
| `DURATION_OUT_OF_2.5-10s` | clean | Sai độ dài quy định |
| `RMS_OUT_OF_RANGE` | clean | RMS ngoài [0,02; 0,2] |
| `NOT_CLEAN(est_snr<30dB)` | clean | Nền ồn: SNR ước lượng (p95 − p10 năng lượng khung) < 30 dB |
| `CUT_AT_BOUNDARY` | clean | 30 ms đầu/cuối còn năng lượng giọng, nghi cắt mất âm tiết |
| `DURATION_OUT_OF_10-15s` | noise | Sai độ dài quy định |
| `NEAR_SILENT` | noise | Gần như im lặng |
| `VERY_DYNAMIC` | noise | Dải động > 30 dB: nghe xem có giọng người nói gần không |
| `BURSTY` | noise | Noise ngắt quãng (còi...). Chỉ để thông tin, engine xử lý được |
| `SPEECH_LIKE` | noise | Nhóm `loudspeaker`, hoặc tên nhóm có `talk` (vd `rain_talk`): bắt buộc nghe |
| `BAD_NAME` | noise | Tên không theo dạng `noise_<nhóm>[_<mã nguồn>]_NN.wav` |

**Tên file noise:** engine lấy nhóm bằng cách bỏ mọi phần số ở cuối tên:
- `noise_fan_395645_01` → `fan`
- `noise_rain_tin_23087_01` → `rain_tin`
- `noise_loudspeaker_350722_06` → `loudspeaker`

### 3.2. Biên bản duyệt (thủ công) **[ĐIỀN]**

**Clean speech (Đạt)** — tổng: ___ file, bị flag: ___, đã nghe: ___ file ngẫu nhiên.

- [ ] Nền âm tĩnh, không có tiếng quạt ù, không có reverb quá mức
- [ ] Không bị cắt mất âm tiết đầu/cuối
- [ ] Âm lượng đồng đều, không overdrive
- [ ] Phân bố vùng miền ~40 % Bắc / 40 % Nam / 20 % Trung; cân bằng nam/nữ

**Noise bank (Hoàng)** — tổng: ___ file, số nhóm: ___, bị flag: ___.

- [ ] Có các nhóm: quạt, mưa (mái tôn...), xe máy/giao thông, babble quán cà phê, loa phường. **Nhóm rao hàng (`street_vendor`) đã bị loại** theo quyết định ngày 09/10.
- [ ] Không có giọng một người nói gần, rõ chữ (dễ nhầm thành 2 người nói). Nghe kỹ `loudspeaker` và `rain_talk`.
- [ ] Không có artifact nén mp3 (âm kim loại)

| File bị loại | Nguồn | Lý do | Xử lý (thay / cắt lại / bỏ) |
|---|---|---|---|
| | | | |

Kết luận: ☐ Approved ☐ Approved có điều kiện ☐ Yêu cầu làm lại. Các file bị loại được ghi vào `reports/reject.txt`, mỗi dòng một tên file. Không cần chờ Đạt/Hoàng sửa xong mới gen: chạy lại cùng seed thì các file cũ ra giống hệt.

---

## 4. Thiết kế DSP engine

### 4.1. Chuỗi xử lý

1. **Chuẩn hóa mức giọng:** đưa active-speech RMS về 0,07 (VAD năng lượng, khung 20 ms), để SNR so sánh được giữa các file.
2. **unnatural_pause:** chạy đầu tiên vì làm thay đổi trục thời gian.
3. **reverb** (tùy chọn, tắt mặc định).
4. **noise.**
5. **distortion:** lỗi trên kênh truyền.
6. **Gain output ngẫu nhiên ±6 dB** (an toàn đỉnh ≤ 0,99, không bao giờ tạo thêm clipping), sau đó ghi PCM16.

**Nguyên tắc chung:** pause và noise đều **đặt tại khe giữa 2 âm tiết**. Khe là điểm năng lượng thấp hơn mức giọng ≥ 12 dB (tiếng Việt đơn âm tiết, nên khe ≈ ranh giới âm tiết/từ), cách đầu/cuối câu ≥ 0,3 s. Không tìm được khe thì chèn ít hơn, chứ không bao giờ cắt ngang âm tiết.

### 4.2. Tỉ lệ công thức lỗi (recipe)

Tỉ lệ được đặt trong code (`DEFAULT_RECIPE_P`). Đổi bằng tham số, ví dụ `--recipe-probs '{"clean": 0.2}'`.

| Recipe | Tỉ lệ | Recipe | Tỉ lệ |
|---|---|---|---|
| clean (không lỗi) | 12 % | noise + distortion | 10 % |
| noise | 20 % | noise + pause | 10 % |
| distortion | 18 % | distortion + pause | 8 % |
| pause | 15 % | cả 3 lỗi | 7 % |

Phần 12 % mẫu sạch dạy model rằng tiếng Việt chất lượng tốt thì không có lỗi và được điểm cao. Thiếu phần này, model dễ báo lỗi bừa trên file tiếng Việt sạch, làm giảm F1 và PCC.

### 4.3. Lỗi `unnatural_pause`: khoảng ngắt thuần

- Mỗi file có lỗi pause: **1–2 đoạn dừng** (từ v2.2; trước là 2–4, nhưng audio thật ngắn, trung vị 4,4 s, nên 2–4 đoạn nghe quá nhiều, tổng dừng ~0,9 s ≈ 20 % câu), mỗi đoạn **0,4–0,8 s** (từ v2.3; trước là 0,15–0,5 s, nhưng đoạn < ~0,25 s không phân biệt được với chỗ ngắt tự nhiên giữa các từ, nên thành nhãn nhiễu), **chèn thêm** vào khe giữa 2 âm tiết. Không xóa âm nào, các đoạn cách nhau ≥ 0,4 s. Nghe như người nói bị khựng, ngắt quãng.
- Đoạn dừng được lấp bằng **"room tone" của chính file** (từ v1.9): engine cắt các khung 10 ms yên tĩnh nhất của file, tức thấp hơn mức giọng ≥ 40 dB, rồi nối lại với crossfade 2,5 ms. Mức của đoạn lấp **luôn ≤ mức giọng − 45 dB**. Nếu file không có đoạn lặng thật (clip cắt sát), engine dùng nền rất nhỏ ở mức giọng − 60 dB. Không dùng số 0 tuyệt đối, và **không bao giờ chứa noise**. Ramp 5 ms chống click.
  - *Lỗi ở v1.8 đã sửa:* bản cũ lấy mức nền theo phân vị 10 của năng lượng khung. Với clip cắt sát, nói nhanh, phân vị này rơi vào phụ âm nhỏ, nên đoạn lấp chỉ thấp hơn giọng khoảng 26 dB và **nghe như một đoạn noise xè xè**. Bản v1.9 đo trên cùng file cho ra −54 dB, nghe là im lặng.
- Chế độ "ngắt cụt" (thay một đoạn giọng bằng im lặng) **tắt mặc định** vì nghe như bị mute mất âm tiết. Bật lại bằng `--p-dropout 0.25`.

### 4.4. Lỗi `noise`: đoạn chen ngắn tại khe âm tiết

- **Số lượng và độ dài:** mỗi file có noise được **1–2 đoạn, mỗi đoạn 0,3–1,5 s**, tổng ≤ 50 % độ dài file. Không có noise phủ toàn file.
- **Hai cách đưa noise vào, tỉ lệ 75/25** (team chốt 09/10, `--p-noise-insert 0.25`). Mỗi file chọn một cách:
  - **Trộn chồng lên giọng (overlay), ~75 % số file có noise:** đoạn noise bắt đầu tại khe giữa 2 âm tiết rồi trộn lẫn vào phần giọng phía sau, như quạt hoặc xe chen vào lúc đang nói. Giọng và noise nghe cùng lúc, độ dài audio không đổi. Các đoạn cách nhau ≥ 0,35 s và dừng trước pause kế tiếp ≥ 0,1 s.
  - **Chèn thêm vào khe (insert), ~25 %:** tách 2 âm tiết ra, nhét đoạn noise (cộng nền nhỏ của file) vào giữa. Audio dài thêm; nhãn pause phía sau được dời theo. Đoạn này chỉ gắn nhãn `noise`, không gắn `unnatural_pause`.
  - Đo trên 600 file giả lập, tỉ lệ thực tế là 78 % / 22 %.
- **Không đặt noise vào trong pause:** noise tránh các đoạn pause đã chèn ≥ 0,1 s.
- **SNR:** U(0; 15) dB, tính giữa **mức active** của giọng và của noise (kiểu ITU-T P.56), không dùng công suất trung bình. Với noise ngắt quãng như còi, cách tính trung bình làm tiếng còi to hơn mức yêu cầu khoảng 11 dB.
- **Chọn nhóm noise theo trọng số** (`--category-weights`):
  - `fan`: ×3
  - `rain*` (cả `rain_tin`, `rain_talk`), `traffic`, `motorbike`, `horn`, `street*`: ×0,4 (mưa hạ xuống bằng giao thông theo quyết định team 09/10, vì nghe mưa quá to và quá nhiều)
  - các nhóm khác: ×1

  Với kho noise thật (sau khi loại street_vendor), tỉ lệ bốc dự kiến là: quạt ~45 %, restaurant_babble ~15 %, loa phường ~15 %, rain_tin / rain_talk / xe máy / giao thông mỗi nhóm ~6 % (tổng mưa ~12 %). Mỗi file dùng **một nhóm noise và một SNR** cho mọi đoạn (cùng một môi trường); mỗi đoạn lấy một khúc ngẫu nhiên khác nhau.
- **Noise dạng giọng nói** (`loudspeaker`): SNR riêng **10–20 dB**, kèm xử lý "nghe như ở xa": lọc thông thấp 2,5–4,5 kHz và reverb RT60 0,4–0,9 s, DRR −6 đến 0 dB. Cấu hình bằng `--category-snr` và `--distant-categories`.
- **Nhãn:** chỉ phủ các khung 20 ms mà noise nghe thấy được, tức noise không nhỏ hơn giọng quá 30 dB. Fade 20 ms ở hai đầu đoạn.

### 4.5. Lỗi `distortion`

| Loại | Tỉ trọng | Mô tả |
|---|---|---|
| `narrowband` | 0,30 | Giống G.711: xuống 8 kHz, lọc 300–3400 Hz, µ-law 8 bit, lên lại 16 kHz. Nghe "đóng hộp", kiểu điện thoại |
| `clipping` | 0,30 | Kẹp cứng sao cho 2–15 % số mẫu giọng bị kẹp, rồi trả lại âm lượng (xem mục 5). Nghe vỡ tiếng, rè theo giọng |
| `codec_lowbitrate` | 0,25 | Mô phỏng codec nén bitrate rất thấp (MP3/AAC/Opus): mỗi khung 32 ms chỉ giữ 6–15 % thành phần phổ mạnh nhất, cắt dải cao 3,5–5,5 kHz. Nghe kim loại, "dưới nước", méo lơ lớ (từ v2.6) |
| `quantize` | tắt | Giảm độ sâu bit còn 4–6 bit. **Tắt từ v2.6** vì nghe như noise (xem bên dưới); bật lại bằng `--dist-kinds` |
| `packet_loss` | 0,15 | Mất 10–30 % số khung 20 ms, che bằng cách lặp khung trước (tiếng robot kiểu VoIP) |
| `reverb` | tắt | RIR tổng hợp, RT60 0,3–0,9 s; bật bằng `--reverb` |

- **Vì sao bỏ `quantize`** (team nghe thấy "distortion nghe như noise", 09/10): đo phần sai lệch mà mỗi loại méo thêm vào tín hiệu. Độ phẳng phổ của sai lệch: 1 = giống hệt tiếng xè trắng, 0 = bám theo giọng.

  | Loại | Độ phẳng phổ của sai lệch | Sai lệch lúc có giọng so với lúc im lặng |
  |---|---|---|
  | `quantize` | **0,97 (≈ tiếng xè trắng)** | 27 dB |
  | `codec_lowbitrate` | 0,49 | 35 dB |
  | `clipping` | 0,22 | 58 dB |
  | `packet_loss` | 0,02 | 17 dB |
  | `narrowband` | 0,01 | 52 dB |

  Giảm bit tạo ra lớp xè trắng phủ đều lên giọng, về bản chất giống noise nền. Nếu giữ, model sẽ bị dạy rằng tiếng xè lúc thì là `noise`, lúc thì là `distortion`, làm lẫn hai nhãn và giảm F1 cả hai loại. Các loại còn lại đều méo **theo giọng**: lỗi chỉ xuất hiện khi có tiếng nói và có cấu trúc riêng, đúng tính chất distortion.
- **Chỉ méo từng đoạn ngắn, không còn méo toàn file** (team chốt 09/10, từ v2.7): mỗi file có lỗi distortion được **1–2 đoạn, mỗi đoạn 0,4–1,5 s** (tối đa 35 % độ dài file mỗi đoạn), tổng ≤ 50 % file, các đoạn cách nhau ≥ 0,3 s, ghép vào bằng crossfade 10 ms. Mỗi file dùng một loại méo. Lý do: méo phủ toàn file nghe gần như bình thường vì không có đoạn sạch để so sánh. Chính sự chuyển sạch → méo → sạch mới làm distortion nghe ra được, và cho nhãn interval có ý nghĩa với IoU. Bật lại méo toàn file bằng `--p-dist-full`.
- Mọi loại đều được **cân lại về mức âm lượng gốc**, để không tạo bước nhảy âm lượng ở biên đoạn. Riêng codec 8 kHz nếu không cân sẽ làm giọng nhỏ đi 5,45 dB.

---

## 5. Quy ước nhãn và file output

| File | Nội dung |
|---|---|
| `wav/vi_aug_XXXXXX.wav` | 16 kHz, mono, PCM16 |
| `labels.jsonl` | `{"file_id", "faults": {noise, distortion, unnatural_pause}}`, đúng format mẫu ở mục 5 của guide. Hiếu thêm `quality_score` |
| `params.jsonl` | Toàn bộ tham số random của từng file: seed, recipe, các đoạn noise (`mode`, `spans`, `inserts`, file noise, offset, SNR), loại distortion, các đoạn pause (độ dài, mức đoạn lấp), gain và mức âm lượng. Đây là đầu vào để Hiếu tính MOS |
| `metadata.csv` | Bảng phẳng: `source_group_id`, `duration`, `n_frames`, `recipe`, các cờ `has_*`, `noise_snr_db`, `noise_category`, `noise_mode`, `noise_n_segments`, `noise_cover_s`, `distortion_kind`, `n_pauses`, `pause_total_s`, `active_level_dbfs`, `peak_dbfs`, `output_gain_db`, `seed` |
| `generation_report.md` | Thống kê tự sinh: tỉ lệ lỗi, recipe, nhóm noise, đoạn noise (cách chèn, số đoạn, độ phủ, histogram SNR), pause (số đoạn, tổng thời gian), loại distortion, mức âm lượng, config |

- **Interval** nằm trên trục thời gian của file **output**. Biên được làm tròn ra ngoài theo 10 ms. Interval chạm cuối file giữ đúng duration để không mất khung 20 ms cuối khi chấm IoU. Các interval đã được sort và merge.
- **`noise` và `unnatural_pause` không bao giờ chồng lên nhau.**
- **Tái lập:** seed mỗi file = sha1(seed chung | tên file clean | số biến thể). Chạy lại cùng seed ra wav và labels giống hệt từng byte.
- **Chống leakage:** `source_group_id` = tên file clean. Mọi biến thể của cùng một câu **phải nằm cùng một fold** (`StratifiedGroupKFold`).

---

## 6. Kiểm chứng (dữ liệu giả lập: 40 clean × 5 biến thể = 200 file)

| Kiểm tra | Kết quả |
|---|---|
| Bỏ các đoạn đã chèn thêm (pause, noise insert) và chia lại gain, rồi so với audio sạch ngoài vùng có nhãn | max \|Δ\| ≤ 3e-5, tức chỉ là sai số làm tròn PCM16 ✅ Nhãn nằm đúng chỗ tín hiệu bị thay đổi, việc dời nhãn cũng đúng |
| Noise không có nhãn trong vùng trộn | Luôn nhỏ hơn mức giọng − 30 dB ✅ |
| Noise | 95 file: overlay 62, insert 33 (mẫu nhỏ; trên 600 file là 78 % / 22 %); 1 đoạn 49 file, 2 đoạn 46 file; độ phủ nhãn trung bình 21 %, tối đa 45 %; mỗi đoạn 0,3–1,5 s ✅ |
| Đoạn noise overlay bắt đầu tại khe âm tiết | 60/60 ✅ |
| Noise không nằm trong pause | 33 file có cả hai, 0 vi phạm ✅ |
| Trọng số nhóm noise | Mưa + quạt chiếm đa số; giao thông + còi ít ✅ |
| Noise dạng giọng nói | SNR trong 10–20 dB, 100 % được làm xa; các nhóm khác trong 0–15 dB ✅ |
| Pause | 119 đoạn dừng trên 83 file, 1–2 đoạn/file (1 đoạn: 47 file, 2 đoạn: 36), mỗi đoạn 0,4–0,8 s, tổng dừng trung bình 0,86 s/file, chỉ chèn thêm (không mute); mức đoạn lấp ≤ −58,5 dB so với giọng (100 % lấy từ room tone của file) ✅ |
| Distortion | 72 file, 1 đoạn: 38, 2 đoạn: 34; độ phủ trung bình 21 %, tối đa 49 %; không file nào méo toàn bộ; bước nhảy âm lượng ở biên đoạn tối đa 0,76 dB ✅ |
| Mức giọng output (p5 / median / p95) | −29,0 / −23,5 / −17,8 dBFS; 0/200 file phải giảm gain ✅ |
| Mỗi file sạch có tối đa 1 bản giữ sạch khi gen nhiều biến thể | ✅ |
| Schema, interval trong [0, duration], 16 kHz mono int16, không NaN | ✅ |
| Chạy lại cùng seed | Giống hệt từng byte ✅ |
| Audit phát hiện file stereo/44,1 kHz, bị clip, quá ngắn, noise dạng giọng nói | ✅ |
| Engine và test lệch phiên bản | Test dừng ngay với thông báo "Version mismatch" ✅ |

---

## 7. Kết quả trên dữ liệu thật

Lần chạy chốt: engine `vi_aug_engine_v2.7`, `--seed 2026`, 09/10/2026. Số liệu lấy từ `data/vietnamese_aug/generation_report.md`.

| Chỉ số | Giá trị |
|---|---|
| Đầu vào | 1.000 file clean (không file nào bị loại); kho noise 54 file sau khi loại 6 file `street_vendor` |
| Số file output | **1.000** (`--variants-per-clean 1`, mỗi audio sạch 1 bản) |
| Duration min / median / max (s) | 2,16 / 4,24 / 12,58 |
| Có lỗi noise / distortion / pause | 456 (45,6 %) / 406 (40,6 %) / 377 (37,7 %) |
| Không lỗi (recipe clean) | 142 (14,2 %) |
| Recipe | noise 223, distortion 169, clean 142, pause 133, noise+pause 96, noise+distortion 89, distortion+pause 81, cả 3 lỗi 67 |
| Nhóm noise | fan 187 (41 %), loudspeaker 81 (18 %), restaurant_babble 80 (18 %), motorbike 36 (8 %), traffic 25 (5 %), rain_tin 24 (5 %), rain_talk 23 (5 %) |
| Noise: cách đưa vào | overlay 333 (73 %) / insert 123 (27 %) |
| Noise: số đoạn / độ phủ | 1 đoạn: 296 file, 2 đoạn: 160 file; nhãn phủ trung bình 23 % file, tối đa 50 % |
| SNR (dB) | min 0,1 / median 9,5 / max 19,9. Histogram: 0–3: 63 · 3–6: 77 · 6–9: 82 · 9–12: 94 · 12–15: 100 · 15–18: 23 · 18–21: 17 (các mức > 15 dB là loa phường) |
| Distortion: loại | narrowband 146 (36 %), clipping 117 (29 %), codec_lowbitrate 89 (22 %), packet_loss 54 (13 %) |
| Distortion: số đoạn | 1 đoạn: 246 file, 2 đoạn: 160 file; không file nào méo toàn bộ |
| Pause | 1 đoạn: 191 file, 2 đoạn: 186 file; tổng thời gian dừng trung bình 0,90 s/file, tối đa 1,57 s |
| Mức giọng p5 / median / p95 (dBFS) | −28,6 / −23,3 / −17,8; đỉnh tối đa −0,09 dBFS; 34 file được giảm gain để không vượt đỉnh (không tạo clipping) 


## 8. Bàn giao

- **Hiếu (chỉ MOS):** nhận `labels.jsonl` (đã có interval lỗi chính xác, lấy đúng vị trí engine chèn) và `params.jsonl`. Trong `params.jsonl`, mỗi file có danh sách lỗi đã chèn, cùng SNR, loại distortion và độ dài pause. Hiếu dùng các thông tin này để tính `quality_score` theo công thức mục 5 của guide. Gợi ý: noise lấy `snr_db` của sự kiện noise đầu tiên; pause lấy tổng thời gian các đoạn `unnatural_pause`.
