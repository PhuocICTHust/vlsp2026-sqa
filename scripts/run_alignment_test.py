from __future__ import annotations

import numpy as np
import torch

from sqa.model import WavLMBackbone

SR = 16_000


def make_transient(duration_sec: float, transient_at: float, sr: int = SR) -> np.ndarray:
    """Audio gần như im lặng, có 1 xung ngắn (transient) rõ ràng tại
    transient_at giây — dùng để dò xem frame nào của backbone phản ứng mạnh
    nhất, và vùng activation trải rộng bao nhiêu frame quanh đó."""
    x = np.zeros(int(duration_sec * sr), dtype=np.float32)
    burst_len = int(0.01 * sr)  # xung 10ms
    start = int(transient_at * sr)
    end = min(len(x), start + burst_len)
    rng = np.random.default_rng(0)
    x[start:end] = 0.8 * rng.standard_normal(end - start).astype(np.float32)
    return x


def run_case(backbone: WavLMBackbone, duration_sec: float, transient_at: float) -> dict:
    x = make_transient(duration_sec, transient_at)
    wav = torch.from_numpy(x).unsqueeze(0)  # [1, num_samples]

    # Baseline: cùng audio nhưng KHÔNG có transient.
    # So sánh hidden states để tìm frame nào thực sự thay đổi
    # do transient gây ra.
    x_base = np.zeros_like(x)
    wav_base = torch.from_numpy(x_base).unsqueeze(0)

    with torch.no_grad():
        h = backbone(wav)       # [1, T, H]
        h_base = backbone(wav_base)

    h = h.squeeze(0)
    h_base = h_base.squeeze(0)

    # Difference caused by transient.
    # Dùng L2 distance giữa hidden state có/không có transient.
    activation = (h - h_base).norm(dim=-1).numpy()

    t_frames = len(activation)

    # Chỉ dùng duration/T để ước lượng frame spacing.
    hop_est = duration_sec / t_frames

    # Frame phản ứng mạnh nhất với transient.
    peak_idx = int(activation.argmax())
    predicted_center = (peak_idx + 0.5) * hop_est

    # Sai số giữa tâm frame dự đoán và vị trí transient.
    error_ms = abs(predicted_center - transient_at) * 1000.0

    # Activation span:
    # tìm vùng có response đáng kể so với peak.
    peak = float(activation.max())

    if peak > 0:
        threshold = peak * 0.1
        active_indices = np.flatnonzero(activation >= threshold)

        if len(active_indices) > 0:
            first_idx = int(active_indices[0])
            last_idx = int(active_indices[-1])

            activation_span_frames = last_idx - first_idx + 1
        else:
            activation_span_frames = 0
    else:
        activation_span_frames = 0

    return {
        "duration_sec": duration_sec,
        "transient_at": transient_at,
        "T": t_frames,
        "hop_estimate_via_duration_over_T": round(hop_est, 6),
        "peak_frame_idx": peak_idx,
        "predicted_center_s_using_hop_estimate": round(predicted_center, 4),
        "error_ms": round(error_ms, 2),
        "activation_span_frames": activation_span_frames,
    }


def main():
    backbone = WavLMBackbone(freeze=True)
    backbone.eval()

    cases = []
    for duration in (1.0, 2.0, 3.0):
        for pos_ratio in (0.1, 0.5, 0.9):
            transient_at = round(duration * pos_ratio, 3)
            cases.append(run_case(backbone, duration, transient_at))

    header = [
        "duration_sec",
        "transient_at",
        "T",
        "hop_estimate_via_duration_over_T",
        "peak_frame_idx",
        "predicted_center_s_using_hop_estimate",
        "error_ms",
        "activation_span_frames",
    ]
    print("\t".join(header))
    for c in cases:
        print("\t".join(str(c[k]) for k in header))

    hops = [c["hop_estimate_via_duration_over_T"] for c in cases]
    print(f"\nhop_estimate: min={min(hops):.6f} max={max(hops):.6f} "
          f"(nếu 2 giá trị này lệch nhau đáng kể -> backbone KHÔNG có hop cố định, "
          f"cần điều tra thêm trước khi gán MODEL_GRID)")
    print(f"error_ms trung bình: {sum(c['error_ms'] for c in cases) / len(cases):.2f} ms")
    mean_error = sum(c["error_ms"] for c in cases) / len(cases)

    print(f"\nerror_ms trung bình: {mean_error:.2f} ms")
    # Fit time(frame) = offset + frame_idx * hop
    frame_idx = np.array([c["peak_frame_idx"] for c in cases], dtype=np.float64)
    target_time = np.array([c["transient_at"] for c in cases], dtype=np.float64)

    A = np.column_stack([np.ones_like(frame_idx), frame_idx])
    offset_fit, hop_fit = np.linalg.lstsq(A, target_time, rcond=None)[0]

    pred_time = offset_fit + hop_fit * frame_idx
    fit_error_ms = np.abs(pred_time - target_time) * 1000.0

    print("\nLinear fit:")
    print(f"  frame_hop_sec = {hop_fit:.9f}")
    print(f"  offset_sec    = {offset_fit:.9f}")
    print(f"  mean_fit_error_ms = {fit_error_ms.mean():.3f}")
    print(f"  max_fit_error_ms  = {fit_error_ms.max():.3f}")

    print(
        "\nKết luận:\n"
        "- hop_estimate chỉ dùng để kiểm tra frame spacing.\n"
        "- Không dùng peak activation để suy ra receptive field tuyệt đối.\n"
        "- Không cập nhật MODEL_GRID nếu error_ms còn lớn hoặc offset chưa ổn định.\n"
        "- frame_duration_sec cần một phép đo receptive field riêng, không lấy activation_span_frames của test này làm ground truth."
    )


if __name__ == "__main__":
    main()
