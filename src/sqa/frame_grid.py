from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFrameGrid:

    frame_hop_sec: float = 0.02
    offset_sec: float = 0.0
    frame_duration_sec: float = 0.02
    # Tự khai báo rõ quy ước & nguồn gốc con số, để không ai nhầm "0.02 mặc định"
    # là giá trị đã verify.
    convention: str = "frame_start"
    source: str = "default_unverified"  # đổi thành "empirically_verified" sau khi
    # chạy xong test alignment (mục 4.1 kế hoạch) và gán lại các giá trị đo được.

    def frame_output_slot(self, i: int) -> tuple[float, float]:
        s = i * self.frame_hop_sec + self.offset_sec
        return s, s + self.frame_hop_sec

    def frame_receptive_field(self, i: int) -> tuple[float, float]:
        s = i * self.frame_hop_sec + self.offset_sec
        return s, s + self.frame_duration_sec

    def n_frames(self, duration_sec: float) -> int:
        import math

        return max(0, math.ceil((duration_sec - self.offset_sec) / self.frame_hop_sec))


@dataclass(frozen=True)
class EvaluatorGrid:
    frame_hop_sec: float = 0.02
    offset_sec: float = 0.0


MODEL_GRID = ModelFrameGrid(
    frame_hop_sec=0.020052601,
    offset_sec=0.015194489,
    frame_duration_sec=0.020052601,
    source="empirically_verified",
)

EVALUATOR_GRID = EvaluatorGrid()
