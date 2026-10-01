from __future__ import annotations

import io
import wave
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class QualiSpeechDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        zip_path: str | Path,
        quality_column: str = "Overall quality",
        indices: List[int] | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.zip_path = Path(zip_path)
        self.quality_column = quality_column

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP not found: {self.zip_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"id", quality_column}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {self.csv_path}: {sorted(missing_columns)}"
            )

        # Open ZIP once in the main process.
        # For DataLoader(num_workers=0), this is ideal.
        self._zip = zipfile.ZipFile(self.zip_path, "r")

        # Build basename -> ZIP member mapping.
        #
        # The ZIP contains paths such as:
        #   wav/test/sys01tts-0000068.wav
        #   wav/val/sysnisvs-0000825.wav
        #
        # CSV only contains:
        #   sys01tts-0000068.wav
        self._zip_members: Dict[str, str] = {}

        for member in self._zip.namelist():
            if not member.lower().endswith(".wav"):
                continue

            basename = Path(member).name

            if basename in self._zip_members:
                raise ValueError(
                    f"Duplicate WAV basename in ZIP: {basename}\n"
                    f"Existing: {self._zip_members[basename]}\n"
                    f"New:      {member}"
                )

            self._zip_members[basename] = member

        # Keep only rows whose audio actually exists in this ZIP.
        available_mask = self.df["id"].isin(self._zip_members.keys())
        available_df = self.df.loc[available_mask].copy()

        if available_df.empty:
            raise RuntimeError(
                "No CSV rows have matching audio in the ZIP."
            )

        available_df = available_df.reset_index(drop=True)

        if indices is not None:
            if len(indices) == 0:
                raise ValueError("indices must not be empty.")

            if min(indices) < 0 or max(indices) >= len(available_df):
                raise IndexError(
                    f"indices must be in [0, {len(available_df) - 1}]"
                )

            available_df = available_df.iloc[indices].reset_index(drop=True)

        self.df = available_df

        # Statistics useful for diagnostics.
        self.total_csv_rows = len(pd.read_csv(self.csv_path))
        self.available_rows = len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _decode_wav(self, wav_bytes: bytes) -> Tuple[torch.Tensor, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            sample_rate = wav.getframerate()
            num_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            num_frames = wav.getnframes()
            raw = wav.readframes(num_frames)

        if sample_width == 2:
            # PCM signed int16.
            waveform = torch.frombuffer(
                bytearray(raw),
                dtype=torch.int16,
            ).clone().float()

            waveform /= 32768.0

        elif sample_width == 1:
            # Unsigned 8-bit PCM.
            waveform = torch.frombuffer(
                bytearray(raw),
                dtype=torch.uint8,
            ).clone().float()

            waveform = (waveform - 128.0) / 128.0

        elif sample_width == 4:
            # Signed int32 PCM.
            waveform = torch.frombuffer(
                bytearray(raw),
                dtype=torch.int32,
            ).clone().float()

            waveform /= 2147483648.0

        else:
            raise ValueError(
                f"Unsupported WAV sample width: "
                f"{sample_width} bytes"
            )

        # [T * channels] -> [T, channels]
        if num_channels > 1:
            waveform = waveform.view(-1, num_channels)

            # Downmix to mono.
            waveform = waveform.mean(dim=1)

        # Safety normalization.
        waveform = waveform.clamp(-1.0, 1.0)

        return waveform, sample_rate

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        file_id = str(row["id"])
        target = float(row[self.quality_column])

        if file_id not in self._zip_members:
            raise KeyError(
                f"Audio {file_id!r} is not present in ZIP."
            )

        member = self._zip_members[file_id]

        wav_bytes = self._zip.read(member)
        waveform, sample_rate = self._decode_wav(wav_bytes)

        # QualiSpeech archive inspected above is 16 kHz mono.
        # Keep this as an explicit invariant rather than silently
        # resampling inside Dataset.
        if sample_rate != 16000:
            raise ValueError(
                f"Unexpected sample rate for {file_id}: "
                f"{sample_rate} Hz; expected 16000 Hz."
            )

        if waveform.numel() == 0:
            raise ValueError(f"Empty audio: {file_id}")

        return {
            "wav": waveform,
            "target": torch.tensor(target, dtype=torch.float32),
            "file_id": file_id,
            "sample_rate": sample_rate,
        }

    def close(self):
        """Close the underlying ZIP file."""
        if getattr(self, "_zip", None) is not None:
            self._zip.close()
            self._zip = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def qualispeech_collate(batch):
    if not batch:
        raise ValueError("Empty batch.")

    waveforms = [item["wav"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])

    lengths = torch.tensor(
        [wav.numel() for wav in waveforms],
        dtype=torch.long,
    )

    max_length = int(lengths.max().item())

    padded = torch.zeros(
        len(waveforms),
        max_length,
        dtype=torch.float32,
    )

    attention_mask = torch.zeros(
        len(waveforms),
        max_length,
        dtype=torch.bool,
    )

    for i, wav in enumerate(waveforms):
        length = wav.numel()

        padded[i, :length] = wav
        attention_mask[i, :length] = True

    return {
        "wav": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "target": targets,
        "file_id": [item["file_id"] for item in batch],
    }