from __future__ import annotations

import hashlib
import io
import subprocess
import wave

import numpy as np

from .config import (
    AudioAugmentationConfig,
    AudioAugmentationProfileConfig,
    ClippingAugmentationConfig,
    CodecAugmentationConfig,
    NoiseAugmentationConfig,
    PacketLossAugmentationConfig,
    ReverbAugmentationConfig,
)


def _seed_for_sample(base_seed: int, dataset_id: str, sample_id: str) -> int:
    payload = f"{base_seed}\0{dataset_id}\0{sample_id}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _encode_pcm_wav_bytes(waveform: np.ndarray, *, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _decode_pcm_wav_bytes(raw_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw_bytes), "rb") as fh:
        if fh.getnchannels() != 1:
            raise RuntimeError(f"Expected mono WAV from ffmpeg augmentation path, got {fh.getnchannels()} channels.")
        if fh.getsampwidth() != 2:
            raise RuntimeError(f"Expected 16-bit PCM WAV from ffmpeg augmentation path, got sample width={fh.getsampwidth()}.")
        frames = fh.readframes(fh.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return np.ascontiguousarray(pcm / 32768.0)


def _limit_peak(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak
    return np.ascontiguousarray(waveform.astype(np.float32, copy=False))


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    start, end = bounds
    if end <= start:
        return float(start)
    return float(rng.uniform(start, end))


def _sample_int_range(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    start, end = bounds
    if end <= start:
        return int(start)
    return int(rng.integers(start, end + 1))


class WaveformAugmenter:
    """Dataset-aware, deterministic waveform corruption applied before Whisper feature extraction."""

    def __init__(self, config: AudioAugmentationConfig) -> None:
        self.config = config
        self.ffmpeg_bin = config.ffmpeg_bin

    def _run_ffmpeg(self, input_bytes: bytes, args: list[str], sample_id: str) -> bytes:
        cmd = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin"] + list(args)
        try:
            proc = subprocess.run(
                cmd,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg binary not found for audio augmentation: {self.ffmpeg_bin!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg timed out while augmenting sample `{sample_id}`.") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed while augmenting sample `{sample_id}` with exit code {proc.returncode}: {stderr}"
            )
        return proc.stdout

    def _codec_roundtrip(
        self,
        waveform: np.ndarray,
        *,
        mode: str,
        cfg: CodecAugmentationConfig,
        sample_id: str,
        sample_rate: int,
    ) -> np.ndarray:
        input_wav = _encode_pcm_wav_bytes(waveform, sample_rate=sample_rate)
        phone_band_filters = f"highpass=f={cfg.highpass_hz:g},lowpass=f={cfg.lowpass_hz:g}"

        if mode == "mulaw_narrowband":
            encoded = self._run_ffmpeg(
                input_wav,
                [
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(cfg.sample_rate),
                    "-af",
                    phone_band_filters,
                    "-c:a",
                    "pcm_mulaw",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                sample_id,
            )
            decoded = self._run_ffmpeg(
                encoded,
                [
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                sample_id,
            )
            return _decode_pcm_wav_bytes(decoded)

        if mode == "gsm_narrowband":
            encoded = self._run_ffmpeg(
                input_wav,
                [
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(cfg.sample_rate),
                    "-af",
                    phone_band_filters,
                    "-c:a",
                    "libgsm",
                    "-f",
                    "gsm",
                    "pipe:1",
                ],
                sample_id,
            )
            decoded = self._run_ffmpeg(
                encoded,
                [
                    "-f",
                    "gsm",
                    "-ar",
                    str(cfg.sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                sample_id,
            )
            return _decode_pcm_wav_bytes(decoded)

        raise RuntimeError(f"Unsupported codec mode requested for audio augmentation: {mode!r}")

    def _apply_codec(
        self,
        waveform: np.ndarray,
        *,
        rng: np.random.Generator,
        cfg: CodecAugmentationConfig,
        sample_id: str,
        sample_rate: int,
    ) -> np.ndarray:
        out = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        mode_names = list(cfg.modes)
        weights = np.asarray([cfg.modes[name] for name in mode_names], dtype=np.float64)
        weights = weights / weights.sum()
        mode = mode_names[int(rng.choice(len(mode_names), p=weights))]
        for _ in range(cfg.roundtrips):
            out = _limit_peak(
                self._codec_roundtrip(out, mode=mode, cfg=cfg, sample_id=sample_id, sample_rate=sample_rate)
            )
        return out

    @staticmethod
    def _apply_packet_loss(
        waveform: np.ndarray,
        *,
        rng: np.random.Generator,
        cfg: PacketLossAugmentationConfig,
        sample_rate: int,
    ) -> np.ndarray:
        out = waveform.copy()
        burst_count = _sample_int_range(rng, cfg.bursts)
        for _ in range(burst_count):
            burst_ms = _sample_int_range(rng, cfg.burst_ms)
            burst_len = max(1, int(round((burst_ms / 1000.0) * sample_rate)))
            if burst_len >= out.shape[0]:
                start = 0
                end = out.shape[0]
            else:
                start = int(rng.integers(0, out.shape[0] - burst_len + 1))
                end = start + burst_len
            if cfg.fill == "hold" and start > 0:
                out[start:end] = out[start - 1]
            else:
                out[start:end] = 0.0
        return out

    @staticmethod
    def _apply_clipping(
        waveform: np.ndarray,
        *,
        rng: np.random.Generator,
        cfg: ClippingAugmentationConfig,
    ) -> np.ndarray:
        threshold = _sample_range(rng, cfg.threshold)
        return np.clip(waveform, -threshold, threshold).astype(np.float32, copy=False)

    @staticmethod
    def _apply_noise(
        waveform: np.ndarray,
        *,
        rng: np.random.Generator,
        cfg: NoiseAugmentationConfig,
    ) -> np.ndarray:
        snr_db = _sample_range(rng, cfg.snr_db)
        signal_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
        signal_rms = max(signal_rms, 1e-4)
        noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = rng.normal(loc=0.0, scale=noise_rms, size=waveform.shape).astype(np.float32)
        return waveform + noise

    @staticmethod
    def _apply_reverb(
        waveform: np.ndarray,
        *,
        rng: np.random.Generator,
        cfg: ReverbAugmentationConfig,
        sample_rate: int,
    ) -> np.ndarray:
        rt60_s = _sample_range(rng, cfg.rt60_ms) / 1000.0
        rir_len = max(64, int(round(min(0.6, max(0.08, rt60_s * 1.5)) * sample_rate)))
        time_axis = np.arange(rir_len, dtype=np.float32) / float(sample_rate)
        decay = np.exp(-6.907755 * time_axis / max(rt60_s, 1e-3)).astype(np.float32)
        rir = rng.normal(loc=0.0, scale=1.0, size=rir_len).astype(np.float32) * decay
        rir[0] += 1.0
        for _ in range(int(rng.integers(3, 8))):
            position = int(rng.integers(1, min(rir_len, max(2, int(0.05 * sample_rate)))))
            rir[position] += float(rng.uniform(0.1, 0.6))
        normalization = float(np.sum(np.abs(rir), dtype=np.float64))
        if normalization > 0.0:
            rir = rir / normalization
        wet = np.convolve(waveform, rir, mode="full")[: waveform.shape[0]].astype(np.float32, copy=False)
        wet_mix = 0.25
        return ((1.0 - wet_mix) * waveform + wet_mix * wet).astype(np.float32, copy=False)

    @staticmethod
    def _clean_waveform(waveform: np.ndarray) -> np.ndarray:
        cleaned = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return np.ascontiguousarray(cleaned)

    def _find_policy(self, dataset_name: str, dataset_repo_id: str | None):
        for key in (dataset_name, dataset_repo_id, "*"):
            if not key:
                continue
            policy = self.config.datasets.get(key)
            if policy is not None:
                return policy, str(key)
        return None, dataset_name or (dataset_repo_id or "")

    def _maybe_augment_with_profile(
        self,
        waveform: np.ndarray,
        *,
        profile: AudioAugmentationProfileConfig,
        rng: np.random.Generator,
        sample_id: str,
        sample_rate: int,
    ) -> np.ndarray:
        out = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        if profile.reverb is not None and float(rng.uniform()) < profile.reverb.p:
            out = _limit_peak(self._apply_reverb(out, rng=rng, cfg=profile.reverb, sample_rate=sample_rate))
        if profile.noise is not None and float(rng.uniform()) < profile.noise.p:
            out = _limit_peak(self._apply_noise(out, rng=rng, cfg=profile.noise))
        if profile.clipping is not None and float(rng.uniform()) < profile.clipping.p:
            out = self._apply_clipping(out, rng=rng, cfg=profile.clipping)
        if profile.codec is not None and float(rng.uniform()) < profile.codec.p:
            out = _limit_peak(
                self._apply_codec(out, rng=rng, cfg=profile.codec, sample_id=sample_id, sample_rate=sample_rate)
            )
        if profile.packet_loss is not None and float(rng.uniform()) < profile.packet_loss.p:
            out = self._apply_packet_loss(out, rng=rng, cfg=profile.packet_loss, sample_rate=sample_rate)

        out = self._clean_waveform(out)
        if out.shape[0] > waveform.shape[0]:
            out = out[: waveform.shape[0]]
        elif out.shape[0] < waveform.shape[0]:
            out = np.pad(out, (0, waveform.shape[0] - out.shape[0]))
        return np.ascontiguousarray(out, dtype=np.float32)

    def maybe_augment(
        self,
        waveform: np.ndarray,
        *,
        sample_id: str,
        dataset_name: str,
        dataset_repo_id: str | None,
        dataset_split: str,
        sample_rate: int,
    ) -> np.ndarray:
        clean = self._clean_waveform(np.asarray(waveform, dtype=np.float32))
        if not self.config.enabled:
            return clean

        policy, policy_key = self._find_policy(dataset_name, dataset_repo_id)
        if policy is None or not policy.enabled:
            return clean

        normalized_split = str(dataset_split or "train").strip().lower() or "train"
        if self.config.train_only and normalized_split != "train":
            return clean
        if normalized_split not in {split.lower() for split in policy.splits}:
            return clean

        rng = np.random.default_rng(_seed_for_sample(self.config.seed, policy_key, sample_id))
        if float(rng.uniform()) >= policy.apply_p:
            return clean

        profile = self.config.profiles[policy.profile]
        return self._maybe_augment_with_profile(
            clean,
            profile=profile,
            rng=rng,
            sample_id=sample_id,
            sample_rate=sample_rate,
        )


def build_waveform_augmenter(config: AudioAugmentationConfig) -> WaveformAugmenter | None:
    if not config.enabled:
        return None
    return WaveformAugmenter(config)
