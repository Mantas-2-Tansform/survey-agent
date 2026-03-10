"""
NOISE CANCELLATION MODULE
==========================
Cleans incoming PCM audio before sending to Gemini Live.
Designed for: PCM 16-bit, 16kHz, mono (your agent's format).

Three modes:
1. spectral_gate  — noisereduce library (pure Python, no model, ~15-30ms latency)
2. rnnoise        — Mozilla RNNoise via pyrnnoise (C lib, ~2-5ms, best quality)
3. webrtc_ns      — WebRTC noise suppression (C lib, ~2ms, lightweight)

Usage in agent.py:
    from src.utils.noise_cancel import NoiseCanceller
    self.noise_canceller = NoiseCanceller(method="spectral_gate")
    
    # In send_audio():
    cleaned = self.noise_canceller.process(audio_data)
"""

import numpy as np
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes per sample


class NoiseCanceller:
    """
    Wraps noise cancellation with a consistent interface.
    Falls back gracefully — if the library isn't installed or fails,
    it passes audio through unchanged.
    """

    def __init__(self, method: str = "spectral_gate", aggressiveness: int = 2):
        """
        Args:
            method: "spectral_gate" | "rnnoise" | "webrtc_ns" | "none"
            aggressiveness: 0-3 for webrtc_ns, ignored for others.
                            For spectral_gate, maps to noise reduction strength.
        """
        self.method = method
        self.aggressiveness = aggressiveness
        self._enabled = True
        self._backend = None
        self._noise_profile = None
        self._chunk_count = 0
        self._profile_buffer = bytearray()
        self._profile_ready = False
        self._total_latency_ms = 0.0
        self._process_count = 0

        self._init_backend()

    def _init_backend(self):
        """Attempt to initialize the chosen backend."""
        if self.method == "none":
            self._enabled = False
            logger.info("🔇 Noise cancellation disabled (method='none').")
            return

        if self.method == "spectral_gate":
            try:
                import noisereduce as nr
                self._backend = nr
                logger.info("🔇 Noise cancellation: spectral_gate (noisereduce) ready.")
            except ImportError:
                logger.warning("⚠️ noisereduce not installed. pip install noisereduce. Passing audio through.")
                self._enabled = False

        elif self.method == "rnnoise":
            try:
                import pyrnnoise
                self._backend = pyrnnoise
                self._denoiser = pyrnnoise.RNNoise()
                logger.info("🔇 Noise cancellation: rnnoise ready.")
            except ImportError:
                logger.warning("⚠️ pyrnnoise not installed. pip install pyrnnoise. Passing audio through.")
                self._enabled = False

        elif self.method == "webrtc_ns":
            try:
                from webrtc_noise_gain import AudioProcessor
                # webrtc_noise_gain expects 10ms frames at 16kHz = 160 samples
                self._backend = AudioProcessor(sample_rate=SAMPLE_RATE)
                logger.info("🔇 Noise cancellation: webrtc_ns ready.")
            except ImportError:
                logger.warning("⚠️ webrtc-noise-gain not installed. Passing audio through.")
                self._enabled = False
        else:
            logger.warning(f"⚠️ Unknown noise cancellation method: {self.method}. Disabled.")
            self._enabled = False

    # Minimum chunk size worth processing (100ms = 3200 bytes at 16kHz/16bit)
    # Below this, FFT overhead exceeds chunk duration — just pass through.
    MIN_PROCESS_BYTES = 3200

    def process(self, pcm_data: bytes) -> bytes:
        """
        Process a chunk of PCM 16-bit mono 16kHz audio.
        Returns cleaned audio as bytes, same format.
        
        Chunks smaller than 100ms are passed through unchanged (FFT cost > benefit).
        Your MIN_AUDIO_CHUNK_MS=200 means this rarely triggers.
        
        If noise cancellation fails for any reason, returns original audio.
        """
        if not self._enabled or not pcm_data:
            return pcm_data

        # Skip denoising for very small chunks (not worth the FFT overhead)
        if len(pcm_data) < self.MIN_PROCESS_BYTES:
            return pcm_data

        start = time.monotonic()

        try:
            if self.method == "spectral_gate":
                result = self._process_spectral_gate(pcm_data)
            elif self.method == "rnnoise":
                result = self._process_rnnoise(pcm_data)
            elif self.method == "webrtc_ns":
                result = self._process_webrtc(pcm_data)
            else:
                result = pcm_data
        except Exception as e:
            logger.warning(f"⚠️ Noise cancellation error ({self.method}): {e}. Passing through.")
            result = pcm_data

        elapsed_ms = (time.monotonic() - start) * 1000
        self._total_latency_ms += elapsed_ms
        self._process_count += 1

        # Log latency periodically (every 100 chunks ~ every 2 seconds at typical rates)
        if self._process_count % 100 == 0:
            avg = self._total_latency_ms / self._process_count
            logger.debug(f"🔇 Noise cancel avg latency: {avg:.1f}ms over {self._process_count} chunks")

        return result

    # =========================================================================
    # SPECTRAL GATING (noisereduce)
    # =========================================================================
    def _process_spectral_gate(self, pcm_data: bytes) -> bytes:
        """
        Uses noisereduce's spectral gating. 
        
        Strategy: 
        - First ~0.5s of audio is used to build a noise profile (assumes
          the call starts with a brief silence/ambient before greeting).
        - After that, uses the profile for stationary noise reduction.
        - If no profile yet, uses non-stationary mode (adaptive).
        """
        nr = self._backend
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Build noise profile from first ~0.5 seconds (8000 samples at 16kHz)
        if not self._profile_ready:
            self._profile_buffer.extend(pcm_data)
            if len(self._profile_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH:  # 0.5 seconds
                profile_np = np.frombuffer(bytes(self._profile_buffer[:SAMPLE_RATE * SAMPLE_WIDTH]),
                                           dtype=np.int16).astype(np.float32) / 32768.0
                self._noise_profile = profile_np
                self._profile_ready = True
                logger.info(f"🔇 Noise profile captured ({len(self._profile_buffer)} bytes)")

        # Map aggressiveness (0-3) to prop_decrease (0.5 - 1.0)
        strength = min(0.5 + (self.aggressiveness * 0.15), 1.0)

        if self._profile_ready and self._noise_profile is not None:
            # Stationary reduction using captured noise profile
            cleaned = nr.reduce_noise(
                y=audio_np,
                sr=SAMPLE_RATE,
                y_noise=self._noise_profile,
                prop_decrease=strength,
                stationary=True,
                n_fft=512,      # Smaller FFT for lower latency
                hop_length=128,
            )
        else:
            # Adaptive mode until we have a profile
            cleaned = nr.reduce_noise(
                y=audio_np,
                sr=SAMPLE_RATE,
                prop_decrease=strength,
                stationary=False,
                n_fft=512,
                hop_length=128,
            )

        # Back to int16 PCM bytes
        cleaned_int16 = np.clip(cleaned * 32768, -32768, 32767).astype(np.int16)
        return cleaned_int16.tobytes()

    # =========================================================================
    # RNNOISE (Mozilla)
    # =========================================================================
    def _process_rnnoise(self, pcm_data: bytes) -> bytes:
        """
        RNNoise processes fixed 480-sample frames (at 48kHz).
        For 16kHz input, we need to handle frame alignment.
        pyrnnoise handles resampling internally.
        """
        # pyrnnoise expects int16 numpy array
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        cleaned = self._denoiser.process_audio(audio_np, sample_rate=SAMPLE_RATE)
        return cleaned.tobytes()

    # =========================================================================
    # WEBRTC NOISE SUPPRESSION
    # =========================================================================
    def _process_webrtc(self, pcm_data: bytes) -> bytes:
        """
        WebRTC NS processes 10ms frames (160 samples at 16kHz).
        Very low latency, good for real-time.
        """
        FRAME_SIZE = 320  # 160 samples * 2 bytes = 10ms at 16kHz
        output = bytearray()

        for i in range(0, len(pcm_data), FRAME_SIZE):
            frame = pcm_data[i:i + FRAME_SIZE]
            if len(frame) < FRAME_SIZE:
                # Pad last frame if needed
                frame = frame + b'\x00' * (FRAME_SIZE - len(frame))
            
            cleaned_frame = self._backend.process(frame, ns_level=self.aggressiveness)
            output.extend(cleaned_frame)

        return bytes(output[:len(pcm_data)])  # Trim to original length

    # =========================================================================
    # UTILITIES
    # =========================================================================
    def get_stats(self) -> dict:
        """Returns processing statistics."""
        avg_latency = (self._total_latency_ms / self._process_count) if self._process_count > 0 else 0
        return {
            "method": self.method,
            "enabled": self._enabled,
            "chunks_processed": self._process_count,
            "avg_latency_ms": round(avg_latency, 2),
            "noise_profile_ready": self._profile_ready,
        }

    def reset_profile(self):
        """Reset the noise profile (e.g., for a new call)."""
        self._noise_profile = None
        self._profile_ready = False
        self._profile_buffer = bytearray()
        self._chunk_count = 0
        self._total_latency_ms = 0.0
        self._process_count = 0
        logger.info("🔇 Noise profile reset.")