import logging
import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 5
_CONFIDENCE_THRESHOLD = 0.35
_MIN_AUDIO_SECONDS = 0.3
_SAMPLE_RATE = 16_000
_SUPPORTED_LANGS = frozenset({"pt", "en", "auto"})


class Transcriber:
    _DEFAULT_MODEL = "base"

    def __init__(self, language: str = "pt") -> None:
        self._lang: str = language if language in _SUPPORTED_LANGS else "pt"
        self._model = None
        self._ctx: list[str] = []

    @property
    def language(self) -> str:
        return self._lang

    @language.setter
    def language(self, value: str) -> None:
        if value in _SUPPORTED_LANGS:
            self._lang = value
            self.clear_context()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_size: str = _DEFAULT_MODEL) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=2,
            cpu_threads=4,
        )
        logger.info("Whisper model '%s' loaded", model_size)

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        min_samples = int(_MIN_AUDIO_SECONDS * _SAMPLE_RATE)
        if len(audio) < min_samples:
            return "", 0.0, self._lang

        prompt = " ".join(self._ctx[-_CONTEXT_WINDOW:]) if self._ctx else None
        lang = self._lang if self._lang != "auto" else None

        segments, info = self._model.transcribe(
            audio,
            language=lang,
            initial_prompt=prompt,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 200,
            },
        )

        parts: list[str] = []
        log_probs: list[float] = []

        for seg in segments:
            if seg.no_speech_prob < 0.6 and seg.text.strip():
                parts.append(seg.text.strip())
                log_probs.append(seg.avg_logprob)

        text = " ".join(parts).strip()
        confidence = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
        detected = getattr(info, "language", self._lang) or self._lang

        if text and confidence >= _CONFIDENCE_THRESHOLD:
            self._ctx.append(text)

        return text, confidence, detected

    def clear_context(self) -> None:
        self._ctx.clear()


class TranscriptionWorker:
    def __init__(
        self,
        transcriber: Transcriber,
        on_result: Callable[[str, float, str], None],
        on_error: Callable[[str], None],
        on_idle_change: Callable[[bool], None],
    ) -> None:
        self._transcriber = transcriber
        self._on_result = on_result
        self._on_error = on_error
        self._on_idle_change = on_idle_change
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="transcriber",
        )
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def submit(self, audio: np.ndarray) -> None:
        self._q.put(audio)

    def stop(self) -> None:
        self._running = False
        self._q.put(None)
        self._thread.join(timeout=6.0)

    def _run(self) -> None:
        while self._running:
            item = self._q.get()
            if item is None:
                break
            self._on_idle_change(False)
            try:
                text, conf, lang = self._transcriber.transcribe(item)
                if text:
                    self._on_result(text, conf, lang)
            except Exception as exc:
                logger.exception("Transcription error")
                self._on_error(str(exc))
            finally:
                self._on_idle_change(True)