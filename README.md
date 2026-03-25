# Features

- Real-time speech-to-text via OpenAI Whisper ***(faster-whisper, CPU · int8 fully offline after first download)***.
- Brazilian Portuguese and English transcription.
- WebRTC VAD (primary) + energy-based fallback.
- Context-aware transcription (Previous sentences used as prompt for better accuracy).
- Compatible with Windows 10/11 and ***Linux (GTK3)***.
- All background threads are daemon threads — no process leaks on abnormal exit.
- Clipboard access uses ***wx.TheClipboard (OS-level, no third-party service)***.

![Front01](https://github.com/user-attachments/assets/e4ff53cf-ff43-43a4-9d90-c2f69ac8177a)

![Front02](https://github.com/user-attachments/assets/3081c5d2-120a-467b-a62d-3b60adcbbe98)


# Requirements

        pip install -r requirements.txt

- The Whisper base model (~150 MB) is downloaded from Hugging Face and cached locally, all subsequent runs are fully offline.

# Required for wxPython on Linux (GTK3)

        sudo apt install libgtk-3-dev
        
# Ubuntu/Debian pre-built wheel:

        pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 wxPython

# Running

        python main.py

# AI / Audio Pipeline


            Microphone  (sounddevice, 16 kHz, 30 ms frames)
        │
        ▼
    WebRTC VAD  ──► energy-based fallback if unavailable
        │  complete speech utterances only
        ▼
    TranscriptionWorker queue  (background thread, thread-safe)
        │
        ▼
    faster-whisper  (CPU · int8)
        ├─ Silero VAD filter        (removes residual silence)
        ├─ Context prompt           (rolling 5-sentence window → accuracy boost)
        ├─ Beam search (beam=5)     + temperature fallback
        └─ Confidence gate (≥0.35)  (discards low-quality results)
        │
        ▼
    wx.CallAfter  ──► main thread ──► TextCtrl.AppendText()

# Disclaimer

- This tool is under development and may contain bugs.
