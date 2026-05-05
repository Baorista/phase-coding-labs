#!/usr/bin/env python3
"""Thư viện encoder/decoder chuẩn — sinh viên import để dùng."""
import numpy as np
import scipy.io.wavfile as wavfile
import time

def text_to_bits(msg):
    bits = []
    for c in msg:
        for b in range(7, -1, -1):
            bits.append((ord(c) >> b) & 1)
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        val = 0
        for bit in bits[i:i+8]:
            val = (val << 1) | bit
        chars.append(chr(val))
    return "".join(chars)

def encode(wav_path, msg, seg_len=4096, start_bin=256):
    """Encode message vào WAV. Trả về dict kết quả."""
    t0 = time.time()
    fs, raw = wavfile.read(wav_path)
    audio = raw.astype(np.float64)
    if raw.dtype in (np.float32, np.float64):
        audio *= 32767.0

    bits = text_to_bits(msg)
    msg_len = len(bits)

    if start_bin + msg_len >= seg_len // 2:
        return {"error": "msg_too_long", "encode_ms": 0}

    num_segs = len(audio) // seg_len
    if num_segs < 2:
        return {"error": "audio_too_short", "encode_ms": 0}

    seg0 = audio[:seg_len]
    fft0 = np.fft.fft(seg0)
    amps0 = np.abs(fft0)
    phases0 = np.angle(fft0)

    for i in range(msg_len):
        idx = start_bin + i
        phases0[idx] = np.pi/2 if bits[i] == 0 else -np.pi/2
        phases0[seg_len - idx] = -phases0[idx]

    output = np.copy(audio)
    output[:seg_len] = np.real(np.fft.ifft(amps0 * np.exp(1j * phases0)))

    diff = output - audio
    sig_pow = np.sum(audio**2)
    noise_pow = np.sum(diff**2)
    snr = 10*np.log10(sig_pow/noise_pow) if noise_pow > 0 else 999

    encode_ms = (time.time() - t0) * 1000
    spec_mod = msg_len / (seg_len // 2) * 100
    capacity = (seg_len // 2 - start_bin - 1) // 8

    return {
        "error": None, "output": output, "fs": fs,
        "orig_dtype": raw.dtype, "snr": snr,
        "encode_ms": encode_ms, "spec_mod": spec_mod,
        "capacity_chars": capacity, "bits": bits
    }

def decode(wav_path, num_chars, seg_len=4096, start_bin=256):
    """Decode message từ WAV stego."""
    fs, raw = wavfile.read(wav_path)
    audio = raw.astype(np.float64)
    if raw.dtype in (np.float32, np.float64):
        audio *= 32767.0

    if len(audio) < seg_len:
        return {"error": "audio_too_short"}

    fft0 = np.fft.fft(audio[:seg_len])
    phase0 = np.angle(fft0)
    msg_len = num_chars * 8

    ext_bits = []
    confidences = []
    for i in range(msg_len):
        idx = start_bin + i
        ph = phase0[idx]
        d0 = abs(ph - np.pi/2)
        d1 = abs(ph + np.pi/2)
        ext_bits.append(0 if d0 < d1 else 1)
        conf = abs(d0 - d1) / (d0 + d1) if (d0 + d1) > 0 else 0
        confidences.append(conf)

    decoded = bits_to_text(ext_bits)
    avg_conf = np.mean(confidences)

    return {
        "error": None, "decoded": decoded,
        "avg_conf": avg_conf, "bits": ext_bits
    }

def save_wav(output, fs, path, orig_dtype):
    """Lưu WAV đúng format gốc."""
    if orig_dtype in (np.float32, np.float64):
        wavfile.write(path, fs, (output / 32767.0).astype(np.float32))
    else:
        wavfile.write(path, fs, np.int16(np.clip(output, -32768, 32767)))
