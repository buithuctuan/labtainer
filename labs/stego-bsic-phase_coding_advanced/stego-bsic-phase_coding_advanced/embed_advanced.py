import numpy as np
from scipy.io import wavfile

def get_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def get_bits_per_bin(mag_db):
    if mag_db >= -5:
        return 3
    elif mag_db >= -10:
        return 2
    elif mag_db >= -15:
        return 1
    else:
        return 0

def phase_enc_djebbar(signal, text, fs, L=512, FHDmin=16, FHDmax=112):
    if signal.ndim > 1:
        plain = signal[:, 0]
    else:
        plain = signal
    text_bits = get_bits(text)
    pad_len = 3 - (len(text_bits) % 3) if (len(text_bits) % 3 != 0) else 0
    text_bits += '0' * pad_len
    bit_idx = 0
    max_bits = len(text_bits)
    I = len(plain)
    N = I // L
    s = plain[:N * L].reshape(N, L).T
    new_s = np.zeros_like(s)
    for frame_idx in range(N):
        frame = s[:, frame_idx]
        spectrum = np.fft.fft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        mag_db = 10 * np.log10(magnitude + 1e-12)
        for k in range(FHDmin, FHDmax):
            bits_this_bin = get_bits_per_bin(mag_db[k])
            if bits_this_bin == 0:
                continue
            if bit_idx + bits_this_bin > max_bits:
                break
            bits = text_bits[bit_idx : bit_idx + bits_this_bin]
            val = int(bits, 2)
            max_val = 2 ** bits_this_bin
            phase_val = -np.pi + (2 * np.pi * (val + 0.5)) / max_val
            print(f"[ENCODE] Frame {frame_idx}, Bin {k}, Bits: {bits}, Val: {val}, Phase: {phase_val:.4f}, mag_db: {mag_db[k]:.2f}dB")
            phase[k] = phase_val
            phase[-k] = -phase_val
            bit_idx += bits_this_bin
        new_spectrum = magnitude * np.exp(1j * phase)
        new_frame = np.fft.ifft(new_spectrum)
        new_s[:, frame_idx] = np.real(new_frame)
    print(f"[SUMMARY] Total bits encoded: {bit_idx}")
    if bit_idx < max_bits:
        print(f"⚠ Chỉ giấu được {bit_idx} bits trên tổng {max_bits} bits")
    snew = new_s.T.reshape(N * L)
    out = np.concatenate([snew, plain[N * L:]])
    # === Ghi phase trước và sau frame đầu tiên để so sánh trực quan
    np.save("clsb_phase_before.npy", np.angle(np.fft.fft(s[:, 0])))
    np.save("clsb_phase_after.npy", np.angle(np.fft.fft(new_s[:, 0])))

    return out

def embed_audio_djebbar(audio_path, text_path, output_path):
    fs, audio_data = wavfile.read(audio_path)
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    out = phase_enc_djebbar(signal=audio_data, text=text, fs=fs)
    orig_max = np.max(np.abs(audio_data))
    stego_max = np.max(np.abs(out))
    if stego_max > 0:
        out = out * (orig_max / stego_max)
    wavfile.write(output_path, fs, out.astype(np.int16))
    print(f"\n✅ Stego signal saved to: {output_path}")
    return text

if __name__ == "__main__":
    input_path = "input_audio.wav"
    text_path = "text.txt"
    output_path = "output_stego_djebbar.wav"

    embed_audio_djebbar(input_path, text_path, output_path)

