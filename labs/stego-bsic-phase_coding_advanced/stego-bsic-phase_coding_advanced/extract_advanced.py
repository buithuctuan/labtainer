import numpy as np
from scipy.io import wavfile

def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

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

def phase_to_val(phase, bits_per_bin, tolerance=0.05):
    phase = np.round(phase, decimals=4)
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    max_val = 2 ** bits_per_bin
    region_width = (2 * np.pi) / max_val
    val = int(((phase + np.pi + tolerance) // region_width) % max_val)
    val = max(0, min(val, max_val - 1))
    return val, format(val, f'0{bits_per_bin}b')

def phase_dec_djebbar(signal, text_length, fs, L=512, FHDmin=16, FHDmax=112):
    if signal.ndim > 1:
        plain = signal[:, 0]
    else:
        plain = signal
    total_bits = 8 * text_length
    bit_collected = []
    I = len(plain)
    N = I // L
    s = plain[:N * L].reshape(N, L).T
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
            if len(bit_collected) >= total_bits:
                break

            remaining = total_bits - len(bit_collected)
            if bits_this_bin > remaining:
                bits_this_bin = remaining

            retrieved_phase = (phase[k] + np.pi) % (2 * np.pi) - np.pi
            val, bits = phase_to_val(retrieved_phase, bits_this_bin)
            print(f"[DECODE] Frame {frame_idx}, Bin {k}, Phase: {retrieved_phase:.4f}, Val: {val}, Bits: {bits}, mag_db: {mag_db[k]:.2f}dB")
            bit_collected.extend(bits)
        if len(bit_collected) >= total_bits:
            break
    return bits_to_text(''.join(bit_collected[:total_bits]))

def calculate_ber_nc(original_text, recovered_text):
    original_bits = get_bits(original_text)
    recovered_bits = get_bits(recovered_text)
    min_len = min(len(original_bits), len(recovered_bits))
    original_bits = original_bits[:min_len]
    recovered_bits = recovered_bits[:min_len]
    bit_errors = sum(o != r for o, r in zip(original_bits, recovered_bits))
    ber = bit_errors / min_len
    orig_arr = np.array([int(b) for b in original_bits])
    recv_arr = np.array([int(b) for b in recovered_bits])
    nc = np.correlate(orig_arr - np.mean(orig_arr), recv_arr - np.mean(recv_arr)) / (
        np.std(orig_arr) * np.std(recv_arr) * len(orig_arr))
    return ber, nc[0]

if __name__ == "__main__":
    audio_path = "output_stego_djebbar.wav"
    text_path = "text.txt"

    # Đọc lại file text gốc
    with open(text_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    recovered = phase_dec_djebbar(wavfile.read(audio_path)[1], text_length=len(original_text), fs=44100)

    print("\n🔍 Recovered text:", recovered)
    if recovered != original_text:
        print("❌ MISMATCH! Expected:", original_text, "but got:", recovered)
    else:
        print("✅ Match verified!")

    ber, nc = calculate_ber_nc(original_text, recovered)
    print(f"\n📊 BER: {ber:.6f}, NC: {nc:.4f}")

