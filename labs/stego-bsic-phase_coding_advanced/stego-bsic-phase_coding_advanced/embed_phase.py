import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# ===== SUPPORT FUNCTIONS =====

def get_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def plot_phase(phase_before, phase_after, save_prefix='phase'):
    plt.figure(figsize=(12, 5))
    plt.plot(phase_before, label='Before Embedding', alpha=0.7)
    plt.plot(phase_after, label='After Embedding', alpha=0.7)
    plt.title('Phase Comparison Before vs After Embedding')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comparison.png')
    plt.close()

def phase_enc(signal, text, L=1024):
    if signal.ndim > 1:
        plain = signal[:, 0]
    else:
        plain = signal

    text_bits = get_bits(text)
    m = len(text_bits)

    I = len(plain)
    N = I // L
    max_embed_bits = (L // 2) - 1

    if m > max_embed_bits:
        raise ValueError(f"Text too long! Max {(max_embed_bits)//8} characters can be embedded.")

    s = plain[:N*L].reshape(N, L).T
    w = np.fft.fft(s, axis=0)
    Phi = np.angle(w)
    A = np.abs(w)

    # 💾 Save phase before
    phase_before = Phi[:, 0]

    DeltaPhi = np.zeros((L, N))
    for k in range(1, N):
        DeltaPhi[:, k] = Phi[:, k] - Phi[:, k-1]

    PhiData = np.array([np.pi/2 if bit == '0' else -np.pi/2 for bit in text_bits])
    Phi_new = np.copy(Phi)
    center = L // 2
    Phi_new[center - m:center, 0] = PhiData
    Phi_new[center + 1:center + 1 + m, 0] = -PhiData[::-1]

    for k in range(1, N):
        Phi_new[:, k] = Phi_new[:, k-1] + DeltaPhi[:, k]

    # 💾 Save phase after
    phase_after = Phi_new[:, 0]
    np.save("phase_before.npy", phase_before)
    np.save("phase_after.npy", phase_after)
    plot_phase(phase_before, phase_after)

    z = np.fft.ifft(A * np.exp(1j * Phi_new), axis=0)
    z = np.real(z)
    snew = z.T.reshape(N*L)
    out = np.concatenate([snew, plain[N*L:]])

    return out

# ===== MAIN SCRIPT =====

def embed_audio():
    audio_path = 'input_audio.wav'
    text_path = 'text.txt'
    output_path = 'output_phase.wav'

    fs, audio_data = wavfile.read(audio_path)
    audio_data = audio_data.astype(np.float32) / 32767.0

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\nAudio loaded: {len(audio_data)} samples, Sampling Rate: {fs} Hz")
    print(f"Text loaded: {len(text)} characters, {len(get_bits(text))} bits to embed")

    out = phase_enc(audio_data, text)

    if np.max(np.abs(out)) > 1.0:
        out = out / np.max(np.abs(out))

    wavfile.write(output_path, fs, np.int16(out * 32767))
    print(f"\nDone! Data hidden successfully to: {output_path}")
    print("Phase plot saved as phase_comparison.png")

if __name__ == "__main__":
    embed_audio()
