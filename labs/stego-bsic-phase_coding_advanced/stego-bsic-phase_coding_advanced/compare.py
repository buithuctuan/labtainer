import numpy as np
import matplotlib.pyplot as plt

# === Load dữ liệu đã lưu từ 2 phương pháp ===
phase_coding_before = np.load("phase_before.npy")
phase_coding_after = np.load("phase_after.npy")

clsb_before = np.load("clsb_phase_before.npy")
clsb_after = np.load("clsb_phase_after.npy")

# === Vẽ biểu đồ: So sánh pha trước và sau của từng phương pháp ===
plt.figure(figsize=(14, 10))

# Subplot 1: Phase Coding
plt.subplot(2, 1, 1)
plt.plot(phase_coding_before, label='Trước khi nhúng', alpha=0.7)
plt.plot(phase_coding_after, label='Sau khi nhúng', alpha=0.7)
plt.title('Phase Coding - So sánh pha trước và sau khi nhúng')
plt.xlabel('Tần số (Frequency Bin)')
plt.ylabel('Pha (Radian)')
plt.legend()
plt.grid(True)

# Subplot 2: CLSB
plt.subplot(2, 1, 2)
plt.plot(clsb_before, label='Trước khi nhúng', alpha=0.7)
plt.plot(clsb_after, label='Sau khi nhúng', alpha=0.7)
plt.title('CLSB - So sánh pha trước và sau khi nhúng')
plt.xlabel('Tần số (Frequency Bin)')
plt.ylabel('Pha (Radian)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print("Succes")
