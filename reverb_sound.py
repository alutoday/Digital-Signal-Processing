import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# 1. Đọc tín hiệu âm thanh gốc (dry signal)
input_audio, sr = sf.read("input_signals/lab_female.wav")  # Đọc file WAV

# 2. Đọc tín hiệu đáp ứng xung (Impulse Response - IR)
ir, sr_ir = sf.read("ir.wav")

print("input_audio shape:", input_audio.shape)
print("ir shape:", ir.shape)

# Chuẩn hóa độ dài nếu cần (IR thường ngắn hơn tín hiệu gốc)
if len(ir) > len(input_audio):
    ir = ir[:len(input_audio)]
else:
    ir = np.pad(ir, (0, len(input_audio) - len(ir)), mode='constant')

# 3. Phép tích chập trong miền thời gian (chậm)
output_time = signal.convolve(input_audio, ir, mode='same')

# 4. Phép tích chập trong miền tần số (FFT - nhanh hơn)
N = len(input_audio) + len(ir) - 1  # Độ dài tín hiệu sau khi tích chập
N_fft = 2**np.ceil(np.log2(N)).astype(int)  # Làm tròn lên số mũ của 2 để tăng tốc FFT

# FFT của tín hiệu gốc và IR
X = fft.fft(input_audio, N_fft)
H = fft.fft(ir, N_fft)

# Nhân trong miền tần số
Y = X * H

# Biến đổi ngược về miền thời gian
output_freq = fft.ifft(Y)
output_freq = np.real(output_freq[:len(input_audio)])  # Chỉ lấy phần thực

# 5. Ghi lại file âm thanh với hiệu ứng vang
sf.write("output_signals/output_reverb_time.wav", output_time, sr)  # Kết quả từ tích chập thời gian
sf.write("output_signals/output_reverb_fft.wav", output_freq, sr)  # Kết quả từ FFT

# 6. Vẽ đồ thị so sánh
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(input_audio[:1000], label="Original Audio", color="blue")
plt.legend()
plt.title("Tín hiệu âm thanh gốc")

plt.subplot(3, 1, 2)
plt.plot(output_time[:1000], label="Reverb (Time Domain)", color="red")
plt.legend()
plt.title("Hiệu ứng vang - Tích chập miền thời gian")

plt.subplot(3, 1, 3)
plt.plot(output_freq[:1000], label="Reverb (FFT)", color="green")
plt.legend()
plt.title("Hiệu ứng vang - Tích chập miền tần số (FFT)")

plt.tight_layout()
plt.show()
