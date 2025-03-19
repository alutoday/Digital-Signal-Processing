import numpy as np
import librosa
import soundfile as sf

def apply_echo(x, sr, delay_sec=0.3, decay=0.6, num_echoes=5):
    """
    Thêm hiệu ứng echo nhiều lần vào tín hiệu âm thanh.
    
    Parameters:
        x (numpy.ndarray): Tín hiệu âm thanh gốc.
        sr (int): Tần số lấy mẫu (sampling rate).
        delay_sec (float): Độ trễ giữa các echo (giây).
        decay (float): Hệ số suy giảm của mỗi echo (0 < decay < 1).
        num_echoes (int): Số lần echo.
        
    Returns:
        y (numpy.ndarray): Tín hiệu đã thêm echo.
    """
    delay_samples = int(delay_sec * sr)  # Chuyển độ trễ từ giây sang số mẫu
    y = np.zeros(len(x) + delay_samples * num_echoes)  # Mở rộng tín hiệu để chứa echo

    # Thêm tín hiệu gốc
    y[:len(x)] = x

    # Thêm từng echo với suy giảm dần
    for i in range(1, num_echoes + 1):
        y[i * delay_samples : i * delay_samples + len(x)] += (decay ** i) * x

    # Chuẩn hóa tín hiệu tránh clipping
    y = y / np.max(np.abs(y)) * np.max(np.abs(x))

    return y

# Đọc file âm thanh gốc
input_file = "input_signals/lab_female.wav" 
x, sr = librosa.load(input_file, sr=None)

# Thêm hiệu ứng echo
y = apply_echo(x, sr, delay_sec=0.4, decay=0.5, num_echoes=6)

# Lưu file kết quả
output_file = "output_signals/output_echomulti.wav"
sf.write(output_file, y, sr)

print("✅ Echo effect applied! Output saved to:", output_file)
