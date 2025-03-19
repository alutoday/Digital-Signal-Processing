import numpy as np
import librosa
import soundfile as sf

def apply_single_echo(x, sr, delay_sec=0.3, decay=0.6):
    """
    Thêm hiệu ứng echo đơn vào tín hiệu âm thanh.
    
    Parameters:
        x (numpy.ndarray): Tín hiệu âm thanh gốc.
        sr (int): Tần số lấy mẫu (sampling rate).
        delay_sec (float): Độ trễ của echo (giây).
        decay (float): Hệ số suy giảm của echo (0 < decay < 1).
        
    Returns:
        y (numpy.ndarray): Tín hiệu đã thêm echo.
    """
    delay_samples = int(delay_sec * sr)  # Chuyển độ trễ từ giây sang số mẫu
    y = np.zeros(len(x) + delay_samples)  # Mở rộng tín hiệu để chứa echo

    # Thêm tín hiệu gốc
    y[:len(x)] = x

    # Thêm echo với suy giảm
    y[delay_samples: delay_samples + len(x)] += decay * x

    # Chuẩn hóa tín hiệu tránh clipping
    y = y / np.max(np.abs(y)) * np.max(np.abs(x))

    return y

# Đọc file âm thanh gốc
input_file = "input_signals/lab_female.wav" 
x, sr = librosa.load(input_file, sr=None)

# Thêm hiệu ứng echo đơn
y = apply_single_echo(x, sr, delay_sec=0.4, decay=0.5)

# Lưu file kết quả
output_file = "output_signals/output_echosingle.wav"
sf.write(output_file, y, sr)

print("✅ Single echo effect applied! Output saved to:", output_file)
