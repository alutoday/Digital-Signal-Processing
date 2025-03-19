import numpy as np
import soundfile as sf

# Đọc file IR gốc (có 2 kênh)
ir, sr = sf.read("ir.wav")

# Kiểm tra nếu IR có 2 kênh, chuyển thành mono
if len(ir.shape) > 1:  
    ir_mono = np.mean(ir, axis=1)  # Lấy trung bình 2 kênh
else:
    ir_mono = ir  # Nếu đã là mono thì giữ nguyên


sf.write("ir2.wav", ir_mono, sr)

print("Đã chuyển đổi IR từ stereo thành mono và lưu vào 'ir2.wav'")
