import torch

# Kiểm tra phiên bản PyTorch
print("Phiên bản PyTorch:", torch.__version__)

# Kiểm tra xem CUDA có được hỗ trợ hay không
print("Có hỗ trợ CUDA không:", torch.cuda.is_available())

# Nếu CUDA có hỗ trợ, kiểm tra phiên bản CUDA
if torch.cuda.is_available():
    print("Phiên bản CUDA:", torch.version.cuda)
    print("Số lượng GPU có thể sử dụng:", torch.cuda.device_count())
    print("Tên GPU đang sử dụng:", torch.cuda.get_device_name(0))
else:
    print("Không có hỗ trợ CUDA hoặc không có GPU NVIDIA trên hệ thống.")
