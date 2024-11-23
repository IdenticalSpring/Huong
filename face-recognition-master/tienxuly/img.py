import cv2
import os

# Đường dẫn tới video
video_path = './tienxuly/IMG_7314.MOV'
output_folder = './datasets/backup/Huong'

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Cài đặt frame_interval để cắt từ mỗi khung hình
frame_interval = 1

# Đọc video
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu không thể mở video
if not cap.isOpened():
    print("Không thể mở video.")
else:
    frame_count = 0
    image_count = 0

    # Lặp qua từng khung hình
    while True:
        ret, frame = cap.read()
        
        # Thoát vòng lặp nếu hết khung hình
        if not ret:
            break

        # Lưu mỗi khung hình
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Lưu ảnh: {image_path}")
            image_count += 1

        frame_count += 1

    # Giải phóng tài nguyên
    cap.release()
    print("Hoàn thành cắt ảnh từ video.")
