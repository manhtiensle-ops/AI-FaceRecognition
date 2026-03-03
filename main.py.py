import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

DETECT_EVERY = 3          
DOWNSCALE_MAX_W = 640     
BOX_THICKNESS = 2
MIN_FACE = 40             
THRESHOLDS = [0.6, 0.7, 0.7] 
# Ngưỡng khoảng cách nhận diện (Euclidean distance). Dưới 0.8 thường là cùng một người.
RECOGNITION_THRESHOLD = 0.8 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    thresholds=THRESHOLDS, 
    min_face_size=MIN_FACE, 
    post_process=False
)

# InceptionResnetV1 dùng để NHẬN DIỆN, biến ảnh thành 512 con số
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#Ép ảnh khuôn mặt về chuẩn 160x160 cho Resnet
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def clamp_box_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W - 1, int(x2)); y2 = min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

# ===== 3.ENROLLMENT =====#

print("Đang quét thư mục dataset để học khuôn mặt...")

base_dataset_folder = 'dataset'
known_embeddings = {} 

for person_name in os.listdir(base_dataset_folder):
    person_folder = os.path.join(base_dataset_folder, person_name)
    
    if not os.path.isdir(person_folder):
        continue
        
    print(f" -> Đang học khuôn mặt của: {person_name} ...")
    embeddings_list = []
    
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(person_folder, filename)
            ref_img = cv2.imread(img_path)
            
            if ref_img is None: continue
            
            # Lấy kích thước ảnh gốc
            H, W = ref_img.shape[:2]
            
            #DOWNSCALE 
            scale = 1.0
            if W > DOWNSCALE_MAX_W:
                scale = DOWNSCALE_MAX_W / float(W)
                newW, newH = int(W * scale), int(H * scale)
                # Tạo một bản nháp thu nhỏ 
                img_small = cv2.resize(ref_img, (newW, newH), interpolation=cv2.INTER_LINEAR)
            else:
                img_small = ref_img
                
            # Đổi màu cho cả bản gốc (để cắt) và bản nháp (để tìm)
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            # Bắt MTCNN tìm mặt trên BẢN NHÁP THU NHỎ
            ref_boxes, _ = mtcnn.detect(Image.fromarray(rgb_small))
            
            if ref_boxes is not None:
                inv_scale = 1.0 / scale
                box = ref_boxes[0]
                
                # --- ÁP DỤNG KỸ THUẬT NHÂN NGƯỢC ---
                # Phóng to tọa độ từ bản nháp về lại kích thước thật
                rx1 = box[0] * inv_scale
                ry1 = box[1] * inv_scale
                rx2 = box[2] * inv_scale
                ry2 = box[3] * inv_scale
                
                # Ép khung không bị tràn ra ngoài viền ảnh GỐC (W, H)
                clamped = clamp_box_xyxy(rx1, ry1, rx2, ry2, W, H)
                
                if clamped is not None:
                    cx1, cy1, cx2, cy2 = clamped
                    
                    # Cắt mặt từ BẢN GỐC SẮC NÉT
                    face_cropped = ref_rgb[cy1:cy2, cx1:cx2]
                    
                    if face_cropped.shape[0] < 40 or face_cropped.shape[1] < 40:
                        continue
                        
                    face_tensor = face_transform(face_cropped).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = resnet(face_tensor)[0]
                        embeddings_list.append(emb)

    if len(embeddings_list) > 0:
        known_embeddings[person_name] = torch.stack(embeddings_list).mean(dim=0)
        print(f" [OK] Đã lưu {person_name} (từ {len(embeddings_list)} ảnh).")
    else:
        print(f" [Cảnh báo] Không tìm thấy khuôn mặt hợp lệ cho {person_name}.")

if len(known_embeddings) == 0:
    print("Lỗi: Không học được khuôn mặt của ai cả. Hãy kiểm tra lại dataset!")
    raise SystemExit(1)

print("\nĐã học xong tất cả! Bắt đầu mở Webcam...")

# ===== 4. VÒNG LẶP XỬ LÝ WEBCAM =======#

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam")
    raise SystemExit(1)

last_tracked_faces = [] 
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    if frame_id % DETECT_EVERY == 0:
        # Thu nhỏ ảnh 
        scale = 1.0
        if W > DOWNSCALE_MAX_W:
            scale = DOWNSCALE_MAX_W / float(W)
            newW, newH = int(W * scale), int(H * scale)
            frame_small = cv2.resize(frame, (newW, newH), interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Gọi MTCNN tìm mặt
        boxes, _ = mtcnn.detect(Image.fromarray(rgb_small))

        new_tracked_faces = []
        if boxes is not None:
            inv_scale = 1.0 / scale 
            for box in boxes:
                # Phóng to tọa độ về kích thước ảnh gốc
                x1, y1 = box[0] * inv_scale, box[1] * inv_scale
                x2, y2 = box[2] * inv_scale, box[3] * inv_scale
                
                clamped = clamp_box_xyxy(x1, y1, x2, y2, W, H)
                if clamped is not None:
                    cx1, cy1, cx2, cy2 = clamped
                    
                    # CẮT KHUÔN MẶT TỪ ẢNH GỐC SẮC NÉT ĐỂ ĐƯA ĐI NHẬN DIỆN
                    face_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[cy1:cy2, cx1:cx2]
                    
                    # Bỏ qua nếu mặt quá mờ hoặc quá nhỏ
                    if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                        continue

                    # Đưa mặt qua AI Resnet để trích xuất 512 con số
                    face_tensor = face_transform(face_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor)[0]
                    
                    # --- SO SÁNH VỚI NHIỀU NGƯỜI TRONG TỪ ĐIỂN ---
                    min_dist = float('inf') # Đặt khoảng cách ban đầu là vô cực
                    best_match_name = "Unknown"
                    
                    # Lặp qua tất cả những người AI đã học để tìm người giống nhất
                    for person_name, known_emb in known_embeddings.items():
                        dist = (emb - known_emb).norm().item()
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_match_name = person_name
                    
                    # Nếu độ chênh lệch nhỏ hơn ngưỡng cho phép thì nhận diện thành công
                    if min_dist < RECOGNITION_THRESHOLD:
                        name = f"{best_match_name} ({min_dist:.2f})"
                        color = (0, 255, 0) # Xanh lá
                    else:
                        name = f"Unknown ({min_dist:.2f})"
                        color = (0, 0, 255) # Đỏ

                    x, y, w, h = xyxy_to_xywh(clamped)
                    new_tracked_faces.append((x, y, w, h, name, color))

        last_tracked_faces = new_tracked_faces

    # Vẽ khung và in tên lên ảnh
    for (x, y, w, h, name, color) in last_tracked_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition He Thong", frame)

    frame_id += 1
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        break

cap.release()
cv2.destroyAllWindows()