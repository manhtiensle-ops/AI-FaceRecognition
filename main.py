# import requests
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms


BASE = "http://127.0.0.1:5000"
DETECT_EVERY = 3          
DOWNSCALE_MAX_W = 640     
BOX_THICKNESS = 2
MIN_FACE = 40             
THRESHOLDS = [0.6, 0.7, 0.7] 
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
        
    print(f"Đang học khuôn mặt của: {person_name} ...")
    embeddings_list = []
    
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(person_folder, filename)
            ref_img = cv2.imread(img_path)
            
            if ref_img is None: continue
            
            # take original image size
            H, W = ref_img.shape[:2]
            
            #DOWNSCALE 
            scale = 1.0
            if W > DOWNSCALE_MAX_W:
                scale = DOWNSCALE_MAX_W / float(W)
                newW, newH = int(W * scale), int(H * scale)
                # resize image
                img_small = cv2.resize(ref_img, (newW, newH), interpolation=cv2.INTER_LINEAR)
            else:
                img_small = ref_img
                
            # Change to RGB
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            # use mtcnn to detect on small image
            ref_boxes, _ = mtcnn.detect(Image.fromarray(rgb_small))
            
            if ref_boxes is not None:
                inv_scale = 1.0 / scale
                box = ref_boxes[0]
                
                # Caculate real coordinate from small image
                rx1 = box[0] * inv_scale
                ry1 = box[1] * inv_scale
                rx2 = box[2] * inv_scale
                ry2 = box[3] * inv_scale
                
                # solving with overflow picture's border
                clamped = clamp_box_xyxy(rx1, ry1, rx2, ry2, W, H)
                
                if clamped is not None:
                    cx1, cy1, cx2, cy2 = clamped
                    
                    # cat face
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

# ------Open WebCam -----#

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can not open webcam")
    raise SystemExit(1)

last_tracked_faces = [] 
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    if frame_id % DETECT_EVERY == 0:
        # Resize pictures
        scale = 1.0
        if W > DOWNSCALE_MAX_W:
            scale = DOWNSCALE_MAX_W / float(W)
            newW, newH = int(W * scale), int(H * scale)
            frame_small = cv2.resize(frame, (newW, newH), interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Using mtcnn to detect faces
        boxes, _ = mtcnn.detect(Image.fromarray(rgb_small))

        new_tracked_faces = []
        if boxes is not None:
            inv_scale = 1.0 / scale 

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Change to RGB
            for box in boxes:
                # Take back real coordination 
                x1, y1 = box[0] * inv_scale, box[1] * inv_scale
                x2, y2 = box[2] * inv_scale, box[3] * inv_scale
                
                clamped = clamp_box_xyxy(x1, y1, x2, y2, W, H)
                if clamped is not None:
                    cx1, cy1, cx2, cy2 = clamped
                    
                    # Cut faces from RGB picture
                    face_crop = frame_rgb[cy1:cy2, cx1:cx2]
                    
                    # Ignore faces that are too blurry or small
                    if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                        continue

                    # Using Resnet to take faces data
                    face_tensor = face_transform(face_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor)[0]
                    
                    # --- Compare with other faces ---#
                    min_dist = float('inf')
                    best_match_name = "Unknown"
                    
                    for person_name, known_emb in known_embeddings.items():
                        dist = (emb - known_emb).norm().item()
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_match_name = person_name
                    
                    if min_dist < RECOGNITION_THRESHOLD:
                        name = f"{best_match_name} ({min_dist:.2f})"
                        color = (0, 255, 0) # Green
                    else:
                        name = f"Unknown ({min_dist:.2f})"
                        color = (0, 0, 255) # Red

                    x, y, w, h = xyxy_to_xywh(clamped)
                    new_tracked_faces.append((x, y, w, h, name, color))


        last_tracked_faces = new_tracked_faces

    # print frame
    for (x, y, w, h, name, color) in last_tracked_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    # #---------------send API ----------------#
    # payload = {
    #     "data":f"Detect {name}"
    # }
    # response = requests.post(BASE + "/AI", json=payload)
    # print(response.json())
    # #----------------------------------------#
    
    frame_id += 1
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()