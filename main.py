import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
import sys

# ==========================================
# 參數設定
# ==========================================
VIDEO_PATH = 'demo.mp4'      # 影片路徑
MODEL_PATH = 'best.pt'       # 模型路徑
CONFIDENCE_THRESHOLD = 0.35  # 信心度
MERGE_DISTANCE = 50          # 合併距離
WINDOW_NAME = "Smart Safety Monitor"
# ==========================================

DANGER_ZONE = np.array([
    [100, 100], [600, 100], 
    [600, 600], [100, 600]
], np.int32)

print("--- 系統啟動 ---")
print("功能鍵說明：")
print(" [P] 暫停 / 繼續")
print(" [R] 重播影片 (歸零計數)")
print(" [Q] 結束程式")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"錯誤：找不到 {MODEL_PATH}")
    sys.exit()

model_names = model.names
cap = cv2.VideoCapture(VIDEO_PATH)

BAD_KEYWORDS = ['no', 'head', 'face'] 
HUMAN_KEYWORDS = ['person', 'worker', 'head', 'helmet', 'vest', 'mask', 'no']

# 狀態變數
is_paused = False

def count_violating_people(boxes):
    if not boxes: return 0
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        centers.append((x1 + x2) // 2)
    centers.sort()
    person_count = 0
    if len(centers) > 0:
        person_count = 1
        current_ref = centers[0]
        for i in range(1, len(centers)):
            if abs(centers[i] - current_ref) > MERGE_DISTANCE:
                person_count += 1
                current_ref = centers[i]
    return person_count

while True:
    # 1. 處理鍵盤輸入 (放在迴圈開頭或結尾都可以，這裡放開頭方便處理暫停邏輯)
    key = cv2.waitKey(1) & 0xFF
    
    # [Q] 結束
    if key == ord('q') or key == 27: # 27 is Esc
        print("結束程式。")
        break
    
    # [P] 暫停開關
    if key == ord('p'):
        is_paused = not is_paused # 切換狀態
        print(f"暫停狀態: {is_paused}")

    # [R] 重播
    if key == ord('r'):
        print("重播影片...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_paused = False # 重播時自動取消暫停
        continue

    # 檢查視窗關閉 (X 按鈕)
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        # 注意：剛啟動時還沒建立視窗，getProperty 可能會回傳 -1，所以要確保視窗已建立
        # 這裡加個簡單判斷，避免程式剛跑還沒視窗就跳出
        if cap.get(cv2.CAP_PROP_POS_FRAMES) > 1: 
            break

    # ==========================
    # 暫停邏輯
    # ==========================
    if is_paused:
        # 如果暫停中，我們不讀取新影格，只顯示"PAUSED"文字
        # 利用上一幀的 img (因為 Python 變數作用域關係，img 會保留上一圈的值)
        # 為了不破壞原本的 img，我們畫在 copy 上
        if 'img' in locals():
            paused_img = img.copy()
            # 畫一個半透明黑色遮罩讓畫面變暗
            overlay = paused_img.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, paused_img, 0.6, 0, paused_img)
            
            # 寫上 PAUSED
            cvzone.putTextRect(paused_img, "PAUSED", (540, 360), scale=5, thickness=5, colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(paused_img, "Press 'P' to Continue", (520, 450), scale=2, thickness=2, colorR=(0, 0, 0), offset=10)
            cv2.imshow(WINDOW_NAME, paused_img)
        continue # 跳過後面的 YOLO 偵測，直接進入下一次迴圈等待按鍵

    # ==========================
    # 正常播放邏輯
    # ==========================
    success, img = cap.read()
    if not success:
        print("影片播放結束。 (按 'R' 重播，按 'Q' 離開)")
        # 影片結束時自動進入暫停狀態，方便使用者決定下一步
        is_paused = True 
        continue
    
    img = cv2.resize(img, (1280, 720))

    # 繪製危險區域
    overlay = img.copy()
    cv2.fillPoly(overlay, [DANGER_ZONE], (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # AI 偵測
    results = model(img, stream=True)
    
    violation_boxes = []
    zone_boxes = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            name = model_names[cls].lower()

            if conf > CONFIDENCE_THRESHOLD:
                # 違規偵測
                color = (0, 255, 0)
                if any(k in name for k in BAD_KEYWORDS):
                    color = (0, 0, 255) # 紅色
                    violation_boxes.append([x1, y1, x2, y2])

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=color)
                cvzone.putTextRect(img, f"{name} {conf}", (max(0, x1), max(35, y1)), 
                                   scale=1, thickness=1, colorR=color, offset=5)

                # 危險區偵測
                cx, cy = x1 + w // 2, y1 + h // 2
                in_zone = cv2.pointPolygonTest(DANGER_ZONE, (cx, cy), False) >= 0
                is_human_obj = any(k in name for k in HUMAN_KEYWORDS)

                if in_zone and is_human_obj:
                    zone_boxes.append([x1, y1, x2, y2])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)

    # 計數
    real_violation_count = count_violating_people(violation_boxes)
    real_zone_count = count_violating_people(zone_boxes)

    # 儀表板
    cvzone.putTextRect(img, f'Violations: {real_violation_count}', (50, 50), scale=2, thickness=2, colorR=(0,0,255), offset=10)
    bg_color = (0, 165, 255) if real_zone_count > 0 else (0, 255, 0)
    cvzone.putTextRect(img, f'Zone People: {real_zone_count}', (850, 50), scale=2, thickness=2, colorR=bg_color, offset=10)

    if real_zone_count > 0:
        cvzone.putTextRect(img, 'WARNING: ZONE INTRUSION', (350, 120), scale=3, thickness=3, colorR=(0,0,255), offset=20)

    # 顯示畫面
    cv2.imshow(WINDOW_NAME, img)

cap.release()
cv2.destroyAllWindows()
sys.exit()