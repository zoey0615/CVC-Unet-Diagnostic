# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import gradio as gr #不需要撰寫複雜的 HTML、CSS 或 JavaScript，就能在 Python 中直接定義網頁元件
import torch
import torch.nn as nn #Neural Networks（神經網路）。 PyTorch 中專門建立、訓練、以及優化深度學習模型而設計的子庫（Sub-library）。
import albumentations #影像增強
import segmentation_models_pytorch as smp #基於 PyTorch 的開源套件，專門為影像分割任務提供了大量預先寫好的模型架構、編碼器與損失函數。
import carinanet
from PIL import Image
import pandas as pd
import time
from skimage.morphology import skeletonize

# ─── Configuration ────────────────
IMAGE_SIZE  = 1024
KERNEL_TYPE = "unet++b5_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo_cvc" #模型架構與骨幹(unet++ EfficientNet-B5)、損失函數2-channel Binary Cross Entropy (2-channel BCE)、與影像1024
#繪製了一個半徑為 15 像素的圓形區域作為標籤、學習率1*10^{-4}、批次大小4、數據增強、訓練輪次 (Epochs)30

ENET_TYPE   = "timm-efficientnet-b5" #backbone :EfficientNet-B5
MODEL_DIR   = "C:/WorkArea/zoey/unet/model" 
N_FOLDS     = 5 #5-Fold 交叉驗證(參與集成預測的模型總數/資料切分的份數)
CSV_PATH    = "output_masks/coordinates_with_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Preprocessing ────────────────────────────────────────────────────────────
transforms_val = albumentations.Compose([albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE)]) #縮放大小
#將原始的圖片檔案格式，轉換為模型可以運算的數學張量
def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, torch.Tensor]: #型別提示(函式預期接收什麼樣的輸入，並會回傳什麼樣的結果。)
    image_np = np.array(pil_image.convert("RGB")) #將讀入的 PIL 影像強制轉換為 RGB 三通道模式，將 PIL 物件轉換為 NumPy 陣列。
    res = transforms_val(image=image_np) #統一影像尺寸
    image_t = res["image"].astype(np.float32).transpose(2, 0, 1) / 255.0 #將像素值從 8 位元整數（0-255）轉為 32 位元浮點數，
    tensor = torch.tensor(image_t).unsqueeze(0) #將NumPy 陣列轉換為 PyTorch 的張量（Tensor）物件，在第 0 維（最前面）增加一個大小為 1 的新維度。[bchw]
    return res["image"], tensor #調整尺寸後的原始影像矩陣。給 AI 模型（Unet++）進行數學運算的結構

# ─── Helper: Make Overlay ───────據預測的機率圖（prob_map）覆蓋上一層半透明的彩色遮罩──────────
def make_overlay(image_np, prob_map, threshold, alpha):
    overlay = image_np.copy() #複製影像
    mask_colors = [(0, 255, 0), (255, 0, 0)] # Green: Clip, Red: Tip
    for i in range(2): #迴圈會依序處理這兩項特徵
        mask = (prob_map[i] >= threshold).astype(np.uint8) #將機率圖（值為 0~1 的浮點數）與門檻值（例如 0.5）進行比較，產生一個布林值矩陣（True/False）。
        color_mask = np.zeros_like(image_np)  #建立一個跟原圖大小完全相同、但全是黑色（數值全為 0）的畫布。
        color_mask[:] = mask_colors[i] #把 mask_colors[i] 這個顏色，填滿 color_mask 畫布裡的每一個像素點
        idx = mask > 0 #找出遮罩中數值為 1 的所有位置（像素點）。
        overlay[idx] = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)[idx] #半透明疊加=目前的原始影像*原圖透明度+要疊加的顏色*透明度+不額外增亮
    return overlay

# ─── Model Class ──────────────────────────────────────────────────────────────
class SegModel(nn.Module): #繼承 nn.Module父類別管理神經網路模塊
    def __init__(self, backbone: str): #初始化定義這個模型結構
        super().__init__() #繼承父類別
        # 將接收到的 backbone 傳給 encoder_name
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights=None, classes=2, activation=None) #U-Net++:捕捉 CVC 導管邊緣或尖端
    def forward(self, x):
        feats = self.seg.encoder(x) #特徵提取（編碼）
        dec_out = self.seg.decoder(feats) #特徵還原（解碼
        return self.seg.segmentation_head(dec_out) #最終分類（輸出頭）回傳logit
#用於集成學習、讀入5個權重檔
def load_models():
    models = [] #建立一個空的列表，用來存放載入完成的模型實例
    for fold in range(N_FOLDS):
        path = os.path.join(MODEL_DIR, f"{KERNEL_TYPE}_best_fold{fold}.pth") #構建權重檔案的路徑
        if os.path.exists(path):
            m = SegModel(ENET_TYPE) #實例SegModel（含有 U-Net++ 和 EfficientNet-B5 骨幹）
            m.load_state_dict(torch.load(path, map_location=DEVICE), strict=False) #讀取 .pth 權重字典。
            m.to(DEVICE).eval() #將模型切換至評估模式
            models.append(m) #將設定好的模型加入列表。
    return models

print("Loading UNet++ Models...")
MODELS = load_models()

# ─── Logic Functions ─────mask-> points
def extract_path_points(mask, sample_step=15):
    """把預測出的綠色路徑（Clip Mask）轉化成一串座標點，並存入 CSV"""
    # 確保 mask 是二值化的
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) #將輸入的灰階遮罩（0-255）進行切分。像素值大於 127 的變成 255（純白），其餘變成 0（純黑）。回傳門檻值(不使用)、二值化影像矩陣
    # 找出所有非零像素點
    points = cv2.findNonZero(binary) #掃描整張影像，找出所有像素值為「白（非零）」的座標。
    if points is None:
        return "[]" #字串
    # 將[n,[x,y]]轉為列表格式 [(x, y), ...]
    pts = [p[0].tolist() for p in points]
    
    # 為了避免點太多導致 CSV 爆炸，進行等距採樣 (例如每 15 個點取一個)
    sampled_pts = pts[::sample_step] #list[start : stop : step]
    
    # 確保最後一個點 (通常是 Tip) 有被包含進去
    if pts[-1] not in sampled_pts:
        sampled_pts.append(pts[-1])
        
    return str(sampled_pts)
def get_ordered_path(clip_mask, carina_pos=None, tip_hint=None):
    """
    tip_hint: (x, y) 由外部經由過濾後的 AI Tip 遮罩質心提供
              Level 1 仲裁使用，專門精準破解「折返/Upward Malposition」案例的起點逆序問題
    """
    skeleton = skeletonize(clip_mask > 0).astype(np.uint8)
    points = np.column_stack(np.where(skeleton > 0))  # [[y, x], ...]
    if len(points) < 10:
        return points

    # ── 內部工具函式 ──────────────────────────────────────────
    def run_nn_from(start_idx, pts, target_idx=None):
        """從 start_idx 出發做 Nearest Neighbor，可指定 target_idx 作為終點"""
        ordered = []
        remaining = list(range(len(pts)))
        curr = start_idx
        while remaining:
            ordered.append(pts[curr])
            if target_idx is not None and curr == target_idx:
                break
            remaining.remove(curr)
            if not remaining:
                break
            dists = np.sum((pts[remaining] - pts[curr]) ** 2, axis=1)
            nearest = np.argmin(dists)
            if dists[nearest] > 225:  # 15px 斷線門檻
                break
            curr = remaining[nearest]
        return np.array(ordered)

    def find_point_idx(target_yx, pts):
        """找出 target_yx 在 pts 陣列中最近的索引"""
        dists = np.sum((pts - target_yx) ** 2, axis=1)
        return int(np.argmin(dists))

    # ── 找骨架端點（鄰居數 == 1 的像素）────────────────────────
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    endpoint_mask = (skeleton > 0) & (neighbor_count == 1)
    endpoints = np.column_stack(np.where(endpoint_mask))  # [[y, x], ...]

    if len(endpoints) < 2:
        # 端點不足，退回傳統邏輯：Y 最小當起點
        return run_nn_from(np.argmin(points[:, 0]), points)

    # 把所有端點預先映射到 points 陣列的索引
    ep_indices = np.array([find_point_idx(ep, points) for ep in endpoints])

    tip_idx   = None
    entry_idx = None

    # ════════════════════════════════════════════════════════════
    # 【Level 1】AI 篩選後的 Tip 遮罩直接指定
    #   最可靠！專門處理「折返/Upward Malposition」案例
    #   因為折返時 Tip 在上方，傳統距離仲裁會選錯方向，這裡直接用 tip_hint 強制鎖定終點！
    # ════════════════════════════════════════════════════════════
    if tip_hint is not None:
        tip_hint_yx = np.array([tip_hint[1], tip_hint[0]])  # (x,y) → (y,x)
        ep_dists_to_tip = np.sqrt(np.sum((endpoints - tip_hint_yx) ** 2, axis=1))
        best_ep = np.argmin(ep_dists_to_tip)

        # 距離門檻 120px：太遠代表這個 Tip 提示跟這條骨架對不上，視為雜訊
        if ep_dists_to_tip[best_ep] < 120:
            tip_idx = ep_indices[best_ep]
            # 入口起點 = 離這個 tip_idx 位置物理距離最遠的那個端點（也就是脖子入口）
            tip_pt = points[tip_idx]
            other_mask = ep_indices != tip_idx
            if np.any(other_mask):
                other_pts = points[ep_indices[other_mask]]
                far_in_others = np.argmax(np.sqrt(np.sum((other_pts - tip_pt) ** 2, axis=1)))
                entry_idx = ep_indices[other_mask][far_in_others]

    # ════════════════════════════════════════════════════════════
    # 【Level 2】解剖學區域篩選 + Carina 距離仲裁（備援機制）
    #   無 Tip 遮罩，或 Tip 遮罩距骨架太遠時啟用
    # ════════════════════════════════════════════════════════════
    if tip_idx is None and carina_pos is not None:
        car_y_coord, car_x_coord = carina_pos[1], carina_pos[0]
        upper_mask = endpoints[:, 0] < (car_y_coord + 120)

        if np.any(upper_mask):
            upper_eps     = endpoints[upper_mask]
            upper_ep_idxs = ep_indices[upper_mask]
            dists_to_carina = np.sqrt((upper_eps[:, 0] - car_y_coord) ** 2 + (upper_eps[:, 1] - car_x_coord) ** 2)
            entry_idx = upper_ep_idxs[np.argmax(dists_to_carina)]

            # Tip = 剩餘端點中離 entry 最遠的
            other_mask = ep_indices != entry_idx
            if np.any(other_mask):
                entry_pt   = points[entry_idx]
                other_pts  = points[ep_indices[other_mask]]
                far_in_others = np.argmax(np.sqrt(np.sum((other_pts - entry_pt) ** 2, axis=1)))
                tip_idx = ep_indices[other_mask][far_in_others]
            else:
                tip_idx = ep_indices[0]
        else:
            # 純距離仲裁
            ep_dists = np.sqrt((endpoints[:, 0] - car_y_coord) ** 2 + (endpoints[:, 1] - car_x_coord) ** 2)
            entry_idx = ep_indices[np.argmax(ep_dists)]
            tip_idx   = ep_indices[np.argmin(ep_dists)]

    # ════════════════════════════════════════════════════════════
    # 【Level 3】純 Y 軸（最後防線，連 Carina 都沒抓到時）
    # ════════════════════════════════════════════════════════════
    if tip_idx is None:
        entry_idx = find_point_idx(endpoints[np.argmin(endpoints[:, 0])], points)
        tip_idx   = find_point_idx(endpoints[np.argmax(endpoints[:, 0])], points)

    return run_nn_from(entry_idx, points, target_idx=tip_idx)
def detect_carina(image_np):
    h, w = image_np.shape[:2]
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 2. 註解掉 CLAHE (測試真實影像信心值是否回升)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced_gray = clahe.apply(gray)
    
    # 3. 直接將原始灰階轉回 RGB (因為模型通常預期 3 通道輸入)
    input_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    fallback_x, fallback_y = int(0.5 * w), int(0.42 * h)

    try:
        result = carinanet.predict_carina_ett(input_image)
        car = result.get("carina", None)
        # 提取真實機率分數
        car_conf = result.get("carina_confidence", 0.0) 
        
        if car is None:
            return fallback_x, fallback_y, 0.0
        # 從數據看，(303.5, 186.5) -> 303.5 才是 X
        # 所以我們應該這樣賦值：
        val1, val2 = car[0], car[1]
        
        # 自動邏輯判斷：在 640 空間中，X 通常在中間 (250-400)，
        # 而 Carina 的 Y (高度) 通常比較淺 (150-250)。
        if val1 > val2:
            x_raw, y_raw = val1, val2
        else:
            x_raw, y_raw = val2, val1

        MODEL_SIZE = 640.0
        scale_w = w / MODEL_SIZE
        scale_h = h / MODEL_SIZE

        # 最終轉換
        final_x = int(x_raw * scale_w)
        final_y = int(y_raw * scale_h)

        # 二次安全檢查：確保 X 不會偏離中線太遠
        if not (w * 0.35 < final_x < w * 0.65):
            print("⚠️ 比例異常，改用 Fallback X")
            final_x = fallback_x

        print(f"🎯 修正完成 -> X: {final_x}, Y: {final_y} (原始: {car})")
        return final_x, final_y, car_conf
    except Exception as e:
        return fallback_x, fallback_y, 0.0
#計算導管尖端（Tip）與解剖基準點（Carina）的相對位置，並根據醫學臨床準則給出診斷標籤。
def check_malposition(tip_y, car_y, tip_x, car_x, is_looping_up=False):
    if is_looping_up:
        return "Abnormal (Upward Malposition)", (0, 0, 255)
    if tip_y is None or car_y is None: return "Unknown", (128, 128, 128) #如果其中一個點沒抓到（None），系統會回傳 "Unknown" 並顯示灰色 (128, 128, 128)。
    y_diff = tip_y - car_y #縱向位移：決定深淺(>0導管在carina下方)
    x_diff = tip_x - car_x #橫向位移：保留正負號，用來判斷左右方向（以偵測到的 car_x 為基準）

    if abs(x_diff) > 30: #如果橫向位移>180像素（約 6 公分）
        side = "Right" if x_diff > 0 else "Left" #x_diff > 0 代表 tip 在 carina 右側，反之在左側
        return f"Abnormal (Wrong Vessel - {side})", (0, 0, 255) #標示出偏右或偏左
    if 0 <= y_diff <= 150: #如果縱向位移介於0~150(約為 Carina 下方4.5 ~5$ 公分)之間
        return "Normal", (0, 255, 0)  
    elif (150 < y_diff <= 250) or (-50 <= y_diff < 0):
        return "Borderline", (0, 165, 255) 
    else:
        return "Abnormal (Too Deep/High)", (0, 0, 255)

def segment(pil_image, threshold_pct: float, alpha_pct: float):   #門檻值百分比、透明度百分比
    if not pil_image: return None, None, None, "⚠️ No image"
    threshold, alpha = threshold_pct/100.0, alpha_pct/100.0
    original_np, tensor = preprocess(pil_image)

    # 推論 CVC
    with torch.no_grad():
        if MODELS:
            prob_map_tensor = torch.stack([m(tensor.to(DEVICE)).sigmoid() for m in MODELS]).mean(0)[0]
            prob_map = prob_map_tensor.cpu().numpy()
        else:
            prob_map = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE))

    # 1. 取得 Carina 座標與信心 (這必須在 get_ordered_path 之前)
    car_x, car_y, car_conf = detect_carina(original_np)

    # 2. 提取遮罩
    tip_mask_raw = (prob_map[1] >= threshold).astype(np.uint8) * 255 
    clip_mask = (prob_map[0] >= threshold).astype(np.uint8) * 255 #二值化導管遮罩
    
    # 3. 找出所有尖端連通域 (藍色三角形來源)
    contours, _ = cv2.findContours(tip_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5] 
    
    clean_tip_mask = np.zeros_like(tip_mask_raw) 
    all_tip_coords = [] 
    for cnt in valid_contours: 
        cv2.drawContours(clean_tip_mask, [cnt], -1, 255, -1)
        M = cv2.moments(cnt) 
        if M["m00"] != 0: 
            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"])
            all_tip_coords.append([cx, cy])

    # ─── 核心修改：路徑排序與診斷邏輯 ───
    main_tip_x, main_tip_y = None, None 
    is_looping_up = False 
    is_inferred = False 
    infer_text = ""
    tip_hint = None
    if all_tip_coords:
        if len(all_tip_coords) == 1:
            tip_hint = (all_tip_coords[0][0], all_tip_coords[0][1])
        else:
            # 橫向距離仲裁：真正的 Tip（不論深入還是回勾）都會在上腔靜脈附近，離 Carina 的 X 軸最近
            best_raw_tip = min(all_tip_coords, key=lambda p: abs(p[0] - car_x))
            tip_hint = (best_raw_tip[0], best_raw_tip[1])
    # 重要：傳入 carina 座標，讓路徑從脖子往胸腔方向排序
    path_points = get_ordered_path(clip_mask, carina_pos=(car_x, car_y), tip_hint=tip_hint)

    # A. 判斷方向 (是否向上回勾)
    if len(path_points) > 20:
        # 取路徑最後 15 個點來觀察趨勢
        tail_segment = path_points[-25:] 
        y_start_tail = tail_segment[0][0] # 尾段的起點 Y
        y_end_tail = tail_segment[-1][0]  # 尾段的終點 Y
        
        # 邏輯：如果終點的 Y 比起點的 Y 還要「小」(座標系上方)，代表路徑末端向上翹
        if y_end_tail < y_start_tail - 3: 
            is_looping_up = True

    # B. 決定最終診斷點 (Main Tip)
    if len(path_points) > 0:
        # path_points 已經過排序，最後一個點就是「地理上的末端」
        last_path_point = (int(path_points[-1][1]), int(path_points[-1][0])) # (x, y)
        
        if all_tip_coords:
            # 尋找與「路徑末端」物理距離最近的 Tip 遮罩點
            # 這能有效排除掉出現在導管中段、血管分叉處的「假紅點」
            best_tip = min(all_tip_coords, key=lambda p: np.linalg.norm(np.array(p) - np.array(last_path_point)))
            main_tip_x, main_tip_y = best_tip[0], best_tip[1]
            is_inferred = False
        else:
            # 如果 AI 沒抓到 Tip 遮罩，直接用路徑末端作為推測點
            main_tip_x, main_tip_y = last_path_point[0], last_path_point[1]
            is_inferred = True
            
    elif all_tip_coords:
        # 極端情況：連 Clip 路徑都斷了，只好找最下方的 Tip 點
        best_tip = max(all_tip_coords, key=lambda p: p[1])
        main_tip_x, main_tip_y = best_tip[0], best_tip[1]
        is_inferred = False

    # ─── 後續繪圖與存檔邏輯 (保持不變) ───
    cvc_conf = 0.0
    if main_tip_x is not None and main_tip_y is not None:
        ty_idx = min(max(int(main_tip_y), 0), IMAGE_SIZE - 1)
        tx_idx = min(max(int(main_tip_x), 0), IMAGE_SIZE - 1)
        cvc_conf = float(prob_map[1, ty_idx, tx_idx])

    combined_conf = car_conf * cvc_conf
    label, color_bgr = check_malposition(main_tip_y, car_y, main_tip_x, car_x, is_looping_up)
    overlay = make_overlay(original_np, prob_map, threshold, alpha)
    
# 1. 畫 Carina (黃色實心帶白邊)
    if car_x and car_y:
        cv2.circle(overlay, (car_x, car_y), 10, (255, 255, 255), -1) # 白底
        cv2.circle(overlay, (car_x, car_y), 7, (255, 255, 0), -1)    # 黃色中心 

    # 2. 畫所有偵測到的 Tip 候選點 (藍色空心圓)
    for tx, ty in all_tip_coords:
        size = 12  # 三角形的大小
        # 定義三角形的三個頂點
        pt1 = (tx, ty - size)      # 上頂點
        pt2 = (tx - size, ty + size) # 左下頂點
        pt3 = (tx + size, ty + size) # 右下頂點
        
        # 建立頂點陣列並繪製空心三角形
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(overlay, [triangle_cnt], 0, (0, 0, 255), 2) # (藍色, 線寬2)

    # 3. 標註最終診斷點 (實心圓)
    if main_tip_x is not None and main_tip_y is not None:
        if is_inferred:
            # 【推測點：橘色實心】
            cv2.circle(overlay, (main_tip_x, main_tip_y), 11, (255, 255, 255), -1) # 白邊
            cv2.circle(overlay, (main_tip_x, main_tip_y), 8, (255, 165, 0), -1)     # 橘色實心
            infer_text = " (Inferred)"
        else:
            # 【精確點：紅色實心】
            # 先畫一個較大的白色圓形作為外框，讓它從藍色空心圓中脫穎而出
            cv2.circle(overlay, (main_tip_x, main_tip_y), 12, (255, 255, 255), -1) # 白色背景遮擋下方的藍圈
            cv2.circle(overlay, (main_tip_x, main_tip_y), 9, (255, 0, 0), -1)       # 紅色實心中心
            infer_text = ""

    # ─── 存檔與 CSV 處理 ───
    binary_mask = np.where(clip_mask > 0, 255, 0).astype(np.uint8)  

    save_status = ""
    # 修改存檔判定：只要有主點（不論是偵測到還是推測出）就存檔
    if main_tip_x is not None:
        base_dir = r"C:\WorkArea\zoey\unet\output_masks"
        for d in ["masks", "overlays", "originals", "csv"]: os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        ts = int(time.time() * 1000)
        mask_path = os.path.join(base_dir, "masks", f"mask_{ts}.png")
        over_path = os.path.join(base_dir, "overlays", f"overlay_{ts}_{label}.png")
        orig_path = os.path.join(base_dir, "originals", f"original_{ts}.png")
        csv_path = os.path.join(base_dir, "csv", "database.csv")
        
        cv2.imwrite(mask_path, binary_mask)
        cv2.imwrite(over_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(orig_path, cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR))
        
        df_row = pd.DataFrame([{
            "StudyInstanceUID": ts, 
            "label": f"CVC - {label}{infer_text}", # CSV 標籤註記是否為推測
            "data": str(all_tip_coords), 
            "clip_path_data": extract_path_points(clip_mask, 15), 
            "ts": ts, 
            "tx": main_tip_x, "ty": main_tip_y, 
            "cx": car_x, "cy": car_y,
            "mask_file": mask_path, 
            "overlay_file": over_path, 
            "original_file": orig_path,
            "is_inferred": 1 if is_inferred else 0
        }])
        
        df_row.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path), encoding='utf-8-sig')
        save_status = (
    f"💾 Saved ({'inferred point' if is_inferred else 'correct point'})\n" #f"💾 已存檔 ({'推測點' if is_inferred else '精確點'})\n"
    f"🔺 Tip no.: {len(all_tip_coords)} 個\n" #f"🔺 Tip 候選點數量：{len(all_tip_coords)} 個\n"
    f"📌 CVC tip position: ({main_tip_x}, {main_tip_y})" #f"📌 最終診斷點：({main_tip_x}, {main_tip_y})"
)
    else:
        save_status = "⚠️ CVC tip not detected" #"⚠️ 無法找到導管末端，未存檔。"

    label_display = {f"CVC - {label}{infer_text}": combined_conf}
    status_msg = (
        f"✅ Analysis Complete!\n" #f"✅ 分析完成!\n"
        f"🎯 Carina Confidence: {car_conf:.2%}\n"
        f"📍 Tip Type: {'Inferred (Orange)' if is_inferred else 'Detected (Red)'}\n"
        f"📊 Combined Score: {combined_conf:.2%}\n"
        f"{save_status}"
    )

    return Image.fromarray(binary_mask), Image.fromarray(overlay), label_display, status_msg

# ─── UI Layout ────────────────────────────────────────────────────────────────
 #[容器層開始] - 這是一張大地圖
def build_carina_page():
    gr.Markdown("# 🏥 CVC Position Auto-Diagnostic App")
    gr.Markdown("Support **real-time label** and **auto-classification saving**")  #gr.Markdown("支援 **即時標籤顯示** 與 **自動分類存檔**")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Upload CVC X-ray image") #img_in = gr.Image(type="pil", label="上傳 CXR 影像")
            with gr.Row():
                sld_thr = gr.Slider(1, 99, value=50, label="Threshold (%)") #sld_thr = gr.Slider(1, 99, value=50, label="閾值 Threshold (%)")
                sld_alp = gr.Slider(10, 90, value=45, label="Alpha (%)") #sld_alp = gr.Slider(10, 90, value=45, label="透明度 Alpha (%)")
            run_btn = gr.Button("🔍 Analyzing", variant="primary") #run_btn = gr.Button("🔍 執行分析", variant="primary")

        with gr.Column(scale=1):
            out_label = gr.Label(label="Diagnose CVC type", num_top_classes=1) #out_label = gr.Label(label="診斷分類結果", num_top_classes=1)
            with gr.Tab("Overlay Result"):
                out_over = gr.Image(label="Overlay")
            with gr.Tab("Binary Mask"):
                out_mask = gr.Image(label="Mask")
            status = gr.Markdown("### Status: Ready.") #status = gr.Markdown("### 狀態: Ready.")

    run_btn.click(
        fn=segment,
        inputs=[img_in, sld_thr, sld_alp],
        outputs=[out_mask, out_over, out_label, status]
    )

    back_btn = gr.Button("⬅ Back", variant="secondary") #back_btn = gr.Button("⬅ 返回大廳", variant="secondary")
    return back_btn

if __name__ == "__main__":
    with gr.Blocks() as demo:
        build_carina_page()
    demo.launch(server_port=12356) # [啟動層] - 把地圖發布到網路上