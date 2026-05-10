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

def get_ordered_path(clip_mask):
    # 1. 骨架化：將粗大的導管縮減為 1 像素寬的線
    skeleton = skeletonize(clip_mask > 0).astype(np.uint8)
    
    # 2. 找到骨架上的點
    points = np.column_stack(np.where(skeleton > 0)) # 回傳 [[y, x], ...]
    
    # 3. 如果點太少，直接回傳
    if len(points) < 10: return points
    
    # 4. 排序點 (簡單做法：以 y 座標排序，假設導管是縱向走)
    # 進階做法可以使用鄰近演算法 (Nearest Neighbor) 來重排序列
    ordered_points = points[points[:, 0].argsort()] 
    return ordered_points
def detect_carina(image_np):
    h, w = image_np.shape[:2]
    
    # 預處理
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    input_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    fallback_x, fallback_y = int(0.5 * w), int(0.42 * h)

    try:
        result = carinanet.predict_carina_ett(input_image)
        car = result.get("carina", None)
        
        if car is None:
            return fallback_x, fallback_y

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
        return final_x, final_y

    except Exception as e:
        print(f"❌ 偵測失敗: {e}")
        return fallback_x, fallback_y
#計算導管尖端（Tip）與解剖基準點（Carina）的相對位置，並根據醫學臨床準則給出診斷標籤。
def check_malposition(tip_y, car_y, tip_x, car_x, is_looping_up=False):
    if is_looping_up:
        return "Abnormal (Upward Malposition)", (0, 0, 255)
    if tip_y is None or car_y is None: return "Unknown", (128, 128, 128) #如果其中一個點沒抓到（None），系統會回傳 "Unknown" 並顯示灰色 (128, 128, 128)。
    y_diff = tip_y - car_y #縱向位移：決定深淺(>0導管在carina下方)
    x_diff = abs(tip_x - car_x) #橫向位移：決定是否誤入血管

    if x_diff > 180: #如果橫向位移>180像素（約 6 公分）
        return "Abnormal (Wrong Vessel)", (0, 0, 255) #回傳字串和RGB 數值
    if 0 <= y_diff <= 150: #如果縱向位移介於0~150(約為 Carina 下方4.5 ~5$ 公分)之間(https://www.droracle.ai/articles/1061589/is-a-right-central-venous-catheter-with-the-tip)
        return "Normal", (0, 255, 0)  
    elif (150 < y_diff <= 250) or (-50 <= y_diff < 0):
        return "Borderline", (0, 165, 255) 
    else:
        return "Abnormal (Too Deep/High)", (0, 0, 255)

def segment(pil_image, threshold_pct: float, alpha_pct: float):   #門檻值百分比、透明度百分比
    if not pil_image: return None, None, None, "⚠️ No image" #如果沒有影像回傳空值（Mask, Overlay, Label, Status）
    threshold, alpha = threshold_pct/100.0, alpha_pct/100.0 #將百分比轉為小數
    original_np, tensor = preprocess(pil_image) #將原始的圖片檔案格式，轉換為模型可以運算的數學張量(B,H, W, C)->(B, C,H, W)
#推論（Inference，即現在的診斷階段）不需計算梯度。
    with torch.no_grad():
        if MODELS: #UNet++ Models
        #輪流叫出每一個模型 m，進行預測，將一組原始數值（稱為 Logits）壓縮到 $[0, 1]$ 之間
        #將上述所有模型產生的機率圖垂直疊放在一起。進行平均計算((3, 1, 2, 1024, 1024)->(1, 2, 1024, 1024)->(2, 1024, 1024)
            prob_map = torch.stack([m(tensor.to(DEVICE)).sigmoid() for m in MODELS]).mean(0)[0].cpu().numpy()
        else:
            prob_map = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE)) #建立一個數值全部都是0的 NumPy 矩陣。
    
    car_x, car_y = detect_carina(original_np) #偵測 Carina 位置，整合 CLAHE 增強、模型預測、座標限制與解剖比例備案。

    # 1. 提取尖端遮罩 (通道 1)
    tip_mask_raw = (prob_map[1] >= threshold).astype(np.uint8) * 255 #逐一檢查每個像素，回傳TRUE、FALSE->1、0， 乘以 255。
    clip_mask = (prob_map[0] >= threshold).astype(np.uint8) * 255 # 整個 CVC 導管的路徑
    
    # 2. 找出所有尖端連通域 (不只取最大的一個)
    contours, _ = cv2.findContours(tip_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Retrieve(檢索/取回) External(外部的) (Contours Only)、(簡單鏈式逼近演算法): 使用外部輪廓檢索(RETR_EXTERNAL)與鏈碼逼近(CHAIN_APPROX_SIMPLE)提取尖端候選區域
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5] #計算輪廓 c 所包圍的面積（像素個數） [ 想要留下來的東西 | 來源 | 過濾條件 ]
    num_tips = len(valid_contours) #有效輪廓數量。
    
    clean_tip_mask = np.zeros_like(tip_mask_raw) #利用 NumPy 建立一個全為 0（黑色）的矩陣。與尖端相同大小
    all_tip_coords = [] # 用來存所有偵測到的尖端中心點

    for cnt in valid_contours: #遍歷有效輪廓清單
        # 把所有有效的尖端填滿畫到 clean_tip_mask 上
        cv2.drawContours(clean_tip_mask, [cnt], -1, 255, -1)
        
        # 計算每個連通域的中心點
        M = cv2.moments(cnt) #計算輪廓 cnt 的所有空間矩（Spatial Moments）。空間權重*像素值
        if M["m00"] != 0: #檢查零階矩 $m_{00}$（即面積）是否不等於零。
            cx = int(M["m10"] / M["m00"]) #總力矩」除以「總重量」，得到的結果就是這塊形狀的 「平衡點（質心）」。
            cy = int(M["m01"] / M["m00"])
            all_tip_coords.append([cx, cy])

# 3. 診斷邏輯
    main_tip_x, main_tip_y = None, None 
    is_looping_up = False # 預設方向正確
    path_points = get_ordered_path(clip_mask)

    if len(path_points) > 20 and len(all_tip_coords) > 0:
        tail_segment = path_points[-15:] # 稍微增加觀察樣本
        y_start_tail = tail_segment[0][0] 
        y_end_tail = tail_segment[-1][0]  
        
        # 判定方向：y 越小越高，所以終點 y 小於 起點 y 代表正在往上
        if y_end_tail < y_start_tail - 5: # 增加 5 像素的容錯
            is_looping_up = True
    
        # 尋找與導管末端最接近的 Tip
        last_path_point = (path_points[-1][1], path_points[-1][0]) 
        best_tip = min(all_tip_coords, key=lambda p: np.linalg.norm(np.array(p) - np.array(last_path_point)))
        main_tip_x, main_tip_y = best_tip[0], best_tip[1]
    elif all_tip_coords:
        # 備案邏輯：如果路徑太短，回歸 y 座標最大法
        best_tip = max(all_tip_coords, key=lambda p: p[1])
        main_tip_x, main_tip_y = best_tip[0], best_tip[1]

    # 執行統一診斷
    label, color_bgr = check_malposition(main_tip_y, car_y, main_tip_x, car_x, is_looping_up)
    overlay = make_overlay(original_np, prob_map, threshold, alpha)
    
    # 繪製 Carina
    if car_x and car_y:
        cv2.circle(overlay, (car_x, car_y), 10, (255, 255, 255), -1)
        cv2.circle(overlay, (car_x, car_y), 7, (0, 255, 255), -1) 

# 1. 先用藍色畫出「所有」偵測到的尖端候選點 (作為底色)
    for tx, ty in all_tip_coords:
        cv2.circle(overlay, (tx, ty), 10, (255, 255, 255), -1) # 白邊
        cv2.circle(overlay, (tx, ty), 7, (255, 0, 0), -1)    # 藍色中心 (BGR: 255, 0, 0)

    # 2. 針對與 Carina 比較的那一個「主尖端」，覆蓋上不同的顏色（例如：紅色或紫色）
    if main_tip_x is not None and main_tip_y is not None:
        # 這裡我們用紅色 (BGR: 0, 0, 255) 來標註最終診斷用的點
        cv2.circle(overlay, (main_tip_x, main_tip_y), 11, (255, 255, 255), -1) # 稍微大一點點的白邊
        cv2.circle(overlay, (main_tip_x, main_tip_y), 8, (0, 0, 255), -1)      # 紅色中心

    mask_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    mask_rgb[:, :, 1] = clip_mask  
    mask_rgb[:, :, 0] = clean_tip_mask  

    save_status = ""
    # 修改存檔條件：只要偵測到尖端就存檔，不論 1 個還是多個
    if num_tips >= 1:
        base_dir = r"C:\WorkArea\zoey\unet\output_masks"
        for d in ["masks", "overlays", "originals", "csv"]: os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        ts = int(time.time() * 1000) #Python 的時間模組，回傳自 1970 年 1 月 1 日 00:00:00 (UTC) 以來的總秒數。
        mask_path = os.path.join(base_dir, "masks", f"mask_{ts}.png")
        over_path = os.path.join(base_dir, "overlays", f"overlay_{ts}_{label}.png")
        orig_path = os.path.join(base_dir, "originals", f"original_{ts}.png")
        csv_path = os.path.join(base_dir, "csv", "database.csv")
        
        cv2.imwrite(mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(over_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(orig_path, cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR))
        full_data_path = str(all_tip_coords)
        full_clip_path = extract_path_points(clip_mask, sample_step=15) 

        # 3. 修改 DataFrame 內容
        df_row = pd.DataFrame([{
            "StudyInstanceUID": ts, 
            "label": f"CVC - {label}", 
            "data": full_data_path,        # 這裡是尖端點
            "clip_path_data": full_clip_path, # 這是新增的欄位：整條導管的座標
            "ts": ts, 
            "tx": main_tip_x, "ty": main_tip_y, 
            "cx": car_x, "cy": car_y,
            "mask_file": mask_path, 
            "overlay_file": over_path, 
            "original_file": orig_path,
            "num_tips_detected": num_tips 
        }])
        
        # 存檔指令保持不變
        df_row.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path), encoding='utf-8-sig')
        save_status = f"💾 已偵測到 {num_tips} 個尖端並存檔。"
    else:
        save_status = "⚠️ 未偵測到尖端，未存檔。"

    label_display = {f"CVC - {label}": 1.0}
    return Image.fromarray(mask_rgb), Image.fromarray(overlay), label_display, f"✅ 分析完成!\n{save_status}"

# ─── UI Layout ────────────────────────────────────────────────────────────────
 #[容器層開始] - 這是一張大地圖
def build_carina_page():
    gr.Markdown("# 🏥 CVC Position Auto-Diagnostic App")
    gr.Markdown("支援 **即時標籤顯示** 與 **自動分類存檔**")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="上傳 CXR 影像")
            with gr.Row():
                sld_thr = gr.Slider(1, 99, value=50, label="閾值 Threshold (%)")
                sld_alp = gr.Slider(10, 90, value=45, label="透明度 Alpha (%)")
            run_btn = gr.Button("🔍 執行分析", variant="primary")

        with gr.Column(scale=1):
            out_label = gr.Label(label="診斷分類結果", num_top_classes=1)
            with gr.Tab("Overlay Result"):
                out_over = gr.Image(label="Overlay")
            with gr.Tab("Binary Mask"):
                out_mask = gr.Image(label="Mask")
            status = gr.Markdown("### 狀態: Ready.")

    run_btn.click(
        fn=segment,
        inputs=[img_in, sld_thr, sld_alp],
        outputs=[out_mask, out_over, out_label, status]
    )

    back_btn = gr.Button("⬅ 返回大廳", variant="secondary")
    return back_btn

if __name__ == "__main__":
    with gr.Blocks() as demo:
        build_carina_page()
    demo.launch(server_port=12356) # [啟動層] - 把地圖發布到網路上