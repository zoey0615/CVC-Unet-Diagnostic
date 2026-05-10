# -*- coding: utf-8 -*-
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import albumentations
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
# =========================================================
# 1. 執行環境與模型配置
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 1024
ENET_TYPE = "timm-efficientnet-b5"
MODEL_DIR = "C:/WorkArea/zoey/unet/model"
KERNEL_TYPE = "unet++b5_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo_cvc"
N_FOLDS = 5

# 預處理配置
transforms_val = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE)
])

def preprocess(pil_image: Image.Image):
    """影像預處理，回傳 numpy 影像與 torch tensor"""
    image_np = np.array(pil_image.convert("RGB"))
    res = transforms_val(image=image_np)
    # 轉置為 (C, H, W) 並正規化
    image_t = res["image"].astype(np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.tensor(image_t).unsqueeze(0)
    return res["image"], tensor

# =========================================================
# 2. 模型架構定義與載入
# =========================================================
class SegModel(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.seg = smp.UnetPlusPlus(
            encoder_name=backbone, 
            encoder_weights=None, 
            classes=2, 
            activation=None
        )

    def forward(self, x):
        feats = self.seg.encoder(x)
        dec_out = self.seg.decoder(feats)
        return self.seg.segmentation_head(dec_out)

def load_models():
    """載入 5-Fold 集成模型"""
    models = []
    print(f"正在從 {MODEL_DIR} 載入模型...")
    for fold in range(N_FOLDS):
        path = os.path.join(MODEL_DIR, f"{KERNEL_TYPE}_best_fold{fold}.pth")
        if os.path.exists(path):
            m = SegModel(ENET_TYPE)
            m.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
            m.to(DEVICE).eval()
            models.append(m)
            print(f" - Fold {fold} 載入成功")
        else:
            print(f" ⚠️ 找不到 Fold {fold} 權重檔: {path}")
    return models

# 全域模型初始化
MODELS = load_models()

# =========================================================
# 3. 8 組資料路徑設定 (根據你的路徑調整)
# =========================================================
EVAL_GROUPS = [
    {
        "name": "v1",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_130\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_152\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v2",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_129\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_152\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v3",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_148\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_148\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v4",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_146\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_146\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v5",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_145\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_145\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v6",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_150\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_150\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v7",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_151\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_151\original_copied_masks",
        "threshold": 0.5,
    },
    {
        "name": "v8",
        "test_img_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_152\images",
        "test_mask_dir": r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\generated_results\version_152\original_copied_masks",
        "threshold": 0.5,
    },
]

OUTPUT_SUMMARY_CSV = "all_groups_dice_summary.csv"
OUTPUT_DETAIL_DIR = "evaluation_details"

# =========================================================
# 4. 核心運算函式
# =========================================================
def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """計算兩張二值化圖的 Dice Similarity Coefficient"""
    y_true_f = y_true.astype(np.float32).flatten()
    y_pred_f = y_pred.astype(np.float32).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + epsilon) / (np.sum(y_true_f) + np.sum(y_pred_f) + epsilon)
def iou_coefficient(y_true, y_pred, epsilon=1e-6):
    """計算 Intersection over Union (IoU)"""
    y_true_f = y_true.astype(np.float32).flatten()
    y_pred_f = y_pred.astype(np.float32).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)
def find_mask_path(mask_dir, image_filename):
    """尋找對應的標註檔 (支援不同副檔名)"""
    stem = Path(image_filename).stem
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
    for ext in exts:
        p = Path(mask_dir) / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None

def predict_one_image(img_path, threshold=0.5):
    """讀取單張影像並進行模型集成推論 (Ensemble)"""
    pil_img = Image.open(img_path).convert("RGB")
    _, tensor = preprocess(pil_img)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        # 取得 5 個模型的 sigmoid 平均值
        probs = [torch.sigmoid(model(tensor)) for model in MODELS]
        prob_map = torch.stack(probs).mean(0)[0].cpu().numpy()
        
        # CVC Path 位於第 0 個通道
        cvc_path_prob = prob_map[0]
        pred_mask = (cvc_path_prob >= threshold).astype(np.uint8)
    return pred_mask
def cldice_coefficient(y_true, y_pred):
    """
    計算 Centerline Dice (clDice)
    y_true, y_pred: 二值化的 numpy array (0 或 1)
    """
    # 確保輸入是布林或 0/1 格式
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    # 1. 提取骨架 (Skeletonization)
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return 0.0
        
    t_skeleton = skeletonize(y_true)
    p_skeleton = skeletonize(y_pred)
    
    # 2. 計算 Tps (Topology-preserving Sensitivity)
    # 預測的骨架落入真實 Mask 的比例
    tps = np.sum(p_skeleton * y_true) / (np.sum(p_skeleton) + 1e-6)
    
    # 3. 計算 Vcl (Topology-preserving Precision)
    # 真實的骨架落入預測 Mask 的比例
    vcl = np.sum(t_skeleton * y_pred) / (np.sum(t_skeleton) + 1e-6)
    
    # 4. 計算 clDice
    return 2.0 * tps * vcl / (tps + vcl + 1e-6)
# =========================================================
# 5. 批次執行主程式
# =========================================================
def run_evaluation():
    os.makedirs(OUTPUT_DETAIL_DIR, exist_ok=True)
    summary_data = []

    for group in EVAL_GROUPS:
        name = group["name"]
        img_dir = Path(group["test_img_dir"])
        mask_dir = Path(group["test_mask_dir"])
        thresh = group["threshold"]

        if not img_dir.exists():
            print(f"⚠️ 跳過組別 {name}：找不到影像資料夾 {img_dir}")
            continue

        print(f"\n🚀 正在評估組別：{name}")
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        group_results = []

        for filename in tqdm(image_files, desc=f"Processing {name}"):
            img_path = img_dir / filename
            mask_path = find_mask_path(mask_dir, filename)

            if mask_path is None:
                continue

            try:
                # 1. 執行預測
                pred_mask = predict_one_image(str(img_path), threshold=thresh)
                
                # 2. 讀取真實遮罩
                true_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # 調整尺寸為 1024 以進行公平對比
                true_mask = cv2.resize(true_mask_raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                true_mask = (true_mask > 127).astype(np.uint8)

                dice_score = dice_coefficient(true_mask, pred_mask)
                iou_score = iou_coefficient(true_mask, pred_mask)
                cldice_score = cldice_coefficient(true_mask, pred_mask) # <--- 新增
                
                group_results.append({
                    "filename": filename,
                    "dice_score": dice_score,
                    "iou_score": iou_score,
                    "cldice_score": cldice_score # <--- 新增
                })

            except Exception as e:
                print(f"❌ 處理 {filename} 時發生錯誤: {e}")

        # 4. 儲存該組詳細資料與計算平均值
        if group_results:
            df_detail = pd.DataFrame(group_results)
            mean_dice = df_detail["dice_score"].mean()
            mean_iou = df_detail["iou_score"].mean()
            mean_cldice = df_detail["cldice_score"].mean() # <--- 新增
            
            detail_path = os.path.join(OUTPUT_DETAIL_DIR, f"{name}_metrics_details.csv")
            df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
            
            summary_data.append({
                "Group_Name": name,
                "Image_Count": len(df_detail),
                "mDice": mean_dice,
                "mIoU": mean_iou,
                "mclDice": mean_cldice # <--- 新增
            })
            print(f"✅ {name} 評估完成。mDice: {mean_dice:.4f}, mclDice: {mean_cldice:.4f}")
    # 5. 產出總表
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"\n📊 全部組別評估完成！總表已輸出至：{OUTPUT_SUMMARY_CSV}")
        print(df_summary)

if __name__ == "__main__":
    if not MODELS:
        print("❌ 錯誤：未載入任何模型，請檢查模型路徑。")
    else:
        run_evaluation()