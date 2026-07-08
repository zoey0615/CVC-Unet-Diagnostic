import os
import torch
import numpy as np
import gradio as gr
import random
import cv2
import json
from PIL import Image
from datetime import datetime
from share import * #依顯存調整避免oom
from cldm.model import create_model, load_state_dict #controlnet +擴散模型結構,載入權重
from ldm.models.diffusion.ddim import DDIMSampler
from scipy.interpolate import splprep, splev  
import torchvision.transforms.functional as TF
import traceback


# ==路徑與配置
BASE_PATH = r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion"
CKPT_PATH = r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\merged_model_version152 .ckpt" 
CONFIG_PATH = r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion\models\cldm_v15.yaml" #Stable Diffusion+ControlNet(文字紀錄)
# ----------------
HEALTHY_IMG_DIR = os.path.join(BASE_PATH, "data/healthy_xrays")
SAVE_DIR = os.path.join(BASE_PATH, "gradio_outputs_auto")

# 建立子資料夾
IMG_SAVE_DIR = os.path.join(SAVE_DIR, "images")
MASK_SAVE_DIR = os.path.join(SAVE_DIR, "masks")
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

# ✨ 記憶進度檔案路徑 (放在根目錄)
LOG_FILE = os.path.join(SAVE_DIR, "processed_history.json")

# ============================================================
# 核心邏輯
# ============================================================
def get_model():
    model = create_model(CONFIG_PATH).cpu()
    model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'), strict=False)
    model.to("cuda:0")
    model.eval()
    return model

model = get_model()
ddim_sampler = DDIMSampler(model)

class ProgressManager:
    """負責記憶已經處理過的 ID，避免重複"""
    @staticmethod
    def get_processed_ids():
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    @staticmethod
    def save_id(img_id):
        processed = list(ProgressManager.get_processed_ids())
        if img_id not in processed:
            processed.append(img_id)
            with open(LOG_FILE, 'w') as f:
                json.dump(processed, f)


class CVCProcessor:
    @staticmethod #不須實例化的靜態方法
    def transform_coords(coords, mode, intensity):
        new_coords = [np.array(p) for p in coords]
        if len(new_coords) < 2: return new_coords
        
        if mode == "縮短 (Too Shallow)":
            # 靈敏度調整：每 5 單位強度縮短一個點
            cutoff = max(2, len(new_coords) - int(intensity / 5)) 
            new_coords = new_coords[:cutoff]
        elif mode == "增長 (Too Deep)":
            p_last, p_prev = new_coords[-1], new_coords[-2]
            direction = p_last - p_prev
            unit_vec = direction / (np.linalg.norm(direction) + 1e-6)
            # 步長設為 5 像素，確保延伸路徑平滑
            for i in range(1, int(intensity / 5) + 1):
                new_coords.append(p_last + unit_vec * (i * 5))
                
        return [tuple(p.astype(int)) for p in new_coords]

    @staticmethod
    def draw_mask(coords):
        # 1. 建立空白遮罩
        mask = np.zeros((512, 512), dtype=np.uint8)
        points = np.array(coords, dtype=np.float32)
        
        # 2. B-spline 平滑插值 (參考妳提供的邏輯)
        if len(points) > 3:
            try:
                # s=0 表示曲線必須經過所有點
                tck, u = splprep([points[:, 0], points[:, 1]], s=0)
                u_fine = np.linspace(0, 1, 200) 
                x_fine, y_fine = splev(u_fine, tck)
                pts_smooth = np.vstack((x_fine, y_fine)).T.astype(np.int32)
            except:
                # 如果插值失敗（例如點重疊），退回到直線畫法
                pts_smooth = points.astype(np.int32)
        else:
            pts_smooth = points.astype(np.int32)

        # 3. 繪製平滑曲線 
        cv2.polylines(mask, [pts_smooth], isClosed=False, color=255, 
                      thickness=4, lineType=cv2.LINE_AA)

        # 4. 形態學與模糊處理 (讓邊緣更有肉感，不會太利)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.0)
        
        return mask
    @staticmethod
    def _apply_local_enhance(image_np, mask_np):
        """
        保留底圖原始色彩與質感，僅在 mask 位置疊加強對比特徵
        """
        if mask_np.shape[:2] != image_np.shape[:2]:
            # 將 mask 縮放到與底圖相同的 (H, W)
            mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        # 1. 確保 mask 是二值化且稍微膨脹，確保模型能感應到邊緣
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        strong_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # 2. 建立一個「高亮導管」層 (例如純白色)
        highlight = np.full_like(image_np, 255) 
        
        # 3. 使用 Alpha 混合：在 mask 的地方放 highlight，其他地方保留 image_np
        alpha = strong_mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1) # 擴展成 (512, 512, 1) 以進行廣播運算
        if len(alpha.shape) == 2:
            alpha = np.expand_dims(alpha, axis=-1)
        # 混合公式：結果 = 原始圖 * (1 - mask) + 高亮圖 * mask
        # 這裡我們稍微調低權重，不要 100% 死白，保留一點底層組織紋理 (0.9 權重)
        output = (image_np.astype(float) * (1.0 - alpha * 0.9) + 
                  highlight.astype(float) * (alpha * 0.9))
        
        return output.astype(np.uint8)
    
def load_reference_image(file_obj):
    """從 JSON 抓一個 ID，然後去資料夾找對應的 X 光片墊在畫布下"""
    if file_obj is None: return None
    try:
        with open(file_obj.name, 'r') as f:
            data = json.load(f)
        
        # 隨機挑選一筆
        selected = random.choice(data) if isinstance(data, list) else data
        img_id = selected.get('name', 'unknown')
        img_path = os.path.join(HEALTHY_IMG_DIR, f"{img_id}.png")
        if not os.path.exists(img_path):
            # 如果找不到特定的，就隨機從健康底圖資料夾抓一張
            bg_files = [f for f in os.listdir(HEALTHY_IMG_DIR) if f.endswith(('.png', '.jpg'))] #從一個資料夾中，精確地挑選出所有圖片檔案（.png 或 .jpg），並存成一個清單
            img_path = os.path.join(HEALTHY_IMG_DIR, random.choice(bg_files))

        ref_img = Image.open(img_path).convert('RGB').resize((512, 512))
        
        # Gradio ImageMask 的 value 格式：把影像放在 background
        return {"background": np.array(ref_img), "layers": [], "composite": None}
    except Exception as e:
        print(f"底圖載入失敗: {e}")
        return None


@torch.no_grad()
def process_cvc(hand_data, json_file, mode, intensity, label_type, use_bg,ddim_steps, scale, seed):
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        current_seed = int(seed) 
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        # ============================================================
        # 1. 遮罩 (input_mask) 與 底圖 (base_image) 提取邏輯
        # ============================================================
        aug_mask = np.zeros((512, 512), dtype=np.uint8)
        target_cv2_rgb = None
        is_hand_drawn = False
        source_info = "Unknown source"

        if isinstance(hand_data, dict):
            # A. 提取底圖 (base_image)
            if hand_data.get("background") is not None:
                bg = hand_data["background"]
                # 處理 RGBA 轉 RGB
                target_cv2_rgb = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB) if bg.shape[-1] == 4 else bg
                source_info = "Background image defined by user"
            
            # B. 提取手繪筆跡 (從 layers 或 composite 提取 Alpha 通道)
            if "layers" in hand_data and len(hand_data["layers"]) > 0:
                for layer in hand_data["layers"]:
                    layer_np = np.array(layer)
                    if layer_np.shape[-1] == 4: 
                        alpha = layer_np[:, :, 3]
                        if np.max(alpha) > 0:
                            # --- 關鍵修正：檢查 alpha 尺寸是否為 512 ---
                            if alpha.shape[0] != 512 or alpha.shape[1] != 512:
                                # 如果尺寸不對 (例如是 24)，建立一個空的 512 畫布再貼上去
                                alpha = cv2.resize(alpha, (512, 512), interpolation=cv2.INTER_NEAREST)
                            
                            aug_mask[alpha > 0] = 255
                            is_hand_drawn = True
                            break
            
            if not is_hand_drawn and hand_data.get("composite") is not None:
                comp = np.array(hand_data["composite"])
                if comp.shape[-1] == 4:
                    alpha = comp[:, :, 3]
                    if np.max(alpha) > 0:
                        aug_mask[alpha > 0] = 255
                        is_hand_drawn = True

        # C. 處理 JSON 自動模式 (僅在手繪為空時觸發)
        json_id_found = None
        if not is_hand_drawn and json_file is not None:
            try:
                with open(json_file.name, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    processed_ids = ProgressManager.get_processed_ids()
                    pool = [d for d in data if d['name'] not in processed_ids]
                    if pool:
                        selected = random.choice(pool)
                        json_id_found = selected['name']
                        # 執行變形算法
                        transformed = CVCProcessor.transform_coords(selected['coords'], mode, intensity)
                        aug_mask = CVCProcessor.draw_mask(transformed)
                        source_info = f"JSON 自動變形 ({json_id_found})"
            except Exception as e:
                print(f"JSON 處理失敗: {e}")

        # 防呆：無遮罩不生成
        if np.max(aug_mask) == 0:
            return None, "❌ 找不到遮罩！請在畫布上手繪或提供有效 JSON。"

        # ============================================================
        # 2. 封裝條件字典 (Cond & UC) - 修正通道數與維度
        # ============================================================
        
        # --- A. 準備 Mask Tensor (修正 1 通道轉 3 通道問題) ---
        aug_mask_rgb = cv2.cvtColor(aug_mask, cv2.COLOR_GRAY2RGB)
        # 強制 resize 為 512x512
        aug_mask_rgb = cv2.resize(aug_mask_rgb, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        control_mask = torch.from_numpy(aug_mask_rgb).float().to(device) / 255.0
        control_mask = control_mask.permute(2, 0, 1).unsqueeze(0)
        
        # --- B. 準備底圖 Tensor ---
        if not use_bg:
            target_cv2_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 強制底圖 resize 為 512x512
        target_cv2_rgb = cv2.resize(target_cv2_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        enhanced_bg = CVCProcessor._apply_local_enhance(target_cv2_rgb, aug_mask)
        # 再次確保 enhance 後的結果也是 512
        enhanced_bg = cv2.resize(enhanced_bg, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        control_image = (torch.from_numpy(enhanced_bg).float().to(device) / 127.5 - 1.0)
        control_image = control_image.permute(2, 0, 1).unsqueeze(0)

        # --- C. 組裝正向與負向條件 (適配妳提到的 UC 零張量邏輯) ---
        full_prompt = f"clinical chest x-ray, central venous catheter, {label_type}, sharp, high contrast"
        cond_dict = {
            "c_crossattn": [model.get_learned_conditioning([full_prompt])],
            "c_concat_mask": [control_mask], #除了沒底圖沒損失、沒底圖有損失"c_concat"其他都要用"c_concat_mask"
            "c_concat_image": [control_image]
        }

        # 無條件引導時，遮罩與底圖設為零張量，讓模型自由發揮
        uc_dict = {
            "c_crossattn": [model.get_learned_conditioning(["blur, low quality, distorted, artifacts"])],
            "c_concat_mask": [torch.zeros_like(control_mask)], #除了沒底圖沒損失、沒底圖有損失"c_concat"其他都要用"c_concat_mask"
            "c_concat_image": [torch.zeros_like(control_image)]
        }

        # ============================================================
        # 3. 執行採樣與儲存
        # ============================================================
        with model.ema_scope():
            samples, _ = ddim_sampler.sample(
                ddim_steps, 1, (4, 64, 64), cond_dict, 
                unconditional_guidance_scale=scale, 
                unconditional_conditioning=uc_dict, 
                eta=0.0
            )

        # 解碼影像
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        result_img = (x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

        # --- 儲存區塊 (同時儲存影像與遮罩) ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_id = json_id_found if json_id_found else "manual"
        
        # 存影像
        img_filename = f"{save_id}_{ts}.png"
        Image.fromarray(result_img).save(os.path.join(IMG_SAVE_DIR, img_filename))
        
        # 存遮罩 (妳要求的 Ground Truth)
        mask_filename = f"{save_id}_{ts}_mask.png"
        Image.fromarray(aug_mask).save(os.path.join(MASK_SAVE_DIR, mask_filename))
        
        if json_id_found:
            ProgressManager.save_id(json_id_found)

        return result_img, f"✅ Generate Successfully!\nSave image in: {img_filename}\nSave mask in: {mask_filename}\nSource: {source_info}"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"❌ Errors: {str(e)}"
   
def auto_update_label(file_obj, mode, intensity):
    """根據變形模式自動切換類別標籤"""
    if mode == "Keep":
        return "CVC - Normal", "💡 Mode: Normal"
    elif mode == "Extend":
        return "CVC - Abnormal", "💡 Mode: Extend CVC Length"
    elif mode == "Shorten":
        return "CVC - Abnormal", "💡 Mode: Shorten CVC Length"
    return "CVC - Abnormal", ""

def handle_custom_upload(file):
    """處理使用者自行上傳的 X-ray 圖檔作為手繪底圖"""
    if file is None:
        return None
    try:
        # 開啟圖片並標準化尺寸
        img = Image.open(file.name).convert('RGB').resize((512, 512))
        # 回傳給 ImageMask 的格式：背景是你的圖，圖層清空
        return {"background": np.array(img), "layers": [], "composite": None}
    except Exception as e:
        print(f"自定義底圖載入失敗: {e}")
        return None
# ============================================================
# Gradio 介面 (簡化版)
# ============================================================
def build_gradiop_page():
    gr.Markdown("# CVC Medical Image Generation ")

    with gr.Row():
        with gr.Column():
            with gr.Tab("📁 JSON Mode"):
                file_input = gr.File(label="Upload JSON")
                mode_select = gr.Radio(["Keep", "Deep", "Shallow"], label="Depth Mode", value="Deep")
                intensity = gr.Slider(0, 150, 50, label="Scaling Mode")
                logic_hint = gr.Markdown("💡 System will determine the CVC type based on uploaded file")

            with gr.Tab("🎨 Draw Mode"):
                with gr.Row():
                    load_bg_btn = gr.Button("🖼️ Randomly load stored X-ray image")
                    upload_bg_file = gr.UploadButton("📤 Upload selected X-ray image", file_types=["image"])# 新增：讓使用者直接上傳自己的 X-ray

                img_input = gr.ImageMask(
                    label="Draw CVC trajectory",
                    type="numpy",
                    height=512,
                    width=512,
                    sources=["upload"],
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=2),
                    value={"background": np.zeros((512, 512, 3), dtype=np.uint8), "layers": [], "composite": None}
                )

            label_type = gr.Dropdown(["CVC - Normal", "CVC - Abnormal", "CVC - Borderline"], value="CVC - Abnormal", label="CVC Type")
            use_bg_checkbox = gr.Checkbox(label="Use Background", value=True)
            run_btn = gr.Button("Generate Next 🚀", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Generative Image")
            output_info = gr.Textbox(label="Image Information", interactive=False)

    logic_inputs = [file_input, mode_select, intensity]
    file_input.change(fn=auto_update_label, inputs=logic_inputs, outputs=[label_type, logic_hint])
    mode_select.change(fn=auto_update_label, inputs=logic_inputs, outputs=[label_type, logic_hint])
    intensity.change(fn=auto_update_label, inputs=logic_inputs, outputs=[label_type, logic_hint])

    upload_bg_file.upload(fn=handle_custom_upload, inputs=[upload_bg_file], outputs=[img_input])
    load_bg_btn.click(fn=load_reference_image, inputs=[file_input], outputs=[img_input])

    run_btn.click(
        fn=process_cvc,
        inputs=[img_input, file_input, mode_select, intensity, label_type, use_bg_checkbox, gr.State(50), gr.State(9.0), gr.State(42)],
        outputs=[output_img, output_info]
    )

    back_btn = gr.Button("⬅ Return Home", variant="secondary")
    return back_btn
custom_css = """
.gradio-container {
    font-family: 'Microsoft JhengHei', sans-serif;
}
"""
if __name__ == "__main__":
    with gr.Blocks() as demo:
        build_gradiop_page()
    demo.launch(server_port=12356)