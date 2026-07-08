import sys
import os

# =========================
# 路徑設定
# =========================
DIFFUSION_DIR = r"C:\WorkArea\zoey\tar\media\Siamese-Diffusion"
UNET_DIR = r"C:\WorkArea\zoey\unet"

# 避免舊 config cache
if "config" in sys.modules:
    del sys.modules["config"]

# Siamese-Diffusion 放最前面
sys.path.insert(0, DIFFUSION_DIR)

# UNET 路徑
sys.path.insert(0, UNET_DIR)

# 切換工作目錄
os.chdir(DIFFUSION_DIR)

# 測試目前 config 是否正確
import config
print("目前 config 路徑:", config.__file__)

# =========================
# Import 模組
# =========================
import gradio as gr
import carina
import gradiop
def show_home():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def show_carina():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def show_gen():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="CVC 系統大廳", theme=gr.themes.Soft()) as demo:
    with gr.Group(visible=True) as home_page:
        gr.Markdown("# 🏥 CVC Entrance")
        gr.Markdown("請選擇要進入的功能頁面")
        with gr.Row():
            btn_carina = gr.Button("進入 Carina 偵測", variant="primary")
            btn_gen = gr.Button("進入 CVC 生成", variant="primary")

    with gr.Group(visible=False) as carina_page:
        back_btn1 = carina.build_carina_page()

    with gr.Group(visible=False) as gen_page:
        back_btn2 = gradiop.build_gradiop_page()

    btn_carina.click(show_carina, outputs=[home_page, carina_page, gen_page])
    btn_gen.click(show_gen, outputs=[home_page, carina_page, gen_page])

    back_btn1.click(show_home, outputs=[home_page, carina_page, gen_page])
    back_btn2.click(show_home, outputs=[home_page, carina_page, gen_page])

demo.launch(server_name="0.0.0.0", server_port=12356,share=True)