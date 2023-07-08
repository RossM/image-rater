import gradio as gr

from modules import (devices, script_callbacks, scripts, shared)
from modules.ui import create_output_panel, create_refresh_button

def on_ui_tabs():
    with gr.Blocks() as ui_tab:
        with gr.Accordion(label="Config"):
            with gr.Row():
                with gr.Column():
                    images_path = gr.Textbox(label="Images path", scale=1)
                    with gr.Row():
                        load_images = gr.Button(value="Load", scale=1)
                        clear_images = gr.Button(value="Clear", scale=1)
                status_area = gr.Textbox(label="Status", interactive=False, scale=2, lines=3)
        gr.HTML("Pick the better image!", elem_id="imagerater_calltoaction")
        with gr.Row(elem_id="imagerater_image_row"):
            with gr.Column():
                gr.Image(interactive=False, shape=(600,600), container=False)
                gr.Button(value="Pick", shape=(600,600), container=False)
            with gr.Column():
                gr.Image(interactive=False)
                gr.Button(value="Pick")
        gr.Button(value="Skip", elem_id="imagerater_skipbutton")
    
    return (ui_tab, "Image Rater", "imagerater"),


script_callbacks.on_ui_tabs(on_ui_tabs)
