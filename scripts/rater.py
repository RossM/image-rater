import random
import json
import gradio as gr

from modules import (devices, script_callbacks, scripts, shared)
from modules.ui import create_output_panel, create_refresh_button

from PIL import Image
from pathlib import Path

root_path = Path(__file__).parents[3]
image_rater_path = root_path / 'image_rater'
log_path = image_rater_path / 'log'

def generate_comparison(state: dict):
    filepaths = state['files']
    
    if len(filepaths) < 2:
        return None, None
        
    random.shuffle(filepaths)
    
    max_size = 800
    
    outputs = []
    state['current_comparison'] = selected = filepaths[0:2]
    for filepath in selected:
        image = Image.open(filepath)
        padded_image = Image.new(mode="RGB", size=(max_size, max_size), color="white")
        image.thumbnail((max_size, max_size))
        padded_image.paste(image, box=((max_size - image.width) // 2, (max_size - image.height) // 2))
        outputs.append(padded_image)
        
    return outputs

def log_and_generate(result, state: dict):
    print(f"result={result}")
    print(f"log_path={log_path}")
    
    log_path.mkdir(parents=True, exist_ok=True)

    with open(log_path / 'default.json', 'a') as f:
        f.write(json.dumps({
            'files': state['current_comparison'],
            'choice': result,
        }) + "\n")
    
    return generate_comparison(state)

def load_images(images_path: str, state: dict, progress: gr.Progress = gr.Progress()):
    if not images_path:
        yield ["You must provide the path to a directory with image files", None, None]
        return
    
    yield [f"Loading images from {images_path}", None, None]
    
    path = Path(images_path)
    
    filepaths = list(path.glob('*'))
    state['files'] = []
    state['current_comparison'] = []
    state['loading'] = True
    
    if len(filepaths) == 0:
        yield [f"No files found at {images_path}, check that you have the correct directory", None, None]
        return
    
    for filepath in progress.tqdm(filepaths):
        try:
            image = Image.open(filepath)
            state['files'].append(str(filepath))
        except Exception as e:
            continue
    
    state['loading'] = False
    outputs = generate_comparison(state)
    yield [f"Loaded {len(state['files'])} images from {images_path}", *outputs]
    
def clear_images(state: dict, progress: gr.Progress = gr.Progress()):
    state['files'] = []
    state['current_comparison'] = []
    return ["No images loaded", None, None]

def on_ui_tabs():
    with gr.Blocks() as ui_tab:
        with gr.Tab(label="Rate"):
            with gr.Accordion(label="Config"):
                with gr.Row():
                    with gr.Column():
                        images_path = gr.Textbox(label="Images path", scale=1)
                        with gr.Row():
                            load_images_btn = gr.Button(value="Load", scale=1)
                            cancel_btn = gr.Button(value="Unload", scale=1)
                    status_area = gr.Textbox(label="Status", interactive=False, scale=2, lines=3)
            gr.HTML("Pick the better image!", elem_id="imagerater_calltoaction")
            with gr.Row(elem_id="imagerater_image_row"):
                with gr.Column():
                    left_img = gr.Image(interactive=False, container=False)
                    left_btn = gr.Button(value="Pick", container=False)
                    left_val = gr.State(value=0)
                with gr.Column():
                    right_img = gr.Image(interactive=False)
                    right_btn = gr.Button(value="Pick")
                    right_val = gr.State(value=1)
            skip_btn = gr.Button(value="Skip", elem_id="imagerater_skipbutton")
            state = gr.State(value={})
    
        load_event = load_images_btn.click(load_images, inputs=[images_path, state], outputs=[status_area, left_img, right_img])
        cancel_btn.click(clear_images, cancels=[load_event], inputs=[state], outputs=[status_area, left_img, right_img])
        
        skip_btn.click(generate_comparison, inputs=[state], outputs=[left_img, right_img])
        left_btn.click(log_and_generate, inputs=[left_val, state], outputs=[left_img, right_img])
        right_btn.click(log_and_generate, inputs=[right_val, state], outputs=[left_img, right_img])
    
    return (ui_tab, "Image Rater", "imagerater"),


script_callbacks.on_ui_tabs(on_ui_tabs)
