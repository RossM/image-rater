import random
import time
import json
import gradio as gr

from modules import (devices, script_callbacks, scripts, shared, call_queue)
from modules.ui import create_output_panel, create_refresh_button

from PIL import Image
from pathlib import Path

from scripts.modules.embedding import EmbeddingCache

root_path = Path(__file__).parents[3]
image_rater_path = root_path / 'image_rater'
log_path = image_rater_path / 'log'
cache_path = image_rater_path / 'cache'

embedding_config = "ViT-H-14"
embedding_cache = None
max_size = 800

log_path.mkdir(parents=True, exist_ok=True)
cache_path.mkdir(parents=True, exist_ok=True)

def change_embedding_config(config: str):
    global embedding_cache, embedding_config
    if not config:
        yield "No config selected"
    elif config == embedding_config:
        yield f"{config} is already loaded"
    else:
        embedding_config=config
        yield f"Loading OpenCLIP {embedding_config}..."
        embedding_cache = EmbeddingCache(cache_path, config=embedding_config)
        yield "Done"
    

def generate_comparison(state: dict):
    filepaths = state['files']
    
    if len(filepaths) < 2:
        return None, None
        
    random.shuffle(filepaths)
    
    outputs = []
    state['current_comparison'] = selected = filepaths[0:2]
    for filepath in selected:
        image = Image.open(filepath)
        
        # Precalculate embedding
        embedding_cache.get_embedding(str(filepath), image)
        
        padded_image = Image.new(mode="RGB", size=(max_size, max_size), color="white")
        image.thumbnail((max_size, max_size))
        padded_image.paste(image, box=((max_size - image.width) // 2, (max_size - image.height) // 2))
        outputs.append(padded_image)
        
    return outputs

def log_and_generate(result, state: dict):
    print(f"result={result}")
    print(f"log_path={log_path}")
    
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
        
    global embedding_cache, embedding_config
    if not embedding_cache:
        yield [f"Loading OpenCLIP {embedding_config}...", None, None]
        embedding_cache = EmbeddingCache(cache_path, config=embedding_config)
    
    yield [f"Loading images from {images_path}...", None, None]
    
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
            if filepath.suffix == '.txt':
                continue
            image = Image.open(filepath)
            filepath = str(filepath)
            state['files'].append(filepath)
            #embed = embedding_cache.get_embedding(filepath, image)
            #print(f"{filepath}: {embed}")
        except Exception as e:
            print(e)
            continue
    
    state['loading'] = False
    outputs = generate_comparison(state)
    yield [f"Loaded {len(state['files'])} images from {images_path}", *outputs]
    
def clear_images(state: dict, progress: gr.Progress = gr.Progress()):
    state['files'] = []
    state['current_comparison'] = []
    return ["No images loaded", None, None]
    
def calculate_embeddings(state: dict, progress: gr.Progress = gr.Progress()):
    if len(state['files']) == 0:
        yield "No files loaded"
        return

    yield "Calculating embeddings..."
    embedding_cache.precalc_embedding_batch(state['files'], progress)
    yield "Done"
    
def on_ui_tabs():
    with gr.Blocks() as ui_tab:
        with gr.Accordion(label="Config"):
            with gr.Row():
                with gr.Column():
                    images_path = gr.Textbox(label="Images path", scale=1)
                    with gr.Row():
                        load_images_btn = gr.Button(value="Load", scale=1)
                        unload_btn = gr.Button(value="Unload", scale=1)
                status_area = gr.Textbox(label="Status", interactive=False, scale=2, lines=3)
        with gr.Tab(label="Rate"):
            gr.HTML("Pick the better image!", elem_id="imagerater_calltoaction")
            with gr.Row(elem_id="imagerater_image_row"):
                with gr.Column():
                    left_img = gr.Image(interactive=False, container=False)
                    left_btn = gr.Button(value="Pick")
                    left_val = gr.State(value=0)
                with gr.Column():
                    right_img = gr.Image(interactive=False, container=False)
                    right_btn = gr.Button(value="Pick")
                    right_val = gr.State(value=1)
            skip_btn = gr.Button(value="Skip", elem_id="imagerater_skipbutton")
            state = gr.State(value={
                "files": []
            })
        with gr.Tab(label="Analyze"):
            with gr.Row():
                calc_embeddings_btn = gr.Button(value="Calculate embeddings")
                test_train_btn = gr.Button(value="Test logistic regression")
                cancel_btn = gr.Button(value="Cancel", variant="stop")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(label="Model", scale=3, value="ViT-H-14", choices=[
                            "ViT-H-14",
                            "DataComp-ViT-L-14",
                            "ViT-L-14",
                            "convnext_large_d_320",
                        ])
                        load_model_btn = gr.Button(value="Load", scale=1)
                    validation_split = gr.Slider(label="Validation split %", value=20, minimum=0, maximum=95, step=5)
                    maximum_train_samples = gr.Number(label="Maximum train samples", value=100, precision=0)
                    weight_decay = gr.Number(label="Weight decay", value=0.1)
                    optimization_steps = gr.Number(label="Optimization steps", value=100)
                    trials = gr.Slider(label="Trials", value=1, minimum=1, maximum=10, step=1)
                gr.Gallery(scale=3)
        
        load_event = load_images_btn.click(load_images, inputs=[images_path, state], outputs=[status_area, left_img, right_img])
        unload_btn.click(clear_images, cancels=[load_event], inputs=[state], outputs=[status_area, left_img, right_img])
        
        skip_btn.click(generate_comparison, inputs=[state], outputs=[left_img, right_img])
        left_btn.click(log_and_generate, inputs=[left_val, state], outputs=[left_img, right_img])
        right_btn.click(log_and_generate, inputs=[right_val, state], outputs=[left_img, right_img])
        
        calc_embeddings_event = calc_embeddings_btn.click(calculate_embeddings, inputs=[state], outputs=[status_area])
        #test_train_btn.click(test_fn, inputs=[state], outputs=[status_area])
        cancel_btn.click(lambda: "Cancelled", cancels=[calc_embeddings_event], outputs=[status_area])
        
        load_model_btn.click(change_embedding_config, cancels=[calc_embeddings_event], inputs=[model_dropdown], outputs=[status_area])
    
    return (ui_tab, "Image Rater", "imagerater"),


script_callbacks.on_ui_tabs(on_ui_tabs)
