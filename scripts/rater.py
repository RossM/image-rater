import random
import time
import gc
import json
import gradio as gr

import torch
import torch.nn.functional as F

from modules import devices, script_callbacks, scripts, shared, call_queue
from modules.ui import create_output_panel, create_refresh_button

from PIL import Image
from pathlib import Path

from scripts.modules.embedding import EmbeddingCache
from scripts.modules.logistic import LogisticRegression

root_path = Path(__file__).parents[3]
image_rater_path = root_path / 'image_rater'
log_path = image_rater_path / 'log'
cache_path = image_rater_path / 'cache'

embedding_cache = None
max_size = 800

log_path.mkdir(parents=True, exist_ok=True)
cache_path.mkdir(parents=True, exist_ok=True)

def safe_filename(s: str):
    return "".join(x if x.isalnum() or x == " " else "_" for x in s)

def change_embedding_config(config: str, progress: gr.Progress = gr.Progress()):
    global embedding_cache
    if not config:
        return "No config selected"
    elif embedding_cache != None and embedding_cache.config == config:
        return f"{config} is already loaded"
    else:
        progress(0, f"Loading OpenCLIP {config}")
        embedding_cache = EmbeddingCache(cache_path, config=config)
        return "OpenCLIP loaded"

def generate_comparison(state: dict):
    filepaths = state['files']
    
    if len(filepaths) < 2:
        return None, None
        
    random.shuffle(filepaths)
    score_embedding = state.get('score_embedding', None)
    if state['opt_prefer_high_scoring'] and score_embedding != None:
        candidates = filepaths[0:5]
        candidates.sort(reverse=True, key=lambda filename: embedding_cache.get_embedding(filename).dot(score_embedding))
        print(candidates)
        selected = candidates[0:2]
        random.shuffle(selected)
    else:
        selected = filepaths[0:2]
    
    outputs = []
    state['current_comparison'] = selected
    for filepath in selected:
        image = Image.open(filepath)
        
        # Precalculate embedding
        embedding_cache.get_embedding(str(filepath), image)
        
        padded_image = Image.new(mode="RGB", size=(max_size, max_size), color="white")
        image.thumbnail((max_size, max_size))
        padded_image.paste(image, box=((max_size - image.width) // 2, (max_size - image.height) // 2))
        outputs.append(padded_image)
        
    return outputs

def log_and_generate(result, prompt_file, state: dict):
    with open(log_path / (safe_filename(prompt_file) + '.json'), 'a') as f:
        f.write(json.dumps({
            'files': state['current_comparison'],
            'choice': result,
        }) + "\n")
    
    return generate_comparison(state)

def load_images(images_path: str, config: str, state: dict, progress: gr.Progress = gr.Progress()):
    if not images_path:
        return ["You must provide the path to a directory with image files", None, None]
        
    global embedding_cache
    if not embedding_cache or embedding_cache.config != config:
        progress(0, f"Loading OpenCLIP {config}")
        embedding_cache = EmbeddingCache(cache_path, config=config)
        state['score_embedding'] = None
    
    path = Path(images_path)
    
    filepaths = list(path.glob('*'))
    state['current_comparison'] = []
    state['loading'] = True
    
    if len(filepaths) == 0:
        return [f"No files found at {images_path}, check that you have the correct directory", None, None]
    
    for filepath in progress.tqdm(filepaths, desc="Checking files", unit="files"):
        try:
            if filepath.suffix == '.txt':
                continue
            image = Image.open(filepath)
            filepath = str(filepath)
            state['files'].append(filepath)
        except Exception as e:
            print(e)
            continue
    
    state['loading'] = False
    outputs = generate_comparison(state)
    return [f"Loaded {len(state['files'])} total images", *outputs]
    
def clear_images(state: dict, progress: gr.Progress = gr.Progress()):
    state['files'] = []
    state['current_comparison'] = []
    global embedding_cache
    embedding_cache = None
    state['score_embedding'] = None
    gc.collect()
    devices.torch_gc()
    torch.cuda.empty_cache()    
    return ["No images loaded", None, None]
    
def calculate_embeddings(config: str, state: dict, progress: gr.Progress = gr.Progress()):
    if len(state['files']) == 0:
        return "No files loaded"

    global embedding_cache
    if not embedding_cache or embedding_cache.config != config:
        progress(0, f"Loading OpenCLIP {config}")
        embedding_cache = EmbeddingCache(cache_path, config=config)
    
    embedding_cache.precalc_embedding_batch(state['files'], progress)
    return "Done"
    
def test_logistic_regression(
        config: str,
        prompt_file: str,
        validation_split_pct: float,
        max_train_samples: int,
        weight_decay: float,
        optimization_steps: int,
        trials: int,
        lr: float,
        goodness_bias: float,
        state: dict,
        progress: gr.Progress = gr.Progress(),
    ):

    global embedding_cache
    if not embedding_cache or embedding_cache.config != config:
        progress(0, f"Loading OpenCLIP {config}")
        embedding_cache = EmbeddingCache(cache_path, config=config)

    log_entries = []
    with open(log_path / (safe_filename(prompt_file) + '.json'), 'r') as f:
        for line in progress.tqdm(f, desc="Reading log", unit="lines"):
            try:
                log_entry = json.loads(line)
                log_entries.append(log_entry)
            except Exception as e:
                print(e)
    
    logged_files = set(filename for log_entry in log_entries for filename in log_entry['files'])
    
    embedding_cache.precalc_embedding_batch(logged_files, progress)
    
    device = devices.get_device_for('image_rater')
    
    input_tensors = []
    weight_tensors = []
    for log_entry in progress.tqdm(log_entries, desc="Building input", unit="examples"):
        if len(log_entry['files']) < 2:
            print(f"Invalid log entry: {log_entry}")
            continue
        choice = log_entry['choice']
        try:
            embeddings = [embedding_cache.get_embedding(filename) for filename in log_entry['files']]
            embedding_diff = embeddings[1 - choice] - embeddings[choice]
            embedding_mean = (embeddings[0] + embeddings[1]) / 2
            assert(embedding_diff.dtype == float or embedding_diff.dtype == torch.float32)
            if embedding_diff.isinf().any():
                print(f"Infinite embedding, skipping: {log_entry}")
                continue
            input_tensors.append(torch.stack([embedding_diff, embedding_mean]))
        except Exception as e:
            print(e)
    
    validation_samples = validation_split_pct * len(input_tensors) // 100
    train_samples = min(len(input_tensors) - validation_samples, max_train_samples)
    
    print(f"train_samples={train_samples}, validation_samples={validation_samples}")
    
    lr_scheduler_type = "linear"
    
    validation_losses = []
    score_embeddings = []
    for trial in progress.tqdm(range(trials), desc="Running", unit="trials"):
        random.shuffle(input_tensors)
        train_input = torch.stack(input_tensors[0:train_samples]).to(device=device)
        validation_input = torch.stack(input_tensors[train_samples:train_samples+validation_samples]).to(device=device)
        
        model = LogisticRegression(dim=train_input.shape[2])
        model.to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 - epoch / optimization_steps)
        else:
            scheduler = None
        
        for step in progress.tqdm(range(optimization_steps), desc="Optimizing", unit="steps"):
            pred = model(train_input[:,0,:])
            for i in range(0, pred.shape[0]):
                if pred[i].isnan():
                    print(f"Suspicious input: {train_input[i]}")
                    return
            # Goodness bias controls how much comparisons between high-scoring examples are
            # weighted over other examples in computing loss. If set to 0, all examples count equally.
            example_weights = F.softmax(goodness_bias * train_input[:,1,:] @ model.c.t()).detach()
            train_loss = ((pred ** 2) * example_weights).sum()
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
            with torch.no_grad():
                pred = model(validation_input[:,0,:])
                validation_loss = F.mse_loss(pred, torch.zeros_like(pred), reduction="mean")
 
        print(f"validation_loss={validation_loss}")
        validation_losses.append(validation_loss.item())
        score_embeddings.append(model.c.data.clone().detach())
        
    scores = {}
    with torch.no_grad():
        score_embedding = torch.stack(score_embeddings).mean(dim=0).cpu()
        state['score_embedding'] = score_embedding
        candidate_files = state['files'] if len(state['files']) > 0 else logged_files
        topfiles = list(filename for filename in candidate_files if filename in embedding_cache.cache)
        topfiles.sort(reverse=True, key=lambda filename: embedding_cache.get_embedding(filename).dot(score_embedding))
    
    if validation_samples > 0:
        try:
            validation_mean = torch.Tensor(validation_losses).mean()
            with open(image_rater_path / 'regression_trials.csv', 'a') as f:
                f.write(f"{embedding_cache.config},{train_samples},{validation_samples},{lr},{weight_decay},{optimization_steps}," + 
                        f"{trials},{validation_mean},{lr_scheduler_type},{optimizer.__class__},{goodness_bias}\n")
        except Exception as e:
            print(e)
    
    return ["Done", topfiles[0:12]]
    
def on_ui_tabs():
    def get_prompts():
        global prompt_options
        prompts = set(filename.stem for filename in log_path.glob('*.json'))
        prompts.add("default")
        prompt_options = list(prompts)
        prompt_options.sort()
    get_prompts()

    with gr.Blocks() as ui_tab:
        with gr.Accordion(label="Config"):
            with gr.Row():
                with gr.Column():
                    images_path = gr.Textbox(label="Images path")
                    with gr.Row():
                        load_images_btn = gr.Button(value="Load")
                        unload_btn = gr.Button(value="Clear All")
                    with gr.Row():
                        prompt_dropdown = gr.Dropdown(label="Rating prompt", value="default", choices=prompt_options, interactive=True)
                        create_refresh_button(prompt_dropdown, get_prompts, lambda: {"choices": prompt_options, "value": prompt_dropdown.value}, "prompt_dropdown_refresh")
                with gr.Column():
                    status_area = gr.Textbox(label="Status", interactive=False, lines=2)
                    with gr.Row():
                        prefer_high_scoring = gr.Checkbox(label="Prefer high-scoring images", value=True, interactive=True)
        with gr.Tab(label="Rate"):
            prompt_html = gr.HTML(value="Pick the better image!", elem_id="imagerater_calltoaction")
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
                "files": [],
                "opt_prefer_high_scoring": True,
            })
        with gr.Tab(label="Analyze"):
            with gr.Row():
                calc_embeddings_btn = gr.Button(value="Calculate embeddings")
                test_train_btn = gr.Button(value="Test logistic regression")
                cancel_btn = gr.Button(value="Cancel", variant="stop")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(label="OpenCLIP Model", scale=3, value="ViT-H-14", choices=[
                            "ViT-H-14",
                            "DataComp-ViT-L-14",
                            "ViT-L-14",
                            "convnext_large_d_320",
                        ])
                        load_model_btn = gr.Button(value="Load CLIP", scale=1)
                    validation_split = gr.Slider(label="Validation split %", value=20, minimum=0, maximum=95, step=5)
                    maximum_train_samples = gr.Number(label="Maximum train samples", value=10000, precision=0)
                    weight_decay = gr.Number(label="Weight decay", value=2)
                    optimization_steps = gr.Number(label="Optimization steps", value=200, precision=0)
                    trials = gr.Slider(label="Trials", value=1, minimum=1, maximum=10, step=1)
                    lr = gr.Number(label = "Learning rate", value=0.2)
                    goodness_bias = gr.Number(label = "Goodness bias", value=0)
                test_gallery = gr.Gallery(label="Preview", preview=True).style(columns=4, object_fit='contain')
        with gr.Tab(label="Preprocess"):
            with gr.Row():
                with gr.Column():
                    preprocess_input_path = gr.Textbox(label="Source directory")
                    preprocess_input_include_subdirectories = gr.Checkbox(label="Include subdirectories")
                    preprocess_output_path = gr.Textbox(label="Destination directory")
                    preprocess_maximum = gr.Number(label="Maximum output images", value=1000, precision=0)
                    with gr.Row():
                        with gr.Column(scale=2):
                            preprocess_dimension = gr.Number(label="Output dimension", value=512, precision=0)
                        with gr.Column(scale=3, min_width=480):
                            preprocess_resize_mode=gr.Radio(label="Resize mode", value="Don't resize", choices=[
                                "Don't resize",
                                "Area",
                                "Largest dimension",
                                "Smallest dimension",
                            ])
                    preprocess_random_crops = gr.Checkbox(label="Create random crops")
                    with gr.Row():
                        preprocess_crop_size = gr.Number(label="Crop size", value=512, precision=0)
                        preprocess_num_crops = gr.Slider(label="Number of crops", value=4, minimum=1, maximum=10)
                    with gr.Row():
                        with gr.Column(scale=9):
                            preprocess_filter_prompt_include = gr.Textbox(label="Include images matching", interactive=True)
                        with gr.Column(scale=1, min_width=160):
                            preprocess_filter_prompt_include_mode = gr.Radio(label="Match type", value="Any", choices=["Any", "All"], interactive=True)
                    with gr.Row():
                        with gr.Column(scale=9):
                            preprocess_filter_prompt_exclude = gr.Textbox(label="Exclude images matching", interactive=True)
                        with gr.Column(scale=1, min_width=160):
                            preprocess_filter_prompt_exclude_mode = gr.Radio(label="Match type", value="Any", choices=["Any", "All"], interactive=True)
                    preprocess_filter_confidence = gr.Slider(label="Filter confidence threshold", value=0.5, minimum=0, maximum=1, step=0.01)
                    preprocess_caption_txt = gr.Radio(label="Existing caption txt", value="Copy", choices=["Ignore", "Copy", "Append", "Prepend"], interactive=True)
                    with gr.Row():
                        preprocess_random_crops = gr.Checkbox(label="Encourage diversity")
                        preprocess_caption_blip2 = gr.Checkbox(label="Caption using BLIP2")
                        preprocess_caption_wd14 = gr.Checkbox(label="Caption using Waifu Diffusion 1.4 tagger")
                        preprocess_use_hardlink = gr.Checkbox(label="Use hardlinks where possible")
                with gr.Column():
                    gr.Button(value="Preprocess images")
                    gr.Gallery(label="Preview", preview=True).style(columns=4, object_fit='contain')
        
        def prompt_dropdown_change(prompt):
            if prompt == "default":
                return "Pick the better image!"
            else:
                return f"Pick the more {prompt} image!"
        prompt_dropdown.select(prompt_dropdown_change, inputs=[prompt_dropdown], outputs=[prompt_html])
        prompt_dropdown.blur(prompt_dropdown_change, inputs=[prompt_dropdown], outputs=[prompt_html])
        
        def prefer_high_scoring_change(val, state):
            state['opt_prefer_high_scoring'] = val
        prefer_high_scoring.change(prefer_high_scoring_change, inputs=[prefer_high_scoring, state])
        
        load_event = load_images_btn.click(load_images, inputs=[images_path, model_dropdown, state], status_tracker=[status_area], outputs=[status_area, left_img, right_img])
        unload_btn.click(clear_images, cancels=[load_event], inputs=[state], status_tracker=[status_area], outputs=[status_area, left_img, right_img])
        
        skip_btn.click(generate_comparison, inputs=[state], outputs=[left_img, right_img])
        left_btn.click(log_and_generate, inputs=[left_val, prompt_dropdown, state], outputs=[left_img, right_img])
        right_btn.click(log_and_generate, inputs=[right_val, prompt_dropdown, state], outputs=[left_img, right_img])
        
        calc_embeddings_event = calc_embeddings_btn.click(calculate_embeddings, inputs=[model_dropdown, state], status_tracker=[status_area], outputs=[status_area])
        test_train_event = test_train_btn.click(test_logistic_regression, inputs=[
            model_dropdown,
            prompt_dropdown,
            validation_split,
            maximum_train_samples,
            weight_decay,
            optimization_steps,
            trials,
            lr,
            goodness_bias,
            state,
        ], status_tracker=[status_area], outputs=[status_area, test_gallery])
        #cancel_btn.click(lambda: "Cancelled", cancels=[calc_embeddings_event, test_train_event], outputs=[status_area])
        
        load_model_btn.click(change_embedding_config, cancels=[calc_embeddings_event], inputs=[model_dropdown], status_tracker=[status_area], outputs=[status_area])
    
    return (ui_tab, "Image Rater", "imagerater"),


script_callbacks.on_ui_tabs(on_ui_tabs)
