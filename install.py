import launch

if not launch.is_installed("open_clip_torch"):
    launch.run_pip("install open_clip_torch", "open_clip_torch")