# Expandiffusion

Local outpainting studio with a React/Konva editor and a FastAPI backend running real Diffusers Stable Diffusion adapters.

## Structure

```text
apps/
  api/  FastAPI, Diffusers adapters, runtime inspection, jobs API
  web/  React, TypeScript, Vite, Konva editor
```

Runtime adapters:

- `sd15-inpaint`: `stable-diffusion-v1-5/stable-diffusion-inpainting`
- `sd2-inpaint`: `stabilityai/stable-diffusion-2-inpainting`
- `sdxl-inpaint`: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- `flux-fill`: `black-forest-labs/FLUX.1-Fill-dev`
- `chroma-inpaint`: `lodestones/Chroma1-HD`

## Setup

```powershell
cd E:\expandiffusion\apps\api
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,diffusers]"
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

cd E:\expandiffusion\apps\web
npm install
```

For gated Hugging Face models such as `black-forest-labs/FLUX.1-Fill-dev`,
accept access on Hugging Face, create a read token, set `HF_TOKEN` in `.env`,
and restart the API before loading the model.

## Run

```powershell
cd E:\expandiffusion
npm run dev
```

Default ports from `.env.example`:

- API: `http://127.0.0.1:8011`
- Web: `http://127.0.0.1:5180`

The dev runner refuses to start if either fixed port is already in use. Logs are written and echoed with prefixes:

- `apps/api/api.out.log`
- `apps/api/api.err.log`
- `apps/web/web.out.log`
- `apps/web/web.err.log`

## Verify

```powershell
cd E:\expandiffusion
npm run dev:check

cd E:\expandiffusion\apps\api
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest

cd E:\expandiffusion\apps\web
npm run lint
npm run build
```

Real model smoke test:

```powershell
cd E:\expandiffusion\apps\api
$env:EXPANDIFFUSION_RUN_REAL_SD_TESTS="1"
.\.venv\Scripts\python.exe -m pytest -m real_sd
```

## Google Colab

Open the Colab notebook and run the cells in order:

```text
https://colab.research.google.com/github/PailletJuanPablo/expandiffusion/blob/main/notebooks/expandiffusion_colab.ipynb
```

The notebook installs backend and frontend dependencies, starts the API and Vite
servers, checks `/api/health`, then prints a temporary Colab proxy URL for the
web UI.

That UI URL is valid only while the Colab notebook runtime is active. For gated
Hugging Face models, add `HF_TOKEN` as a Colab secret before loading the model.

## Current MVP

- Stable Diffusion, FLUX and Chroma inpaint/outpaint through Diffusers only.
- Runtime inspection shows PyTorch, torchvision, CUDA visibility, selected device and dtype.
- CUDA is used when PyTorch sees GPUs; CPU is explicit and reported when CUDA is unavailable.
- Model setup is separated from the main editing surface.
- Canvas pan/zoom, import base image, add visible reference layers, resize canvas, resize selection, mask brush and eraser.
- Outpaint jobs with WebSocket status/progress, result previews, previous/next navigation, accept/cancel/retry, history and PNG export.
- Save/load `.expd` project archives.
- Plugin manager for enabling/disabling local plugins.
- Included Auto Detailer plugin for optional face/body second-pass refinement.
