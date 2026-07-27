# Local AI runtime

This folder provides an optional development container for training and inference. CVAT remains a separate service under `deployment/cvat/`.

## Native Python setup

Create an isolated environment, then install:

```bash
python -m pip install -e '.[ai,cvat,google]'
```

PyTorch GPU installation depends on the operating system, driver and CUDA runtime. On a GPU workstation, install the appropriate PyTorch build first, then install OpenSLIT with `--no-deps` only when necessary to preserve that build.

## Container setup

```bash
cp deployment/ai/.env.example deployment/ai/.env
docker compose -f deployment/ai/docker-compose.yml build
docker compose -f deployment/ai/docker-compose.yml run --rm openslit-ai
```

The repository and `.runtime` cache are mounted into the container. Clinical images, masks, checkpoints and predictions remain under Git-ignored local paths.

The default compose file is portable and does not request a GPU. GPU passthrough can be added locally after NVIDIA Container Toolkit is installed. Do not commit machine-specific GPU settings.

## Recommended hardware

Manual annotation and benchmark reporting do not need a GPU. Initial 512-pixel U-Net or SegFormer experiments are practical on a modern NVIDIA GPU with adequate memory; batch size should be reduced when memory is limited. CPU training is supported by the code but will be substantially slower.

## Services

```text
CVAT server        deployment/cvat/
AI training        deployment/ai/
Google Drive API   deployment/google/
Workflow state     collaboration_runs/.../workflow/
```

Keep these secrets outside Git:

- Google service-account JSON;
- CVAT access token;
- private model-repository token;
- clinical images and source identifiers;
- model checkpoints and probability maps.
