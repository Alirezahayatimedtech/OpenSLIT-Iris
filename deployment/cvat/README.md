# Local CVAT deployment

This folder deploys the free, self-hosted CVAT Community edition on the computer that stores the OpenSLIT-Iris pilot data.

The wrapper does not copy CVAT source code into this repository. It clones the official tagged CVAT release into `.runtime/cvat`, which is excluded from Git.

## Requirements

- Linux, or Windows with WSL2;
- Docker Engine or Docker Desktop;
- Docker Compose v2;
- Git;
- sufficient local disk space for CVAT containers, database, and images.

A GPU is not required for manual annotation.

## 1. Configure

```bash
cp deployment/cvat/.env.example deployment/cvat/.env
```

The default release is `v2.64.0`. The Python SDK is pinned to the matching `2.64.0` release because CVAT requires server and SDK major/minor versions to match.

## 2. Start CVAT

```bash
chmod +x deployment/cvat/cvat.sh
deployment/cvat/cvat.sh up
```

Open:

```text
http://localhost:8080
```

## 3. Create the administrator

```bash
deployment/cvat/cvat.sh create-superuser
```

Sign in with this account. Create separate accounts for each ophthalmologist. Do not share one account between annotators.

## 4. Install the OpenSLIT CVAT integration

```bash
python -m pip install -e '.[cvat]'
```

Create a read/write Personal Access Token in CVAT under `Profile > Security`, then place it only in `deployment/cvat/.env`:

```bash
CVAT_ACCESS_TOKEN=your_private_token
```

The integration also supports `CVAT_USERNAME` and `CVAT_PASSWORD` for a local pilot, but a Personal Access Token is preferred.

Load the variables into the current shell:

```bash
set -a
source deployment/cvat/.env
set +a
```

## 5. Check the local pilot plan

```bash
openslit-cvat check --config configs/cvat_pilot_v1.json
```

This validates the annotation schema, pilot manifest, selected images, and task plan without connecting to CVAT.

## 6. Create the project and independent tasks

First replace the placeholder CVAT usernames in `configs/cvat_pilot_v1.json` with the real accounts created in CVAT.

```bash
openslit-cvat setup --config configs/cvat_pilot_v1.json
```

The command:

- creates one project using the OpenSLIT-Iris annotation classes;
- selects only images marked for independent double annotation;
- creates one duplicate task per ophthalmologist;
- uploads the same blinded images to each task;
- assigns each task to its designated CVAT account;
- refuses to duplicate existing projects or tasks unless `--allow-existing` is used.

## Management commands

```bash
deployment/cvat/cvat.sh status
deployment/cvat/cvat.sh logs
deployment/cvat/cvat.sh restart
deployment/cvat/cvat.sh down
```

## Data safeguards

- Use only aliased images from `collaboration_runs/.../drive_upload/images`.
- Do not upload the private patient key to CVAT.
- Keep `deployment/cvat/.env`, `.runtime/`, database volumes, exports, and backups out of Git.
- Restrict remote access through the institutional network, VPN, or HTTPS.
- Back up CVAT before software upgrades.

Official references:

- CVAT installation: <https://docs.cvat.ai/docs/administration/basics/installation/>
- CVAT SDK: <https://docs.cvat.ai/docs/api_sdk/sdk/highlevel-api/>
- Personal Access Tokens: <https://docs.cvat.ai/docs/api_sdk/access_tokens/>
