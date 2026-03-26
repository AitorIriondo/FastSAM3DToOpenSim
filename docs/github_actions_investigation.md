# GitHub Actions — Automated Docker CI/CD (Investigation)

This document analyses how to automate Docker image builds and deployment
for the FastSAM3D service using GitHub Actions. **Nothing is implemented here**
— this is a reference for when we add CI/CD automation.

---

## The challenge: image size

The Docker image is ~15 GB (CUDA 12.8 + two conda environments + all ML
dependencies). This has significant implications for any CI/CD approach:

- Build time: ~30–45 min from scratch (cached layers: ~5 min for code-only changes)
- Push/pull: ~10–20 min on a 1 Gbps link
- Registry storage: ~15 GB per tag

---

## Option A: GitHub-hosted runner → GHCR → server pulls

```
Push to main
    ↓ GitHub Actions (ubuntu-latest runner)
    ↓ docker build (30–45 min, uncached)
    ↓ docker push → ghcr.io/aitoririondo/easyergo-fastsam3d:latest
    ↓ (webhook or Watchtower) → server pulls new image
    ↓ docker compose up -d --force-recreate
```

**Pros:**
- Fully automated, no infrastructure to maintain
- Works anywhere the server can reach GHCR

**Cons:**
- GitHub-hosted runners have no GPU → cannot test inference
- Uncached builds use a lot of runner minutes (free tier: 2000 min/month)
- 15 GB push per build stresses the GHCR bandwidth limit

**Mitigation:** Use multi-stage caching with `cache-from: type=gha`.
The CUDA/conda layer (~14 GB) rarely changes; only the code layer (~100 MB)
changes on each push — incremental push takes ~2-3 min.

**Recommended trigger:** Push to `main` OR push of a `v*` tag.

---

## Option B: Self-hosted runner on the 5070 Ti server (Recommended for GPU testing)

```
Push to main
    ↓ GitHub Actions → self-hosted runner (the 5070 Ti server itself)
    ↓ docker build (5 min cached, local build)
    ↓ docker compose up -d --force-recreate (no push/pull needed)
```

**Pros:**
- Local build = no bandwidth, fastest update cycle
- Can run inference tests with the real GPU
- No runner minute limits (self-hosted is free)
- No image push needed — image stays on the server

**Cons:**
- Server must be reachable by GitHub webhooks (needs public IP or ngrok)
- Runner process must be running on the server
- Less isolation — runner shares the server environment

**Setup:**
```bash
# On the server:
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.x.x.tar.gz -L https://github.com/actions/runner/releases/...
tar xzf ./actions-runner-linux-x64-*.tar.gz
./config.sh --url https://github.com/AitorIriondo/EasyErgo_FastSam3DService \
            --token <TOKEN_FROM_GITHUB>
sudo ./svc.sh install && sudo ./svc.sh start
```

**Workflow file** (`.github/workflows/deploy.yml`):
```yaml
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Build and restart
        run: |
          docker compose build
          docker compose up -d --force-recreate
```

---

## Option C: Watchtower (server auto-pulls from GHCR)

```
Push to main → GitHub Actions builds + pushes to GHCR
Watchtower (running on server) polls GHCR every N hours → pulls + restarts
```

**Pros:**
- Simple: no self-hosted runner, no webhooks needed
- Works behind NAT (server initiates the pull)

**Cons:**
- Delayed updates (depends on polling interval)
- 15 GB pull on each update — slow and storage-heavy
- No GPU test in CI

**Setup on server:**
```bash
docker run -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --interval 3600 \     # check every hour
  fastsam3d             # only watch this container
```

---

## Recommendation

**For the current stage (single server, private repo, frequent code changes):**

→ Use **Option B (self-hosted runner)**.

Reasons:
1. No bandwidth cost for 15 GB image
2. Builds in ~5 min (local cached layers)
3. Can test inference after deploy
4. Simple workflow file

**Once we have stable releases with less frequent deploys:**

→ Add **Option A** alongside — build and push a versioned image to GHCR
  so other machines (new servers, team members) can pull without rebuilding.

---

## Multi-stage build optimisation

The Dockerfile can be split to maximise layer caching:

```dockerfile
# Stage 1: CUDA + conda base (changes rarely — only on CUDA/TRT version bumps)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base
# ... system deps, conda install, PyTorch, TRT, OpenSim ...

# Stage 2: Application code (changes on every commit)
FROM base AS app
COPY . /app
WORKDIR /app
```

With this split, a code-only change rebuilds only Stage 2 (~30 seconds).
The full rebuild from scratch (~30 min) only happens when the base image changes.

This is the most impactful optimisation for reducing CI build times.

---

## Security considerations

- Store the GitHub Personal Access Token (for GHCR push) as a repository secret
- The self-hosted runner should run as a non-root user with limited permissions
- Restrict CORS origins to the EasyErgoDashboard domain in production:
  `CORS_ORIGINS=https://easyergo.example.com`
- For public HTTPS, use Caddy as reverse proxy:
  ```
  caddy reverse-proxy --from :443 --to :8000
  ```
