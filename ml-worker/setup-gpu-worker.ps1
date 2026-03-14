# Subarr ML Worker - GPU Setup (Windows + WSL2/Docker Desktop)
# Gereksinimler:
#   1. Docker Desktop (WSL2 backend)
#   2. NVIDIA Container Toolkit
#   3. Tailscale (K8s DNS erisilebilir olmali)
#
# Kurulum:
#   1. Bu repoyu clone'la: git clone git@gitlab.enbitron.com:phirios/subarr.git
#   2. cd subarr/ml-worker
#   3. .env dosyasini olustur (asagidaki ornegi kullan)
#   4. Bu script'i calistir: .\setup-gpu-worker.ps1

Write-Host "=== Subarr ML Worker GPU Setup ===" -ForegroundColor Cyan

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker bulunamadi. Docker Desktop kurun." -ForegroundColor Red
    exit 1
}

# Check NVIDIA
$nvidiaSmi = docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "NVIDIA GPU erisimi yok. NVIDIA Container Toolkit kurun." -ForegroundColor Red
    Write-Host "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Yellow
    exit 1
}
Write-Host "GPU OK:" -ForegroundColor Green
Write-Host $nvidiaSmi

# Check .env
if (-not (Test-Path .env)) {
    Write-Host ".env dosyasi bulunamadi. Olusturuluyor..." -ForegroundColor Yellow
    Copy-Item .env.gpu.example .env
    Write-Host ".env dosyasini duzenleyin ve tekrar calistirin." -ForegroundColor Yellow
    exit 1
}

# Check Tailscale DNS
$redisCheck = docker run --rm redis:7-alpine redis-cli -h redis.phirios.svc.cluster.local -n 1 ping 2>&1
if ($redisCheck -ne "PONG") {
    Write-Host "Redis'e erisilemedi. Tailscale baglantisini kontrol edin." -ForegroundColor Red
    Write-Host "  redis.phirios.svc.cluster.local:6379 erisilebilir olmali" -ForegroundColor Yellow
    exit 1
}
Write-Host "Redis OK" -ForegroundColor Green

# Build and start
Write-Host "Worker build ediliyor..." -ForegroundColor Cyan
docker compose -f docker-compose.gpu.yml up -d --build

Write-Host "=== Worker baslatildi ===" -ForegroundColor Green
Write-Host "Loglar: docker compose -f docker-compose.gpu.yml logs -f" -ForegroundColor Yellow
