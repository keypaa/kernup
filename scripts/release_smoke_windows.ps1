# Run a reproducible local smoke pass for Kernup v0.1.0 on Windows.
$ErrorActionPreference = "Stop"

Write-Host "[1/6] Running test suite..."
conda run -n kernup-dev pytest

Write-Host "[2/6] Running profile dry-run..."
conda run -n kernup-dev kernup profile --hf Qwen/Qwen2.5-7B --dry-run --allow-no-gpu --output ./kernup_results --export

Write-Host "[3/6] Running optimize phase 1 dry-run..."
conda run -n kernup-dev kernup optimize --hf Qwen/Qwen2.5-7B --phase 1 --target balanced --iterations 6 --population 4 --dry-run --allow-no-gpu --output ./kernup_results

Write-Host "[4/6] Running optimize phase 2 dry-run..."
conda run -n kernup-dev kernup optimize --hf Qwen/Qwen2.5-7B --phase 2 --target balanced --iterations 4 --population 4 --dry-run --allow-no-gpu --output ./kernup_results

Write-Host "[5/6] Generating patch artifact..."
conda run -n kernup-dev kernup patch --hf Qwen/Qwen2.5-7B --results ./kernup_results --format simple --output ./patch

Write-Host "[6/6] Generating bench summary..."
conda run -n kernup-dev kernup bench --hf Qwen/Qwen2.5-7B --results ./kernup_results --export --output ./kernup_results

Write-Host "Release smoke pass completed successfully."
