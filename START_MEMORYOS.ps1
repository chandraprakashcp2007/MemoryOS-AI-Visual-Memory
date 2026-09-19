$ErrorActionPreference="Stop"
$work="C:\Users\GODWIN\Downloads\MemoryOS-WORK"
$old="C:\Users\GODWIN\Downloads\MemoryOS-AI-Visual-Memory-main\MemoryOS-AI-Visual-Memory-main\.venv\Scripts\python.exe"
Set-Location $work
$py=if(Test-Path $old){$old}else{(Get-Command python).Source}
$env:MEMORYOS_VISUAL_DEVICE="cuda";$env:MEMORYOS_VISUAL_BATCH_SIZE="24";$env:MEMORYOS_VISUAL_LOCAL_ONLY="1";$env:MEMORYOS_GALLERY_BATCH_LIMIT="24";$env:MEMORYOS_GALLERY_ENRICH_WORKERS="1";$env:MEMORYOS_QUERY_AI_ENABLED="true";$env:MEMORYOS_ASSISTANT_MODEL="gemini-2.5-flash-lite";$env:MEMORYOS_ASSISTANT_WEB="true";$env:HF_HUB_OFFLINE="1";$env:TRANSFORMERS_OFFLINE="1"
foreach($port in @(8000,5173)){$ls=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue;foreach($l in $ls){try{Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue}catch{}}};Start-Sleep 2
$backend=Start-Process -FilePath $py -ArgumentList @("-m","uvicorn","backend.main:app","--host","127.0.0.1","--port","8000") -WorkingDirectory $work -PassThru
for($i=0;$i -lt 90;$i++){try{Invoke-RestMethod "http://127.0.0.1:8000/health/ready" -TimeoutSec 2|Out-Null;break}catch{if($backend.HasExited){throw "Backend failed"};Start-Sleep 1}}
try{Invoke-RestMethod "http://127.0.0.1:8000/assistant/warmup" -Method POST -TimeoutSec 180|Out-Null}catch{}
$npm=(Get-Command npm.cmd -ErrorAction Stop).Source
$frontend=Start-Process -FilePath $npm -ArgumentList @("run","dev","--","--host","127.0.0.1","--port","5173","--strictPort") -WorkingDirectory "$work\frontend" -PassThru
for($i=0;$i -lt 90;$i++){try{Invoke-RestMethod "http://127.0.0.1:5173/memoryos-api/health/ready" -TimeoutSec 2|Out-Null;break}catch{if($frontend.HasExited){throw "Frontend failed"};Start-Sleep 1}}
$status=Invoke-RestMethod "http://127.0.0.1:5173/memoryos-api/assistant/status"
Write-Host "`nMEMORYOS FULL STACK READY" -ForegroundColor Green;Write-Host "Streaming AI :" $status.streaming;Write-Host "Receipt AI   :" $status.receipt_extraction;Write-Host "Gemini       :" $status.configured;Write-Host "Model        :" $status.model
Start-Process "http://127.0.0.1:5173"
Read-Host "Press ENTER to stop MemoryOS"
