param([switch]$NoBrowser)
# Locate the application and its local address.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$prototypeUrl = 'http://127.0.0.1:5081'
Set-Location -LiteralPath $projectRoot

# Reuse only the explicitly versioned instance for this package.
# This build uses a dedicated port/version so an older prototype server cannot be mistaken for this build.
try {
    $health = Invoke-RestMethod -Uri "$prototypeUrl/api/health" -TimeoutSec 2
    if ($health.application -eq 'COS700_5G_Flask_Prototype' -and $health.version -eq 'bright-cards-v2') {
        if (-not $NoBrowser) { Start-Process $prototypeUrl }
        Write-Host "Prototype is ready at $prototypeUrl"
        exit 0
    }
} catch { }

# Prefer an existing project or workspace environment.
$workspaceRuntime = Join-Path (Split-Path (Split-Path $projectRoot -Parent) -Parent) 'work\prototype-runtime\Scripts\python.exe'
$projectRuntime = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonPath = $null
foreach ($candidate in @($projectRuntime, $workspaceRuntime)) {
    if (Test-Path -LiteralPath $candidate) { $pythonPath = $candidate; break }
}

# Bootstrap an isolated environment when opening the project elsewhere.
if (-not $pythonPath) {
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $bundled) {
        & $bundled -m venv (Join-Path $projectRoot '.venv')
    } elseif ($pythonCommand) {
        & $pythonCommand.Source -m venv (Join-Path $projectRoot '.venv')
    } else {
        throw 'Install Python 3.12, then open Start Prototype.cmd again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    $pythonPath = $projectRuntime
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection.' }
    & $pythonPath -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
}

# Start the local server silently and keep readable logs.
$logFolder = Join-Path $projectRoot 'instance'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
$server = Start-Process -FilePath $pythonPath -ArgumentList @('"' + (Join-Path $projectRoot 'app.py') + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logFolder 'server.log') -RedirectStandardError (Join-Path $logFolder 'server-errors.log')
$server.Id | Set-Content -LiteralPath (Join-Path $logFolder 'server.pid')

# Open the interface after the dataset has finished loading.
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri "$prototypeUrl/api/health" -TimeoutSec 2
        if ($health.application -eq 'COS700_5G_Flask_Prototype' -and $health.version -eq 'bright-cards-v2') {
            if (-not $NoBrowser) { Start-Process $prototypeUrl }
            Write-Host "Prototype is ready at $prototypeUrl"
            exit 0
        }
    } catch { }
    if ($server.HasExited) { break }
}
throw 'The server did not start. See instance\server-errors.log for details.'
