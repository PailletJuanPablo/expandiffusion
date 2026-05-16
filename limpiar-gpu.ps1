[CmdletBinding()]
param(
    [int[]]$Ports = @(8011, 5180),
    [switch]$OnlyRepo,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$CurrentProcessId = $PID

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Write-GpuState {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Host "nvidia-smi no esta disponible."
        return
    }

    Write-Host "GPU memory:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

    Write-Host ""
    Write-Host "CUDA compute processes:"
    $apps = nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($apps -join ""))) {
        Write-Host "(sin procesos compute reportados)"
        return
    }
    $apps
}

function Get-ProcessTable {
    Get-CimInstance Win32_Process |
        Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine
}

function Test-RepoOwnedProcess {
    param($Process)

    if ($null -eq $Process) {
        return $false
    }
    if ([int]$Process.ProcessId -eq $CurrentProcessId) {
        return $false
    }

    $name = [string]$Process.Name
    if ($name -notin @("python.exe", "pythonw.exe", "node.exe", "cmd.exe")) {
        return $false
    }

    $commandLine = [string]$Process.CommandLine
    $executablePath = [string]$Process.ExecutablePath

    return (
        $commandLine.Contains($RepoRoot) -or
        $executablePath.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $commandLine.Contains("expandiffusion.main:app") -or
        $commandLine.Contains("--app-dir apps/api") -or
        ($commandLine.Contains("vite") -and $commandLine.Contains("--port 5180")) -or
        $commandLine.Contains("scripts/dev.mjs")
    )
}

function Test-CudaCleanupProcess {
    param($Process)

    if ($null -eq $Process) {
        return $false
    }
    if ([int]$Process.ProcessId -eq $CurrentProcessId) {
        return $false
    }

    $name = [string]$Process.Name
    if ($name -notin @("python.exe", "pythonw.exe", "node.exe")) {
        return $false
    }

    if ($OnlyRepo) {
        return Test-RepoOwnedProcess $Process
    }

    return $true
}

function Get-ListeningProcessIds {
    param([int[]]$ListenPorts)

    if ($ListenPorts.Count -eq 0) {
        return @()
    }

    @(Get-NetTCPConnection -State Listen -LocalPort $ListenPorts -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

function Get-CudaProcessIds {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return @()
    }

    $rows = nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) {
        return @()
    }

    @($rows | ForEach-Object {
        $text = ([string]$_).Trim()
        if ($text -match "^\d+$") {
            [int]$text
        }
    })
}

function Add-Descendants {
    param(
        [int]$ProcessId,
        [object[]]$Processes,
        [System.Collections.Generic.HashSet[int]]$TargetIds
    )

    $children = @($Processes | Where-Object { [int]$_.ParentProcessId -eq $ProcessId })
    foreach ($child in $children) {
        if ($TargetIds.Add([int]$child.ProcessId)) {
            Add-Descendants -ProcessId ([int]$child.ProcessId) -Processes $Processes -TargetIds $TargetIds
        }
    }
}

function Add-RepoParents {
    param(
        [int]$ProcessId,
        [object[]]$Processes,
        [System.Collections.Generic.HashSet[int]]$TargetIds
    )

    $current = $Processes | Where-Object { [int]$_.ProcessId -eq $ProcessId } | Select-Object -First 1
    while ($null -ne $current) {
        $parent = $Processes | Where-Object { [int]$_.ProcessId -eq [int]$current.ParentProcessId } | Select-Object -First 1
        if ($null -eq $parent -or -not (Test-RepoOwnedProcess $parent)) {
            return
        }
        [void]$TargetIds.Add([int]$parent.ProcessId)
        $current = $parent
    }
}

Write-Section "Estado inicial"
Write-GpuState

$processes = @(Get-ProcessTable)
$targetIds = [System.Collections.Generic.HashSet[int]]::new()

foreach ($pidValue in (Get-ListeningProcessIds -ListenPorts $Ports)) {
    if ($pidValue -and [int]$pidValue -ne $CurrentProcessId) {
        [void]$targetIds.Add([int]$pidValue)
    }
}

foreach ($process in $processes) {
    if (Test-RepoOwnedProcess $process) {
        [void]$targetIds.Add([int]$process.ProcessId)
    }
}

foreach ($pidValue in (Get-CudaProcessIds)) {
    $process = $processes | Where-Object { [int]$_.ProcessId -eq [int]$pidValue } | Select-Object -First 1
    if (Test-CudaCleanupProcess $process) {
        [void]$targetIds.Add([int]$pidValue)
    }
}

foreach ($pidValue in @($targetIds)) {
    Add-Descendants -ProcessId $pidValue -Processes $processes -TargetIds $targetIds
    Add-RepoParents -ProcessId $pidValue -Processes $processes -TargetIds $targetIds
}

$targets = @($processes |
    Where-Object { $targetIds.Contains([int]$_.ProcessId) -and [int]$_.ProcessId -ne $CurrentProcessId } |
    Sort-Object ProcessId -Descending)

Write-Section "Procesos a cerrar"
if ($targets.Count -eq 0) {
    Write-Host "No se encontraron procesos API/web o CUDA Python/Node para cerrar."
} else {
    $targets | Select-Object ProcessId, ParentProcessId, Name, CommandLine | Format-Table -Wrap -AutoSize
}

if ($DryRun) {
    Write-Host ""
    Write-Host "DryRun activo: no se cerro ningun proceso."
    exit 0
}

foreach ($target in $targets) {
    Stop-Process -Id ([int]$target.ProcessId) -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 3

Write-Section "Estado final"
Write-GpuState

Write-Host ""
Write-Host "Listeners restantes en puertos $($Ports -join ', '):"
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Ports -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess)
if ($listeners.Count -eq 0) {
    Write-Host "(sin listeners)"
} else {
    $listeners | Format-Table -AutoSize
}
