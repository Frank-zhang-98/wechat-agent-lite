param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$projectName = Split-Path $ProjectRoot -Leaf
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$distDir = Join-Path $ProjectRoot "dist"
$zipPath = Join-Path $distDir "${projectName}-${timestamp}.zip"
$hashPath = "${zipPath}.sha256"

$excludeDirNames = @(".git", ".venv", "data", "output", "tmp", "dist", "__pycache__")
$excludeFilePatterns = @("*.pyc", "*.pyo", "*.db", "*.sqlite3", "*.log", ".env", "tmp_*.json")

New-Item -ItemType Directory -Path $distDir -Force | Out-Null

if (Test-Path $zipPath) {
  Remove-Item -Path $zipPath -Force
}
if (Test-Path $hashPath) {
  Remove-Item -Path $hashPath -Force
}

$rootWithSlash = "$ProjectRoot\"
$files = Get-ChildItem -Path $ProjectRoot -Recurse -File | Where-Object {
  $fullPath = $_.FullName
  $relativePath = $fullPath.Substring($rootWithSlash.Length)
  $segments = $relativePath -split '[\\/]'
  $name = $_.Name

  if ($segments | Where-Object { $excludeDirNames -contains $_ }) {
    return $false
  }
  foreach ($pattern in $excludeFilePatterns) {
    if ($name -like $pattern) {
      return $false
    }
  }
  return $true
}

$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
  foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($rootWithSlash.Length)
    $entryName = $relativePath -replace '\\','/'
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
      $zip,
      $file.FullName,
      $entryName,
      [System.IO.Compression.CompressionLevel]::Optimal
    ) | Out-Null
  }
}
finally {
  $zip.Dispose()
}

$hash = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLowerInvariant()
Set-Content -Path $hashPath -Value "$hash  $(Split-Path $zipPath -Leaf)" -Encoding ascii

Write-Output "package: $zipPath"
Write-Output "sha256:  $hash"
