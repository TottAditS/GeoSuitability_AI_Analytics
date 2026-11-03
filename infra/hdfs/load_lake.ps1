<#
Start HDFS (via start_hdfs.ps1) and copy the analysis data into it (about 55 MB). Safe to run again.

  processed/features/*.csv    -> /lake/processed/features
  processed/experiments       -> /lake/processed/experiments
  processed/spark_output      -> /lake/processed/spark_output   (without checkpoints)
  processed/grid/grid_1km.gpkg -> /lake/processed/grid
  data_dictionary.csv         -> /lake/raw_sample

Not copied: datasets/raw (26 GB of rasters that GDAL reads from disk), Sentinel-2 patches and mosaics.
The local files stay the source of truth.

Run:  powershell -ExecutionPolicy Bypass -File infra/hdfs/load_lake.ps1
#>
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..\..")

# docker prints progress to stderr. Under "Stop", PowerShell 5.1 treats that as an error.
$ErrorActionPreference = "Continue"

& (Join-Path $here "start_hdfs.ps1")
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "docker not found" }

function Hdfs {
    # Run a command in the NameNode container and stop if it fails.
    docker compose exec -T namenode @args
    if ($LASTEXITCODE -ne 0) { throw "failed: $($args -join ' ')" }
}

Push-Location $here
try {
    foreach ($d in @("/lake/processed/features", "/lake/processed/experiments", "/lake/processed/spark_output",
                     "/lake/processed/grid", "/lake/raw_sample")) {
        Hdfs hdfs dfs -mkdir -p $d
    }

    Hdfs sh -c "hdfs dfs -put -f /import/features/*.csv /lake/processed/features/"
    Hdfs sh -c "hdfs dfs -put -f /import/experiments/* /lake/processed/experiments/"
    Hdfs sh -c "for d in /import/spark_output/*; do [ `$(basename `$d) = checkpoints ] || hdfs dfs -put -f `$d /lake/processed/spark_output/; done"
    Hdfs hdfs dfs -put -f /import/grid/grid_1km.gpkg /lake/processed/grid/

    # data_dictionary.csv is outside the mounted folder, so pipe it in through stdin.
    Get-Content -Raw (Join-Path $root "datasets\data_dictionary.csv") |
        docker compose exec -T namenode sh -c "hdfs dfs -put -f - /lake/raw_sample/data_dictionary.csv"

    # Make sure every file has two copies, even if it was written earlier with one.
    Hdfs hdfs dfs -setrep -w 2 /lake | Out-Null

    Write-Host "`nContents of /lake:"
    Hdfs hdfs dfs -du -h -s /lake
    Hdfs hdfs dfs -ls /lake/processed
    Write-Host "`nHDFS is ready. NameNode UI: http://localhost:9870. Spark finds it on its own (hdfs://localhost:9000/lake)."
} finally {
    Pop-Location
}
