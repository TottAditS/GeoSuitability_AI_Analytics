"""Local Apache Spark for the experiments (Spark MLlib in m16, GraphFrames in g01).

Runs in local mode on at most E.N_JOBS cores, the same CPU budget as the other experiments. No cluster.
The Windows settings are the same as in scripts/preprocessing/06_sedona_processing.ipynb: winutils from tools/, Python
workers pinned to this interpreter, driver bound to the loopback interface.
`lake_path` gives the HDFS location of a data file when HDFS is running and the local one otherwise.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import exp_lib as E

GRAPHFRAMES = "graphframes:graphframes:0.8.4-spark3.5-s_2.12"   # GraphFrames for Spark 3.5 / Scala 2.12
SPARK_PACKAGES_REPO = "https://repos.spark-packages.org"
LOCAL_LAKE = E.PROJECT_ROOT / "datasets"


def lake_path(*parts: str) -> str:
    """Path of a file under datasets/, on HDFS if it is running, else on the local disk.

    Same rule as lake_path in scripts/download/common.py. LAKE_URI overrides it ("local" forces the local disk).
    """
    base = os.environ.get("LAKE_URI", "").strip()
    if not base:
        try:
            with socket.create_connection(("localhost", 9000), timeout=0.5):
                base = "hdfs://localhost:9000/lake"
        except OSError:
            base = ""
    if base and base.lower() != "local":
        return "/".join([base.rstrip("/"), *[p.strip("/") for p in parts]])
    return str(LOCAL_LAKE.joinpath(*parts))


def session(app: str, packages: list[str] | None = None, cores: int = E.N_JOBS):
    """A local SparkSession; `packages` are Maven coordinates fetched once into the Ivy cache (needs internet the first time)."""
    hadoop = E.PROJECT_ROOT / "tools" / "hadoop-3.3.6"
    if os.name == "nt" and (hadoop / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop)
        os.environ["PATH"] = str(hadoop / "bin") + os.pathsep + os.environ["PATH"]
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession

    builder = (SparkSession.builder.appName(app).master(f"local[{cores}]")
               .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
               .config("spark.sql.shuffle.partitions", str(2 * cores)).config("spark.driver.memory", "4g")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               # HDFS client options for infra/hdfs (the DataNode is reached through localhost). Ignored for local files.
               .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
               .config("spark.hadoop.dfs.replication", "2")
               # Iterative jobs (GraphFrames) nest cached plans; their full text can outgrow the heap when Spark logs it.
               .config("spark.sql.maxPlanStringLength", str(2 ** 20)))
    if packages:
        builder = builder.config("spark.jars.packages", ",".join(packages)).config("spark.jars.repositories", SPARK_PACKAGES_REPO)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _local_path(url: str) -> str:
    """file:// URL (escaped spaces, a leading slash before the drive letter on Windows) -> local path."""
    if not url.startswith("file:"):
        return url
    path = unquote(urlparse(url).path)
    return path.lstrip("/") if os.name == "nt" else path


def enable_graphframes(spark):
    """Make the Python side of GraphFrames importable: its Python package ships inside the jar Spark downloaded."""
    urls = [u for u in spark.sparkContext.getConf().get("spark.jars", "").split(",") if "graphframes" in u]
    jars = [_local_path(u) for u in urls]
    jars = jars or [str(p) for p in sorted(Path.home().glob(".ivy2*/jars/graphframes_graphframes-*.jar"))]
    if not jars:
        raise FileNotFoundError("GraphFrames jar not found: create the session with packages=[exp_spark.GRAPHFRAMES]")
    jar = jars[0]   # a local path (spark.jars holds file:// URLs with escaped spaces)
    if jar not in sys.path:
        sys.path.insert(0, jar)
        spark.sparkContext.addPyFile(jar)
    checkpoints = E.PROJECT_ROOT / "datasets" / "processed" / "spark_output" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoints))   # needed by GraphFrames' connected components


def stop(spark):
    """Stop the session and delete its checkpoint folder (GraphFrames keeps the last checkpoint of every run)."""
    folder = spark.sparkContext.getCheckpointDir()
    spark.stop()
    if folder:
        shutil.rmtree(_local_path(folder), ignore_errors=True)
