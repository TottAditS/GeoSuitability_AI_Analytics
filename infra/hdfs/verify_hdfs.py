"""Check that Spark can use HDFS: read the fact table, write it back as Parquet, read it again.

Start HDFS first (infra/hdfs/load_lake.ps1), activate the project environment, then run:

    python infra/hdfs/verify_hdfs.py

This script always uses HDFS, even if LAKE_URI is unset.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LAKE_URI", "hdfs://localhost:9000/lake")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "modeling"))

import exp_spark as S  # noqa: E402


def main() -> None:
    spark = S.session("hdfs-verify", cores=2)
    try:
        src = S.lake_path("processed", "features", "tabular_features.csv")
        out = S.lake_path("tmp", "hdfs_roundtrip")
        sdf = spark.read.csv(src, header=True, inferSchema=True)
        n = sdf.count()
        print(f"read   {src}: {n:,} rows x {len(sdf.columns)} columns")
        sdf.write.mode("overwrite").parquet(out)
        back = spark.read.parquet(out).count()
        print(f"wrote  {out}: {back:,} rows read back")
        if back != n:
            raise SystemExit(f"row count mismatch: {n:,} written, {back:,} read back")
        print("HDFS round trip OK")
    finally:
        S.stop(spark)


if __name__ == "__main__":
    main()

