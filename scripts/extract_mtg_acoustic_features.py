#!/usr/bin/env python3
"""Verify official MTG AcousticBrainz shards and retain fixed audio features."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        sha, name = line.split()
        if name in values:
            raise ValueError(f"duplicate checksum name: {name}")
        values[name] = sha
    return values


def feature_vector(data: dict) -> list[float]:
    lowlevel = data["lowlevel"]
    mean = lowlevel["melbands"]["mean"]
    var = lowlevel["melbands"]["var"]
    if not isinstance(mean, list) or not isinstance(var, list) or len(mean) != 40 or len(var) != 40:
        raise ValueError("melbands mean/var is not 40+40 values")
    values = [float(value) for value in mean + var]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite melbands value")
    return [math.log1p(max(0.0, value)) for value in values]


def extract(shard_dir: Path, tar_manifest: Path, track_manifest: Path, output: Path, report_path: Path) -> dict:
    tar_shas = read_checksums(tar_manifest)
    track_shas = read_checksums(track_manifest)
    if len(tar_shas) != 100:
        raise ValueError("expected 100 official tar shards")
    seen: set[str] = set()
    invalid: list[dict[str, str]] = []
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
            for name in sorted(tar_shas):
                path = shard_dir / name
                if digest(path) != tar_shas[name]:
                    raise ValueError(f"tar checksum mismatch: {name}")
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        if member.name not in track_shas or member.name in seen:
                            raise ValueError(f"unexpected or duplicate member: {member.name}")
                        seen.add(member.name)
                        fileobj = archive.extractfile(member)
                        assert fileobj is not None
                        payload = fileobj.read()
                        if hashlib.sha256(payload).hexdigest() != track_shas[member.name]:
                            raise ValueError(f"track checksum mismatch: {member.name}")
                        track = int(Path(member.name).stem)
                        try:
                            vector = feature_vector(json.loads(payload))
                        except (KeyError, TypeError, ValueError) as exc:
                            invalid.append({"track_id": f"track_{track:07d}", "reason": str(exc)})
                            continue
                        compressed.write((json.dumps({"record_id": f"track_{track:07d}", "vector": vector}, separators=(",", ":")) + "\n").encode())
                        count += 1
    if seen != set(track_shas):
        raise ValueError(f"missing official feature members: {len(set(track_shas) - seen)}")
    report = {
        "schema": "fullsongeval-mtg-audio-features/v1",
        "status": "complete",
        "official_tar_count": len(tar_shas),
        "official_track_count": len(track_shas),
        "valid_feature_count": count,
        "invalid_features": invalid,
        "inputs_sha256": {"tar_manifest": digest(tar_manifest), "track_manifest": digest(track_manifest)},
        "output_sha256": digest(output),
        "feature_specification": "log1p(max(0,x)) for lowlevel.melbands.mean[40] and lowlevel.melbands.var[40]",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--tar-manifest", type=Path, required=True)
    parser.add_argument("--track-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.shard_dir, args.tar_manifest, args.track_manifest, args.output, args.report)
    print(json.dumps({key: result[key] for key in ("status", "official_track_count", "valid_feature_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
