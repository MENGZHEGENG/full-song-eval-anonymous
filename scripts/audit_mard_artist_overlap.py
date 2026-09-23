"""Join the MARD classification subset to the creators' album artist metadata.

Only the metadata ZIP member is transferred using HTTP byte ranges. The raw
metadata is held in memory; the project-local ID mapping stays under ignored
data/ and is excluded from reviewer archives.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

from run_mard_prospective_transfer import fold_id


ROOT = Path(__file__).resolve().parents[1]
URL = "https://mtg.upf.edu/system/files/projectsweb/mard.zip"
MEMBER = "mard/mard_metadata.json"
SOURCE = ROOT / "data/external/mard/dataset_classification.json"
PROTOCOL = ROOT / "configs/mard_transfer_protocol.json"
MAPPING = ROOT / "data/external/mard/album_artist_mbid.json"
REPORT = ROOT / "reports/mard_artist_overlap_audit.json"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RangedReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.position
        if count == 0:
            return b""
        data = get_range(self.url, self.position, min(self.size - 1, self.position + count - 1))
        self.position += len(data)
        return data


def get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise ValueError("incomplete MARD byte-range response")
    return data


def load_metadata_member() -> bytes:
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers["Content-Length"])
    with zipfile.ZipFile(RangedReader(URL, size)) as archive:
        info = archive.getinfo(MEMBER)
    header = get_range(URL, info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50 or info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError("unexpected MARD ZIP member encoding")
    start = info.header_offset + 30 + fields[-2] + fields[-1]
    raw = zlib.decompress(get_range(URL, start, start + info.compress_size - 1), -15)
    if len(raw) != info.file_size:
        raise ValueError("MARD metadata length mismatch")
    return raw


def main() -> None:
    subset = json.loads(SOURCE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    raw = load_metadata_member()
    mapping: dict[str, str] = {}
    for line in raw.splitlines():
        row = json.loads(line)
        album = row.get("amazon-id")
        if album in subset:
            artist = row.get("artist-mbid")
            if not artist:
                raise ValueError(f"artist MBID missing for album {album}")
            mapping[album] = str(artist)
    if len(mapping) != len(subset):
        raise ValueError("not all classification albums join to full metadata")
    groups: dict[str, set[int]] = defaultdict(set)
    members: dict[str, set[str]] = defaultdict(set)
    for album, artist in mapping.items():
        groups[artist].add(fold_id(album, seed=protocol["fold_seed"], fold_count=10))
        members[artist].add(album)
    crossing = {artist for artist, folds in groups.items() if len(folds) > 1}
    mapping_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    MAPPING.parent.mkdir(parents=True, exist_ok=True)
    MAPPING.write_bytes(mapping_bytes)
    report = {
        "schema_version": "full-song-eval-mard-artist-audit/v1",
        "status": "complete",
        "source_url": URL,
        "metadata_member": MEMBER,
        "metadata_sha256": sha_bytes(raw),
        "classification_sha256": sha_bytes(SOURCE.read_bytes()),
        "protocol_sha256": sha_bytes(PROTOCOL.read_bytes()),
        "mapping_sha256": sha_bytes(mapping_bytes),
        "album_count": len(mapping),
        "distinct_artist_mbid_count": len(groups),
        "cross_fold_artist_mbid_count": len(crossing),
        "albums_in_cross_fold_artist_groups": sum(len(members[artist]) for artist in crossing),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
