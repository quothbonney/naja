#!/usr/bin/env python3
"""Fetch the exact installed hopsy release's official PyPI source archive."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

root=Path(__file__).resolve().parents[1]
metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/hopsy/1.7.0/json'))
release=next(r for r in metadata['urls'] if r['packagetype']=='sdist')
sources=root/'source'
sources.mkdir(exist_ok=True)
archive=sources/release['filename']
urllib.request.urlretrieve(release['url'],archive)
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==release['digests']['sha256']
with tarfile.open(archive) as tar:
    tar.extractall(sources,filter='data')
record=dict(version='1.7.0',source='official PyPI sdist',url=release['url'],
            sha256=actual,archive=str(archive.relative_to(root)))
(root/'audit/hopsy_source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
