#!/usr/bin/env python3
"""Check portable release paths, syntax, hashes and absence of large assets."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r'/(?:mnt/(?:data-hdd|data-ssd)|home/[^/\s]+|sda/(?:home|data)|deepo_data)(?:/[^\s]*)?'
)
EXCLUDED = {'.git', '.venv', '__pycache__', '.cache', 'data', 'weights', 'outputs'}


def main():
    errors = []
    count = 0
    for p in ROOT.rglob('*'):
        if any(part in EXCLUDED for part in p.relative_to(ROOT).parts):
            continue
        if not p.is_file():
            continue
        count += 1
        if p.stat().st_size > 10 * 1024 * 1024:
            errors.append(f'Large source asset: {p.relative_to(ROOT)}')
        if p.suffix in {'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt'}:
            text = p.read_text()
            if FORBIDDEN.search(text):
                errors.append(f'Workstation path: {p.relative_to(ROOT)}')
            if p.suffix == '.py':
                ast.parse(text, filename=str(p))
    manifest = ROOT / 'provenance/source_manifest.json'
    for record in json.loads(manifest.read_text()):
        p = ROOT / record['destination']
        expected = record.get('release_sha256', record.get('sha256'))
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(f'Release hash mismatch: {record["destination"]}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {count} release files checked; portable paths, syntax and source hashes OK.')


if __name__ == '__main__':
    main()
