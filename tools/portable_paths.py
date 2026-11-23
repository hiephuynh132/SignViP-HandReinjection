"""Sanitize workstation-specific paths in a prepared source export."""
from pathlib import Path
import hashlib
import json
import re


def sanitize_release(root, source_root):
    root, source_root = Path(root), Path(source_root).resolve()
    # Derive old locations from the supplied source root; never embed a username.
    replacements = [
        (str(source_root.parent / 'RWTH-T'), 'data/raw_rwth'),
        (str(source_root.parent / '.cache/huggingface'), 'data/hf_cache'),
        (str(source_root.parent / 'env_hiep_llm/bin/python'), 'python3'),
        (str(source_root.parent / 'hiep_env/bin/python'), 'python3'),
        (str(source_root), '.'),
    ]
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix not in {'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt'}:
            continue
        if p.name in {'portable_paths.py', 'source_manifest.json'}:
            continue
        text = p.read_text()
        before = text
        text = re.sub(r'/' + r'home/[^/\s]+/Downloads/signLanguages_center_crop_hamer_signvip', 'data/hamer', text)
        text = re.sub(r'/' + r'home/[^/\s]+/Downloads/signLanguages_center_crop', 'data/rwth', text)
        for old, new in replacements:
            text = text.replace(old, new)
        # Remove inherited upstream developer paths, including commented examples.
        text = re.sub(r'/sda/(?:home|data)/[^\s\"\'\)]+', 'data/legacy_example', text)
        text = text.replace('D:\\phoenix14t.pami0.train.annotations_only.gzip',
                            'data/raw_rwth/train.annotations_only.gzip')
        if p.name == 'WEIGHT_PATHS.txt':
            text = text.replace('SOURCE_PATH=', 'SOURCE_REFERENCE=')
            text = text.replace('LOCAL_HF_CACHE=', 'CACHE_REFERENCE=')
        if text != before:
            p.write_text(text)
    manifest = root / 'provenance/source_manifest.json'
    if manifest.exists():
        records = json.loads(manifest.read_text())
        for rec in records:
            rec['source_sha256'] = rec.get('source_sha256', rec.pop('sha256', None))
            p = root / rec['destination']
            rec['release_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            rec['release_bytes'] = p.stat().st_size
        manifest.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    sanitize_release(Path(__file__).resolve().parents[1], args.source_root)
