"""Pobieranie oryginalnego Kylberg v1.0 (bez gotowych obrotów)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parent
BASE = 'https://filedn.com/lkCRue0RhPO7ercIrqFRl2Y/datasets/KylbergTextureDatasetV1/without-rotations-zip/'


def list_archives():
    with urlopen(Request(BASE, headers={'User-Agent': 'Kylberg-lab4/1.0'}), timeout=40) as response:
        html = response.read().decode('utf-8')
    match = re.search(r'var directLinkData\s*=\s*(\{.*?\});', html, re.S)
    if not match:
        raise RuntimeError('Nie odnaleziono listy plików serwera autora.')
    listing = json.loads(match.group(1))
    archives = [item for item in listing['content'] if item['name'].lower().endswith('.zip')]
    if not archives:
        raise RuntimeError('Lista archiwów ZIP jest pusta.')
    return sorted(archives, key=lambda item: item['name'])


def download_one(item):
    archives = ROOT / 'data' / 'archives'
    extracted = ROOT / 'data' / 'kylberg'
    archives.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    name = item['name']
    target = archives / name
    url = urljoin(BASE, quote(name))
    expected_size = item['size']
    if not target.exists() or target.stat().st_size != expected_size:
        part = target.with_suffix('.zip.part')
        with urlopen(Request(url, headers={'User-Agent': 'Kylberg-lab4/1.0'}), timeout=120) as response:
            with part.open('wb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if part.stat().st_size != expected_size:
            raise RuntimeError(f'Niepełne pobranie {name}')
        part.replace(target)
    with zipfile.ZipFile(target) as archive:
        # Verify every member, including paths, before writing into data/kylberg.
        for member in archive.infolist():
            destination = (extracted / member.filename).resolve()
            if not destination.is_relative_to(extracted.resolve()):
                raise ValueError(f'Niebezpieczna ścieżka archiwum: {member.filename}')
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Błąd CRC: {bad}')
        for member in archive.infolist():
            destination = extracted / member.filename
            if not member.is_dir() and (not destination.exists() or destination.stat().st_size != member.file_size):
                archive.extract(member, extracted)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f'OK {name} ({expected_size / 1024**2:.1f} MiB)', flush=True)
    return {'filename': name, 'url': url, 'bytes': expected_size, 'sha256': sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--classes', nargs='+', help='Opcjonalnie pobierz tylko wskazane klasy.')
    args = parser.parse_args()
    items = list_archives()
    print(json.dumps(items, ensure_ascii=False, indent=2) if args.list else f'Znaleziono {len(items)} archiwów.', flush=True)
    if args.list:
        return
    if args.classes:
        items = [i for i in items if i['name'].removesuffix('-without-rotations.zip') in args.classes]
        if len(items) != len(args.classes):
            raise ValueError('Nie wszystkie wybrane klasy znaleziono na serwerze.')
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Potrzeba co najmniej 5 GiB wolnego miejsca na dane i wyniki.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download_one, items))
    manifest_path = ROOT / 'data' / 'download_manifest.json'
    manifest_path.write_text(json.dumps({'dataset': 'Kylberg Texture Dataset v1.0, without rotations',
                                         'source_index': BASE, 'archives': manifest}, indent=2), encoding='utf-8')
    pngs = list((ROOT / 'data' / 'kylberg').rglob('*.png'))
    print(f'Gotowe: {len(pngs)} obrazów PNG; manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
