#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


def load_manifest(manifest_path: Path) -> list[Path]:
    items: list[Path] = []
    with manifest_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            items.append(Path(line))
    return items


def repo_path_from_local(local_rel: Path, target_subdir: str) -> str:
    rel = local_rel.as_posix()
    if rel.startswith('data/raw/'):
        suffix = rel[len('data/raw/') :]
    else:
        suffix = local_rel.name
    return f"{target_subdir.rstrip('/')}/{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description='Publish curated JSONL files to a Hugging Face dataset repo.')
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--manifest', default='configs/release_jsonl_manifest.txt')
    parser.add_argument('--target-subdir', default='raw')
    parser.add_argument('--commit-message', default='sync curated jsonl snapshot')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--delete-extra', action='store_true')
    args = parser.parse_args()

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise SystemExit('HF_TOKEN is required in environment')

    repo_root = Path(args.repo_root).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f'manifest not found: {manifest_path}')

    local_rel_paths = load_manifest(manifest_path)
    if not local_rel_paths:
        raise SystemExit('manifest is empty')

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type='dataset', private=args.private, exist_ok=True)

    ops = []
    keep_repo_paths: set[str] = set()

    for rel in local_rel_paths:
        local_abs = (repo_root / rel).resolve()
        if not local_abs.exists():
            raise SystemExit(f'missing file from manifest: {rel}')
        path_in_repo = repo_path_from_local(rel, args.target_subdir)
        keep_repo_paths.add(path_in_repo)
        ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_abs)))

    if args.delete_extra:
        existing = api.list_repo_files(repo_id=args.repo_id, repo_type='dataset')
        prefix = f"{args.target_subdir.rstrip('/')}/"
        for file_path in existing:
            if file_path.startswith(prefix) and file_path not in keep_repo_paths:
                ops.append(CommitOperationDelete(path_in_repo=file_path))

    if not ops:
        print('No changes to commit.')
        return

    api.create_commit(
        repo_id=args.repo_id,
        repo_type='dataset',
        operations=ops,
        commit_message=args.commit_message,
    )

    print(f'Uploaded {len(local_rel_paths)} curated JSONL files to dataset {args.repo_id}.')
    print(f'URL: https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
