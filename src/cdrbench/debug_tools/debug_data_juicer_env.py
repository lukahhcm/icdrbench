#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VENDORED_DJ = ROOT / 'data-juicer'


def print_header(title: str) -> None:
    print(f'\n=== {title} ===')


def print_kv(key: str, value: object) -> None:
    print(f'{key}: {value}')


def clear_data_juicer_modules() -> None:
    stale = [name for name in sys.modules if name == 'data_juicer' or name.startswith('data_juicer.')]
    for name in stale:
        sys.modules.pop(name, None)


def try_import_installed_data_juicer() -> None:
    print_header('Installed Package Check')
    clear_data_juicer_modules()
    try:
        import data_juicer  # type: ignore

        print_kv('installed.data_juicer', data_juicer.__file__)
        try:
            import data_juicer.core.data  # type: ignore

            print_kv('installed.data_juicer.core.data', data_juicer.core.data.__file__)
        except Exception:
            print('installed data_juicer.core.data import failed:')
            traceback.print_exc()
    except Exception:
        print('installed data_juicer import failed:')
        traceback.print_exc()


def try_import_vendored_data_juicer() -> None:
    print_header('Vendored Package Check')
    clear_data_juicer_modules()
    sys.path.insert(0, str(VENDORED_DJ))
    try:
        import data_juicer  # type: ignore

        print_kv('vendored.data_juicer', data_juicer.__file__)
        import data_juicer.core.data  # type: ignore

        print_kv('vendored.data_juicer.core.data', data_juicer.core.data.__file__)
    except Exception:
        print('vendored data_juicer import failed:')
        traceback.print_exc()
    finally:
        if sys.path and sys.path[0] == str(VENDORED_DJ):
            sys.path.pop(0)


def try_runpy_import(script_path: Path) -> None:
    print_header(f'Import {script_path.relative_to(ROOT)}')
    clear_data_juicer_modules()
    sys.path.insert(0, str(VENDORED_DJ))
    try:
        runpy.run_path(str(script_path), run_name='__dj_debug__')
        print('import_only: ok')
    except Exception:
        print('import_only: failed')
        traceback.print_exc()
    finally:
        if sys.path and sys.path[0] == str(VENDORED_DJ):
            sys.path.pop(0)


def main() -> None:
    print_header('Environment')
    print_kv('cwd', os.getcwd())
    print_kv('repo_root', ROOT)
    print_kv('python_executable', sys.executable)
    print_kv('python_version', sys.version.replace('\n', ' '))
    print_kv('sys.path[0]', sys.path[0] if sys.path else '<empty>')

    print_header('Vendored DJ Layout')
    print_kv('vendored_root_exists', VENDORED_DJ.exists())
    print_kv('tools.process_data_exists', (VENDORED_DJ / 'tools' / 'process_data.py').exists())
    print_kv('tools.analyze_data_exists', (VENDORED_DJ / 'tools' / 'analyze_data.py').exists())
    print_kv('core.data_init_exists', (VENDORED_DJ / 'data_juicer' / 'core' / 'data' / '__init__.py').exists())

    try_import_installed_data_juicer()
    try_import_vendored_data_juicer()
    try_runpy_import(VENDORED_DJ / 'tools' / 'process_data.py')
    try_runpy_import(VENDORED_DJ / 'tools' / 'analyze_data.py')

    print_header('How To Share')
    print('Run this command from repo root and paste the full output:')
    print('PYTHONPATH=src python -m cdrbench.debug_tools.debug_data_juicer_env')


if __name__ == '__main__':
    main()
