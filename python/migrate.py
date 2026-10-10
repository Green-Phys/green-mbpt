"""Migrate versioned inputs using green-mbtools >= 1.1.

    python python/migrate.py --input input.h5 --output upgraded/input.h5

Uses the same arguments as python -m green_mbtools.mint.migrate.
For unversioned inputs, first run migrate_legacy_to_green.py to reach 0.2.4.
"""
from green_mbtools.mint.migrate import _main


if __name__ == "__main__":
    _main()
