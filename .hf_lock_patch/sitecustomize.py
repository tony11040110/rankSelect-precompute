import os
import tempfile
import hashlib

try:
    import datasets.builder as db
    from filelock import FileLock as _FileLock

    class TmpFileLock(_FileLock):
        def __init__(self, lock_file, *args, **kwargs):
            base = os.path.basename(lock_file)
            h = hashlib.md5(lock_file.encode("utf-8")).hexdigest()[:10]
            lock_file = os.path.join(tempfile.gettempdir(), f"{base}.{h}.lock")
            super().__init__(lock_file, *args, **kwargs)

    db.FileLock = TmpFileLock
except Exception:
    # Best-effort. If patching fails, datasets will use the default FileLock.
    pass
