"""Loading two research repos into one process.

Uni-Sign and SignSparK are not packages: each is put on sys.path and imports its own top-level
modules by generic names (config, models, utils, ...). Whichever loads second would silently get the
first one's module out of sys.modules. `scoped_path` loads one repo with the other's same-named
modules hidden, then takes its own modules out of sys.modules again. The objects built inside keep
working: functions hold their module globals, not sys.modules entries. What would break is a
lazy `import x` executed later inside that repo; both are loaded completely at startup for that
reason, and neither imports lazily on the inference path we use.
"""

from __future__ import annotations

import contextlib
import os
import sys


def _top_level_names(path: str) -> set[str]:
    names = set()
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry.endswith(".py"):
            names.add(entry[:-3])
        elif os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")):
            names.add(entry)
    return names


def _owned(name: str, names: set[str]) -> bool:
    return name.split(".", 1)[0] in names


@contextlib.contextmanager
def scoped_path(*paths: str):
    paths = [str(p) for p in paths]
    names = set().union(*(_top_level_names(p) for p in paths))
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if _owned(k, names)}
    sys.path[:0] = paths
    try:
        yield
    finally:
        for p in paths:
            if p in sys.path:
                sys.path.remove(p)
        for k in [k for k in sys.modules if _owned(k, names)]:
            del sys.modules[k]
        sys.modules.update(hidden)
