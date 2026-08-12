"""YAML configuration loading with inheritance, dot-access and CLI overrides.

Design goals
------------
* A config file may declare ``defaults: <path>`` to inherit from a base config.
  Inheritance is resolved recursively and merged *deeply*, so an experiment
  config only needs to state what differs from the baseline.
* Values are reachable both as attributes (``cfg.train.epochs``) and as items
  (``cfg["train"]["epochs"]``), which keeps call sites readable.
* ``--set a.b=c`` style overrides are parsed with YAML semantics so that
  ``true``, ``3``, ``1e-4`` and ``[1,2]`` all arrive with the right type.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


class Config(dict):
    """A ``dict`` that also supports attribute access, recursively."""

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = value

    # -- container protocol -------------------------------------------------
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - surfaces typos clearly
            raise AttributeError(f"No config key '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    # -- convenience --------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up ``a.b.c``, returning ``default`` if any level is missing."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Assign ``a.b.c = value``, creating intermediate dicts as needed."""
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        """Plain-``dict`` copy, suitable for YAML dumping or W&B logging."""
        return {k: (v.to_dict() if isinstance(v, Config) else _unwrap(v)) for k, v in self.items()}

    def flat(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield ``("a.b", value)`` pairs; handy for TensorBoard hparams."""
        for key, value in self.items():
            path = f"{prefix}{key}"
            if isinstance(value, Config):
                yield from value.flat(f"{path}.")
            else:
                yield path, value

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` onto ``base`` without mutating either.

    Nested mappings merge key-by-key; every other type (including lists) is
    replaced outright, which is the behaviour you want for things like
    ``augment.scale: [0.7, 1.0]``.
    """
    merged = Config(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, resolving ``defaults:`` inheritance and overrides.

    Parameters
    ----------
    path:
        Path to the YAML file to load.
    overrides:
        Optional ``["train.epochs=40", "model.backbone=convnext_small"]`` list,
        applied last so the command line always wins.
    """
    cfg = _load_with_inheritance(Path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    return cfg


def _load_with_inheritance(path: Path, _seen: set[Path] | None = None) -> Config:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular config inheritance at {path}")
    _seen.add(path)

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("defaults", None)
    if parent is None:
        return Config(raw)

    # `defaults` is resolved relative to the child config's directory.
    parent_cfg = _load_with_inheritance(path.parent / parent, _seen)
    return deep_merge(parent_cfg, raw)
