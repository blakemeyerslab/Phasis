"""
Per-chromosome storage of parsed tags.

The alignment parser used to write one pickle per library holding the tags of
every chromosome.  Cluster scoring then loaded that whole pickle in the main
process and serialised one chromosome at a time to its workers.  That transfer
was serial and cost more than the scoring itself for most tasks.

The parser now writes one pickle ("shard") per library-chromosome key under
``<library>_<phase>.dict.shards/`` and a small index at the old
``<library>_<phase>.dict`` path.  The index has the old layout,
``{akey: [value]}``, but each value is a reference to a shard instead of the
tags themselves, so the main process holds almost nothing and every scoring
worker reads exactly the chromosome it scores.

A ``.dict`` file written by an older Phasis (tags inline) is still accepted by
cluster scoring.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from collections import defaultdict

MARKER = "__phasis_nest_shard__"
SHARD_DIR_SUFFIX = ".shards"
# An index holds a few hundred bytes per chromosome; a legacy pickle with the
# tags inline is far larger.  Files above this size are not opened to check.
_INDEX_MAX_BYTES = 8 * 1024 * 1024


def shard_dir_for(index_path: str) -> str:
    return str(index_path) + SHARD_DIR_SUFFIX


def is_ref(obj) -> bool:
    return isinstance(obj, dict) and MARKER in obj


def is_ref_list(value) -> bool:
    """True for an index value, i.e. ``[shard reference]``."""
    return isinstance(value, list) and len(value) == 1 and is_ref(value[0])


def _file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_nest_shards(index_path: str, tags_by_akey: dict) -> dict:
    """
    Write one shard per key of ``tags_by_akey`` and the index at ``index_path``.

    ``tags_by_akey`` maps akey -> list of ``(position, taginfo)`` and is emptied
    while it is written, so the parser never holds the per-position copy of
    more than one chromosome.  Each shard is the ``defaultdict(list)`` of
    position -> list of taginfo that the scoring worker has always received.
    """
    index_path = os.path.abspath(str(index_path))
    shard_dir = shard_dir_for(index_path)
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir)

    index = {}
    for number, akey in enumerate(list(tags_by_akey), 1):
        entries = tags_by_akey.pop(akey)
        by_position = defaultdict(list)
        for position, taginfo in entries:
            by_position[position].append(taginfo)
        del entries

        name = f"shard_{number:05d}.p"
        path = os.path.join(shard_dir, name)
        with open(path, "wb") as handle:
            pickle.dump(by_position, handle, protocol=pickle.HIGHEST_PROTOCOL)
        index[akey] = [
            {
                MARKER: 1,
                "file": name,
                "path": path,
                "positions": len(by_position),
                # Makes the index content, and so its cache fingerprint,
                # change whenever the tags of a chromosome change.
                "md5": _file_md5(path),
            }
        ]
        del by_position

    with open(index_path, "wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def resolve_refs(loaded: dict, index_path: str) -> dict:
    """Point every shard reference of a loaded index at its file next to the index."""
    shard_dir = shard_dir_for(os.path.abspath(str(index_path)))
    for value in loaded.values():
        if is_ref_list(value):
            ref = value[0]
            candidate = os.path.join(shard_dir, str(ref.get("file", "")))
            if os.path.isfile(candidate):
                ref["path"] = candidate
    return loaded


def load_ref(ref: dict):
    """Load the position -> tags mapping a shard reference points to."""
    with open(ref["path"], "rb") as handle:
        return pickle.load(handle)


def load_expanded(index_path: str) -> dict:
    """
    Load a parser ``.dict`` file with every shard reference replaced by its
    tags, i.e. the structure older versions stored inline.  For tools and
    tests that need the whole library in memory.
    """
    with open(index_path, "rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        return loaded
    resolve_refs(loaded, index_path)
    for akey, value in list(loaded.items()):
        if is_ref_list(value):
            loaded[akey] = [load_ref(value[0])]
    return loaded


def index_is_complete(index_path: str) -> bool:
    """
    False when ``index_path`` is a shard index with a missing shard file.

    Legacy pickles (tags inline) and unreadable files return True: they are
    left to the normal cache checks.
    """
    try:
        if os.path.getsize(index_path) > _INDEX_MAX_BYTES:
            return True
        with open(index_path, "rb") as handle:
            loaded = pickle.load(handle)
    except Exception:
        return True
    if not isinstance(loaded, dict):
        return True

    shard_dir = shard_dir_for(os.path.abspath(str(index_path)))
    for value in loaded.values():
        if is_ref_list(value):
            if not os.path.isfile(os.path.join(shard_dir, str(value[0].get("file", "")))):
                return False
    return True
