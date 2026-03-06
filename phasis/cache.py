from __future__ import annotations
import phasis.runtime as rt
import configparser
import datetime
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from hashlib import md5 as _md5
import hashlib
import os
import shutil
import time
import re

MEM_FILE_DEFAULT = "phasis.mem"


# ---------------------------------------------------------------------------
# Centralized cache API (Phase II)
# ---------------------------------------------------------------------------


@dataclass
class MemCache:
    """Small helper around the phasis.mem (ConfigParser) cache.

    Cache rule:
      - If an output path is NOT registered in mem -> cache miss.
      - Cache hit requires output fingerprint match, and if input_sig is provided,
        also requires signature match stored under "<outpath>.sig".
    """

    memFile: str
    cfg: configparser.ConfigParser

    @classmethod
    def load(cls, memFile: str) -> "MemCache":
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        if memFile:
            cfg.read(memFile)
        return cls(memFile=str(memFile or MEM_FILE_DEFAULT), cfg=cfg)

    def ensure(self, section: str) -> None:
        if not self.cfg.has_section(section):
            self.cfg.add_section(section)

    def get(self, section: str, key: str, default: str | None = None) -> str | None:
        if not self.cfg.has_section(section):
            return default
        return self.cfg[section].get(key, default)

    def set(self, section: str, key: str, value: object) -> None:
        self.ensure(section)
        self.cfg[section][key] = str(value)

    def flush(self) -> None:
        with open(self.memFile, "w") as fh:
            self.cfg.write(fh)

    def fingerprint(self, path: str) -> str:
        _, fp = getmd5(path)
        return str(fp or "")

    def _dbg_enabled(self) -> bool:
        v = str(os.environ.get("PHASIS_CACHE_DEBUG", "")).strip().lower()
        return v in {"1", "true", "yes", "y", "on"}

    def _dbg(self, msg: str) -> None:
        if self._dbg_enabled():
            print(f"[PHASIS:CACHE] {msg}")

    def hit(self, section: str, outpath: str, input_sig: str | None = None) -> bool:
        self.ensure(section)
        if not outpath:
            self._dbg(f"MISS section={section} out=<empty> reason=empty_outpath")
            return False
        if not os.path.isfile(outpath):
            self._dbg(f"MISS section={section} out={outpath} reason=missing_file")
            return False
        cur_fp = self.fingerprint(outpath)
        prev_fp = self.get(section, outpath)
        if not prev_fp:
            self._dbg(f"MISS section={section} out={outpath} reason=not_registered")
            return False
        if not cur_fp:
            self._dbg(f"MISS section={section} out={outpath} reason=fingerprint_failed")
            return False
        if prev_fp != cur_fp:
            self._dbg(
                f"MISS section={section} out={outpath} reason=fingerprint_mismatch prev={prev_fp} cur={cur_fp}"
            )
            return False
        if input_sig is None:
            self._dbg(f"HIT  section={section} out={outpath} mode=hash_only")
            return True
        prev_sig = self.get(section, sig_key(outpath))
        if not prev_sig:
            self._dbg(f"MISS section={section} out={outpath} reason=signature_not_registered")
            return False
        if prev_sig != input_sig:
            self._dbg(
                f"MISS section={section} out={outpath} reason=signature_mismatch prev={prev_sig} cur={input_sig}"
            )
            return False
        self._dbg(f"HIT  section={section} out={outpath} mode=hash+sig")
        return True

    def record(self, section: str, outpath: str, input_sig: str | None = None) -> str:
        self.ensure(section)
        if not outpath or not os.path.isfile(outpath):
            if not outpath:
                self._dbg(f"RECORD_SKIP section={section} out=<empty> reason=empty_outpath")
            else:
                self._dbg(f"RECORD_SKIP section={section} out={outpath} reason=missing_file")
            return ""
        cur_fp = self.fingerprint(outpath)
        if cur_fp:
            self.set(section, outpath, cur_fp)
        if input_sig is not None:
            self.set(section, sig_key(outpath), input_sig)
        self.flush()
        self._dbg(
            f"REC  section={section} out={outpath} fp={cur_fp or '<empty>'} sig={'<none>' if input_sig is None else input_sig}"
        )
        return cur_fp


def stage_signature(
    *,
    files: Iterable[str] | None = None,
    params: Dict[str, object] | None = None,
    extra: Iterable[str] | None = None,
) -> str:
    """Convenience wrapper for compute_cache_signature (central API)."""
    return compute_cache_signature(files=files, params=params, extra=extra)


CLEANUP_PATTERNS = [
    "fas",
    "sam",
    "dict",
    "count",
    "runtime",
    "sum",
    "scoredClusters",
    "candidate.clusters",
    "clusters",
]


def match_pattern(filename, patterns) -> bool:
    for pattern in patterns:
        if str(filename).endswith(str(pattern)):
            return True
    return False


def cleanup(base_dir: str | None = None, patterns=None) -> None:
    """
    Delete PHASIS intermediate files/directories under the run directory.

    Kept as a canonical helper outside legacy.py so the active pipeline can
    support -cleanup without routing cleanup logic through legacy.
    """
    cleanup_patterns = list(patterns or CLEANUP_PATTERNS)
    target_dir = base_dir or getattr(rt, "run_dir", None) or os.getcwd()
    target_dir = os.path.abspath(os.path.expanduser(target_dir))

    if not os.path.isdir(target_dir):
        return None

    for root, dirs, files in os.walk(target_dir, topdown=True):
        for dirname in list(dirs):
            if match_pattern(dirname, cleanup_patterns):
                path = os.path.join(root, dirname)
                shutil.rmtree(path)
                dirs.remove(dirname)

        for filename in files:
            if match_pattern(filename, cleanup_patterns):
                path = os.path.join(root, filename)
                os.remove(path)

    return None


def phase2_basename(base_name:str)->str:
    try:
        is_concat = bool(rt.concat_libs)
    except Exception:
        is_concat = False
    prefix = "concat_" if is_concat else ""
    return f"{prefix}{rt.phase}_{base_name}"

EMPTY_MD5      = "d41d8cd98f00b204e9800998ecf8427e"
_CHUNK_SIZE    = 8 * 1024 * 1024  # 8 MiB
_FINGERPRINT_SAMPLE_BYTES = 64 * 1024  # 64 KiB per sample
_MD5_RETRIES   = 3
_MD5_BACKOFF_S = 0.2

def _wait_size_stable(path, checks=3, interval=0.2, timeout=5.0):
    """Wait until file size stops changing for `checks` consecutive polls."""
    deadline = time.time() + timeout
    last = -1
    stable = 0
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(interval)
            continue
        if size == 0:
            # empty file is "stable" but handled by EMPTY_MD5 in getmd5()
            stable = 0
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last = size
        time.sleep(interval)
    # best effort: if it exists at all, proceed
    return os.path.isfile(path)


def _read_sample_at(fh, offset: int, n: int) -> bytes:
    fh.seek(offset, os.SEEK_SET)
    return fh.read(n)


def _fast_file_fingerprint(path: str) -> str | None:
    """
    Fast, content-based file fingerprint (no timestamps).

    Returns a 32-hex-character string (blake2s digest_size=16) derived from:
      - file size
      - sampled bytes (beginning, middle, end) for large files; full read for small files

    This is intentionally *not* cryptographic integrity like full-file MD5; it is a
    fast cache key suitable for detecting likely changes in large files.
    """
    try:
        st = os.stat(path)
        size = int(st.st_size)

        if size == 0:
            return EMPTY_MD5

        h = hashlib.blake2s(digest_size=16)
        h.update(str(size).encode("utf-8"))
        h.update(b"\0")

        sample = _FINGERPRINT_SAMPLE_BYTES

        with open(path, "rb") as f:
            if size <= sample * 3:
                h.update(f.read())
                return h.hexdigest()

            h.update(_read_sample_at(f, 0, sample))

            mid_off = max(0, (size // 2) - (sample // 2))
            h.update(_read_sample_at(f, mid_off, sample))

            end_off = max(0, size - sample)
            h.update(_read_sample_at(f, end_off, sample))

        return h.hexdigest()
    except Exception:
        return None


def getmd5(afile):
    """
    Return (afile, hex_str) used as a cache key.

    Historically this computed a full-file MD5. We now compute a fast,
    content-based fingerprint (BLAKE2s over file size + sampled bytes)
    that is much faster and sufficiently collision-resistant for cache invalidation.
    """
    p = afile
    try:
        _wait_size_stable(p, checks=3, interval=0.2, timeout=5.0)

        for attempt in range(_MD5_RETRIES):
            try:
                fp = _fast_file_fingerprint(p)
                if fp is None:
                    raise RuntimeError("fingerprint failed")
                return (p, fp)
            except Exception:
                time.sleep(_MD5_BACKOFF_S)

        return (p, "")
    except Exception:
        return (p, "")


def compute_md5_str(path: str) -> str | None:
    """
    Fast fingerprint -> hex string (or None on failure).

    Kept for backward-compatibility with older call sites that expect a function
    named compute_md5_str.
    """
    return _fast_file_fingerprint(path)


def _sig_of_list_str(items):
    """Return a stable 32-hex signature for a list of strings (blake2s)."""
    h = hashlib.blake2s(digest_size=16)
    for s in items:
        h.update(str(s).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def _md5_of_list_str(items):
    """Backward-compatible alias (signatures are *not* MD5)."""
    return _sig_of_list_str(items)


def sig_key(outpath: str) -> str:
    """Return the memFile key used to store the input-signature for outpath."""
    return f"{outpath}.sig"


def compute_cache_signature(
    *,
    files: Iterable[str] | None = None,
    params: Dict[str, object] | None = None,
    extra: Iterable[str] | None = None,
) -> str:
    """
    Compute a stable signature for a stage based on:
      - content fingerprints of input files (getmd5 -> fast content fingerprint)
      - key runtime parameters
      - optional extra strings (counts, library sets, etc.)

    This is used alongside output-file hashes to decide if cached outputs are reusable.
    """
    items = []

    for f in (files or []):
        p = os.path.abspath(os.path.expanduser(str(f)))
        if os.path.isfile(p):
            try:
                _, fp = getmd5(p)
            except Exception:
                fp = ""
            try:
                size = os.path.getsize(p)
            except Exception:
                size = -1
            items.append(f"FILE	{p}	{fp}	{size}")
        else:
            items.append(f"MISSING	{p}")

    if params:
        for k in sorted(params.keys()):
            items.append(f"PARAM	{k}={params.get(k)}")

    if extra:
        for x in extra:
            items.append(f"EXTRA	{x}")

    return _md5_of_list_str(items)


def md5_file_worker(path):
    """
    Return (ABS_REALPATH, md5hex or None).
    - Normalizes path to realpath for stable [CLUSTERED] keys.
    - Uses getmd5(), which now guarantees a string; maps "" -> None for failure.
    - No extra retries here (getmd5 already handles them).
    """
    try:
        p = os.path.realpath(path)
    except Exception:
        p = path

    _, md5hex = getmd5(p)          # always returns a string now
    if not md5hex:                  # "" => treat as failure
        md5hex = None
    return (p, md5hex)

def _chunk_id_from_name(fn):
    # 'ALL_LIBS-10.sRNA_21.cluster' -> 10 (int); robust to oddities
    try:
        after_dash = fn.rsplit("-", 1)[1]            # '10.sRNA_21.cluster'
        num = after_dash.split(".", 1)[0]            # '10'
        return int(num)
    except Exception:
        return 0
    
def discover_scored_prefixes(scored_dir, phase):
    # Returns sorted list of prefixes that have at least one *.cluster
    suffix = f".sRNA_{phase}.cluster"
    prefixes = set()
    if not os.path.isdir(scored_dir):
        return []
    for fn in os.listdir(scored_dir):
        if fn.endswith(suffix):
            prefixes.add(fn.rsplit("-", 1)[0])
    return sorted(prefixes)

def list_chunk_files_for_prefix(scored_dir, prefix, phase):
    # Returns sorted (by numeric chunk id) absolute paths for prefix
    suffix = f".sRNA_{phase}.cluster"
    out = []
    if not os.path.isdir(scored_dir):
        return out
    pref = f"{prefix}-"
    for fn in os.listdir(scored_dir):
        if fn.endswith(suffix) and fn.startswith(pref):
            out.append(os.path.join(scored_dir, fn))
    out.sort(key=lambda p: _chunk_id_from_name(os.path.basename(p)))
    return out

def assemble_candidate_from_chunks(scored_dir, lib_prefix, phase, out_path):
    """
    Stream-concatenate non-empty chunk files for a given lib_prefix into out_path.
    Returns (#chunks_used, total_bytes_written).
    """
    chunks = list_chunk_files_for_prefix(scored_dir, lib_prefix, phase)
    used = 0
    written = 0
    # fresh file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as outfh:
        for cp in chunks:
            try:
                if os.path.getsize(cp) == 0:
                    continue
            except Exception:
                # if stat fails, skip conservatively
                continue
            with open(cp, "rb") as infh:
                # copy in 1 MiB blocks
                while True:
                    blk = infh.read(1024 * 1024)
                    if not blk:
                        break
                    outfh.write(blk)
                    written += len(blk)
            used += 1
    return used, written

def _runtime_params_signature() -> str:
    """Hash of runtime knobs that affect window selection (no JSON dependency)."""
    phase_v = globals().get("phase")
    wl_v    = globals().get("window_len")
    sl_v    = globals().get("sliding")
    mcl_v   = globals().get("minClusterLength")
    sig_str = f"phase={phase_v};wl={wl_v};sl={sl_v};mcl={mcl_v};version=winselect.1"
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()

# === Phase II cache and naming helpers (concat-aware, param-aware, ConfigParser-safe) ===

def _run_signature_keyprefix():
    # No '=' or ':' in the signature to keep ConfigParser option names safe.
    try:
        _ph = str(rt.phase)
    except Exception:
        _ph = "NA"
    try:
        _c = "1" if bool(rt.concat_libs) else "0"
    except Exception:
        _c = "0"
    try:
        _lib_names = [os.path.basename(x or "") for x in (rt.libs or [])]
    except Exception:
        _lib_names = []
    libs_md5   = _md5_of_list_str(sorted(_lib_names))[:8]
    params_md5 = _md5_of_list_str([rt.mindepth, rt.maxhits, rt.clustbuffer])[:8]
    return f"ph-{_ph}_c-{_c}_L-{libs_md5}_P-{params_md5}"

def _normalize_path_for_mem(path: str, base_dir: Optional[str] = None) -> str:
    # Compose a ConfigParser-safe key using only [A-Za-z0-9_.-]
    sig = _run_signature_keyprefix().replace('=', '-').replace('|', '_')
    p = (path or '').strip()
    if not p:
        return p
    if p.startswith('~'):
        abspath = p
    else:
        if base_dir is None:
            base_dir = _mem_base_dir()
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        abspath = os.path.abspath(p)
    pmd5 = _md5(abspath.encode('utf-8')).hexdigest()[:10]
    base = os.path.basename(abspath)
    base_sanitized = re.sub(r'[^A-Za-z0-9_.-]', '_', base)
    key = f"{sig}__path-{pmd5}__base-{base_sanitized}"
    key = re.sub(r'[^A-Za-z0-9_.-]', '_', key)
    return key

def _mem_base_dir() -> str:
    """
    Where to anchor relative paths stored in the mem file.
    Prefer rt.outdir if defined; otherwise fall back to CWD.
    """
    base = getattr(rt, "outdir", None)
    if base:
        return os.path.abspath(base)
    return os.path.abspath(os.getcwd())


def _mem_ini_key_for_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Convert a file path into a stable INI key.
    - No expanduser("~") (per your constraint)
    - If path starts with "~", keep it literal
    - Otherwise normalize to an absolute path (anchored at base_dir for relative inputs)
    """
    p = (path or "").strip()
    if not p:
        return p

    # Do not expand "~" (leave literal)
    if p.startswith("~"):
        return p

    if base_dir is None:
        base_dir = _mem_base_dir()

    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)

    return os.path.normpath(os.path.abspath(p))


def sanitize_mem_md5s(config: configparser.ConfigParser,
                      sections: Iterable[str] = ("CLUSTERS", "CLUSTERED")) -> Dict[str, int]:
    """
    Remove empty-string / None MD5s that can poison cache decisions.
    Returns: dict {section: removed_count}
    """
    removed: Dict[str, int] = {}
    for sect in sections:
        cnt = 0
        if config.has_section(sect):
            for k in list(config[sect].keys()):
                v = config[sect].get(k)
                if v is None or not str(v).strip():
                    del config[sect][k]
                    cnt += 1
        removed[sect] = cnt
    return removed


def mem_get(cfg, section, path):
    if not cfg.has_section(section):
        return None

    base_dir = _mem_base_dir()
    k_new  = _mem_ini_key_for_path(path, base_dir=base_dir)  # stable + run-aware
    k_norm = _normalize_path_for_mem(path, base_dir=base_dir) # older style

    for k in (k_new, k_norm, path):
        v = cfg[section].get(k)
        if v is not None:
            return v
    return None


def mem_set(cfg: configparser.ConfigParser, section: str, path: str, value) -> None:
    """
    Store md5 for 'path' into cfg[section] using a stable normalized key.
    """
    if not cfg.has_section(section):
        cfg.add_section(section)

    base_dir = _mem_base_dir()
    key = _mem_ini_key_for_path(path, base_dir=base_dir)
    cfg[section][key] = str(value).strip()

@dataclass(frozen=True)
class MemBasic:
    ok: bool
    index: Optional[str]
    genomehash: Optional[str]
    indexhash: Optional[str]

def read_mem_basic(mem_file: str) -> MemBasic:
    """
    Pure read of the mem INI file.
    - No prints
    - No global writes
    - No dependency on legacy module state
    """
    config = configparser.ConfigParser()
    config.read(mem_file)

    if not config.has_section("BASIC"):
        return MemBasic(False, None, None, None)

    basic = config["BASIC"]
    genomehash = basic.get("genomehash")
    indexhash = basic.get("indexhash")
    index = basic.get("index")

    ok = bool(genomehash and indexhash and index)
    return MemBasic(ok=ok, index=index, genomehash=genomehash, indexhash=indexhash)


def read_mem_verbose(mem_file: str) -> tuple[bool, str, MemBasic]:
    """
    Legacy-compatible mem reader with prints, but no legacy global writes.

    Returns:
        (memflag, index, mem)
    """
    print("#### Fn: memReader ############################")

    mem = read_mem_basic(mem_file)

    if mem.genomehash is not None:
        print("Existing reference hash          :", str(mem.genomehash))

    if mem.indexhash is not None:
        print("Existing index hash              :", str(mem.indexhash))

    if mem.index is not None:
        print("Existing index location          :", str(mem.index))
        index = str(mem.index)
    else:
        index = ""

    return bool(mem.ok), index, mem


def write_mem_basic(
    mem_file: str,
    *,
    ref_hash: str,
    index_path: str,
    index_hash: str,
    mindepth,
    clustbuffer,
    maxhits,
    mismat,
    timestamp: str | None = None,
) -> None:
    """
    Canonical writer for the PHASIS mem file.
    """
    mem_dir = os.path.dirname(mem_file)
    if mem_dir:
        os.makedirs(mem_dir, exist_ok=True)

    config = configparser.ConfigParser()
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")

    config["BASIC"] = {
        "timestamp": timestamp,
        "genomehash": "" if ref_hash is None else str(ref_hash),
        "index": "" if index_path is None else str(index_path),
        "indexhash": "" if index_hash is None else str(index_hash),
    }
    config["ADVANCED"] = {
        "mindepth": "" if mindepth is None else str(mindepth),
        "clustbuffer": "" if clustbuffer is None else str(clustbuffer),
        "maxhits": "" if maxhits is None else str(maxhits),
        "mismat": "" if mismat is None else str(mismat),
    }

    with open(mem_file, "w") as fh_out:
        config.write(fh_out)


__all__ = [
    "MEM_FILE_DEFAULT",
    "CLEANUP_PATTERNS",
    "match_pattern",
    "cleanup",
    "phase2_basename",
    "getmd5",
    "compute_md5_str",
    "md5_file_worker",
    "list_chunk_files_for_prefix",
    "assemble_candidate_from_chunks",
    "sanitize_mem_md5s",
    "mem_get",
    "mem_set",
    "MemBasic",
    "read_mem_basic",
    "read_mem_verbose",
    "write_mem_basic",
]
