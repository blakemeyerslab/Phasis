import os
import re
import shutil
import subprocess
import sys
import time
import gzip

from phasis import runtime as rt
import phasis.cache as cache
import phasis.index_integrity as index_integrity
from phasis.cache import compute_md5_str, default_memfile_path, read_mem_verbose


# Stage-local globals
reference = None
outdir = None
memFile = default_memfile_path()
ncores = None
runtype = None
mindepth = None
clustbuffer = None
maxhits = None
mismat = None


def sync_from_runtime() -> None:
    """
    Populate indexing-stage globals from phasis.runtime.
    Keep minimal and spawn-safe.
    """
    global reference, outdir, memFile, ncores
    global runtype, mindepth, clustbuffer, maxhits, mismat

    reference = rt.reference
    outdir = rt.outdir
    ncores = rt.cores
    runtype = rt.runtype
    mindepth = rt.mindepth
    clustbuffer = rt.clustbuffer
    maxhits = rt.maxhits
    mismat = rt.mismat

    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(outdir))
        if outdir_abs != outdir:
            outdir = outdir_abs
            rt.outdir = outdir_abs
        os.makedirs(outdir, exist_ok=True)

    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        memFile = default_memfile_path()
        rt.memFile = memFile


def _flush_refclean_record(cur_clean, seq_chunks, fh_out1, fh_out2):
    """
    Write one cleaned FASTA record plus summary entry.
    Returns (written_count_increment, empty_count_increment).
    """
    if cur_clean is None:
        return 0, 0

    seq = "".join(seq_chunks).replace(" ", "").replace("\t", "").replace("\r", "")
    alen = len(seq)

    if alen > 200:
        fh_out1.write(">%s\n%s\n" % (cur_clean, seq))
        fh_out2.write("%s\t%s\n" % (cur_clean, alen))
        return 1, 0

    return 0, 1


def _open_text_maybe_gz(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _archive_file_to_gz(path):
    gz_path = f"{path}.gz"
    if not os.path.isfile(path):
        if os.path.isfile(gz_path):
            return gz_path
        return None

    tmp_path = f"{gz_path}.tmp"

    try:
        with open(path, "rb") as src, open(tmp_path, "wb") as raw_dst:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, mtime=0) as dst:
                shutil.copyfileobj(src, dst)
        os.replace(tmp_path, gz_path)
        os.remove(path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return gz_path


def refClean(filename):
    """
    Cleans FASTA file - multi-line fasta to single line, header clean, empty lines removal.
    For runtype == 'G', forces numeric headers:
      - Chr10/chr01/10 -> 10/1/10
      - non-numeric contigs (Mt, Cp, UNMAPPED, scaffolds, etc.) -> max_numeric+1, +2, ...
    Writes a mapping file: <basename>.chrom_id_map.tsv  (old_id, clean_id)
    """
    global runtype

    sync_from_runtime()

    print("Phasis uses FASTA header as key for identifying the phased loci")
    print("Caching '%s' reference FASTA file" % (filename))

    base = filename.rpartition('/')[-1].rpartition('.')[0]
    fastaclean = "%s/%s.clean.fa" % (os.getcwd(), base)
    fastasumm = "%s/%s.summ.txt" % (os.getcwd(), base)
    mapfile = "%s/%s.chrom_id_map.tsv" % (os.getcwd(), base)

    orig_order = []
    max_numeric = 0
    numeric_candidate = {}

    chr_re = re.compile(r'^(?:chr|Chr|CHR)?0*([0-9]+)$')

    with _open_text_maybe_gz(filename) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue

            orig = line[1:].split()[0].strip()
            orig_order.append(orig)

            if runtype == "G":
                match = chr_re.match(orig)
                if match:
                    val = int(match.group(1))
                    if val > 0:
                        numeric_candidate[orig] = val
                        if val > max_numeric:
                            max_numeric = val
                    else:
                        numeric_candidate[orig] = None
                else:
                    numeric_candidate[orig] = None
            else:
                numeric_candidate[orig] = orig

    mapping = {}
    used = set()

    if runtype == "G":
        for orig in orig_order:
            val = numeric_candidate.get(orig, None)
            if isinstance(val, int):
                clean = str(val)
                if clean in used:
                    mapping[orig] = None
                else:
                    mapping[orig] = clean
                    used.add(clean)
            else:
                mapping[orig] = None

        next_id = max_numeric + 1
        for orig in orig_order:
            if mapping[orig] is None:
                while str(next_id) in used:
                    next_id += 1
                mapping[orig] = str(next_id)
                used.add(str(next_id))
                next_id += 1
    else:
        for orig in orig_order:
            mapping[orig] = str(numeric_candidate[orig])

    with open(mapfile, "w") as mf:
        mf.write("old_id\tclean_id\n")
        for orig in orig_order:
            mf.write("%s\t%s\n" % (orig, mapping[orig]))

    print("Chromosome/contig ID equivalence table (also saved to %s):" % (mapfile))
    if len(orig_order) <= 50:
        for orig in orig_order:
            print("  %s\t=>\t%s" % (orig, mapping[orig]))
    else:
        for orig in orig_order[:25]:
            print("  %s\t=>\t%s" % (orig, mapping[orig]))
        print("  ... (%d more) ..." % (len(orig_order) - 35))
        for orig in orig_order[-10:]:
            print("  %s\t=>\t%s" % (orig, mapping[orig]))

    acount = 0
    empty_count = 0
    cur_clean = None
    seq_chunks = []

    with open(fastaclean, "w") as fh_out1, open(fastasumm, "w") as fh_out2:
        fh_out2.write("Name\tLen\n")

        with _open_text_maybe_gz(filename) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    wrote_inc, empty_inc = _flush_refclean_record(cur_clean, seq_chunks, fh_out1, fh_out2)
                    acount += wrote_inc
                    empty_count += empty_inc
                    seq_chunks = []

                    cur_orig = line[1:].split()[0].strip()
                    cur_clean = mapping.get(cur_orig, None)
                    if not cur_clean:
                        cur_clean = None
                else:
                    if cur_clean is not None:
                        seq_chunks.append(line.strip())

        wrote_inc, empty_inc = _flush_refclean_record(cur_clean, seq_chunks, fh_out1, fh_out2)
        acount += wrote_inc
        empty_count += empty_inc

    print("Fasta file with reduced header: '%s' with total entries %s is prepared" % (fastaclean, acount))
    print("There were %s entries found with empty sequences and were removed\n" % (empty_count))

    return fastaclean, fastasumm


def indexBuilder(reference, ncores):
    """
    Generic index building module.
    """
    global memFile, mindepth, clustbuffer, maxhits, mismat

    sync_from_runtime()

    print("#### Fn: indexBuilder #######################")
    fastaclean, fastasumm = refClean(reference)
    _ = fastasumm

    print("**Deleting old index")
    shutil.rmtree('./index', ignore_errors=True)
    os.mkdir('./index')

    genoIndex = '%s/index/%s' % (os.getcwd(), fastaclean.rpartition('/')[-1].rpartition('.')[0])
    print('Creating index of cDNA/genomic sequences:%s**\n' % (genoIndex))

    ncores = str(ncores)
    retcode = subprocess.call(["hisat2-build", "-p", ncores, "-f", fastaclean, genoIndex])
    print(retcode)

    if retcode != 0:
        print("There is some problem preparing index of reference '%s'" % (reference))
        print("Is Hisat2 installed? And added to environment variable?")
        print("Script will exit now")
        sys.exit()

    print("Generating fingerprint for HiSat2 index")
    try:
        refHash, indexHash, _index_marker = index_integrity.compute_index_fingerprints(
            reference=reference,
            genoIndex=genoIndex,
            compute_fingerprint_fn=cache.compute_md5_str,
        )
    except FileNotFoundError:
        print("File extension for index couldn't be determined properly")
        print("It could be an issue from 'HiSat2'")
        print("This needs to be reported to 'PHASIS' developer, report issue here\nhttps://github.com/atulkakrana/PHASIS/issues")
        print("Script will exit")
        sys.exit()

    cache.write_mem_basic(
        memFile,
        ref_hash=refHash,
        index_path=genoIndex,
        index_hash=indexHash,
        mindepth=mindepth,
        clustbuffer=clustbuffer,
        maxhits=maxhits,
        mismat=mismat,
    )
    _archive_file_to_gz(fastaclean)
    print("Index prepared:%s\n" % (genoIndex))
    return genoIndex


def getindex(fh_run):
    """
    Stage version of legacy.getindex(fh_run), preserving behavior/prints.
    """
    global reference, memFile, ncores

    sync_from_runtime()

    if not os.path.isfile(memFile):
        print("This is first run - create index")
        indexflag = False
    else:
        memflag, index, mem = read_mem_verbose(memFile)

        if memflag is False:
            print("Memory file is empty - seems like previous run crashed")
            print("Creating index")
            indexflag = False
        else:
            exist_ref_hash = str(mem.genomehash) if mem.genomehash is not None else ""
            current_ref_hash = compute_md5_str(reference) or ""

            if current_ref_hash != exist_ref_hash:
                print("Index status                     : Re-make")
                indexflag = False
                print("Existing index does not matches specified genome - It will be recreated")
            elif not index:
                print("Index status                     : Re-make")
                indexflag = False
                print("Existing index path missing from memory file - It will be recreated")
            else:
                indexIntegrity, indexExt = index_integrity.indexIntegrityCheck(index)
                _ = indexExt

                if indexIntegrity:
                    print("Index status                     : Re-use")
                    genoIndex = index
                    indexflag = True
                    fh_run.write("Indexing Time: 0s\n")
                else:
                    print("Index status                     : Re-make")
                    indexflag = False

    if indexflag is False:
        tstart = time.time()
        genoIndex = indexBuilder(reference, ncores)
        tend = time.time()
        fh_run.write("Indexing Time:%ss\n" % (round(tend - tstart, 2)))

    print("Index to be used:%s" % (genoIndex))
    return genoIndex
