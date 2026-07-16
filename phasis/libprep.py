import os
import time
import collections
import gzip
import phasis.runtime as rt
from phasis.fastq import count_fastq_tags

# Module-level default used by legacy-style call sites.
# Workers read the value from phasis.runtime when this stays unset.
mindepth = None


def _resolve_mindepth():
    """Return the effective mindepth in both fork and spawn workers."""
    if mindepth is not None:
        return mindepth

    rt_mindepth = getattr(rt, "mindepth", None)
    if rt_mindepth is not None:
        return rt_mindepth

    raise RuntimeError(
        "mindepth is not initialized in phasis.libprep; ensure the runtime snapshot is available to workers"
    )


def _open_text_maybe_gz(path):
    path = str(path)
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    if os.path.isfile(path):
        return open(path, "r")
    gz_path = f"{path}.gz"
    if os.path.isfile(gz_path):
        return gzip.open(gz_path, "rt")
    return open(path, "r")


def _input_stem(path):
    base = os.path.basename(str(path))
    if base.lower().endswith(".gz"):
        base = base[:-3]
    for ext in (".fastq", ".fq", ".fasta", ".fa", ".tag"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    stem, dot, _ = base.rpartition(".")
    return stem if dot else base


def _default_output_paths(alib, out_fas=None, out_sum=None):
    countFile = out_fas if out_fas is not None else f"{_input_stem(alib)}.fas"
    sumFile = out_sum if out_sum is not None else f"{countFile.rpartition('.')[0]}.sum"
    return countFile, sumFile

def isfasta(afile):
    '''
    test if file is fasta format
    '''
    fh_in       = _open_text_maybe_gz(afile)
    firstline   = fh_in.readline()
    fh_in.close()
    if not firstline.startswith('>') and len(firstline.split('\t')) > 1:
        print("\nERROR: File '%s' doesn't seems to be a FASTA" % (afile))
        print("------Please provide correct setting for '-libformat'")
        abool = False
    else:
        abool = True
    return abool

def isfiletagcount(afile):
    '''
    test if file is tab seprated tag and counts file
    '''
    fh_in       = _open_text_maybe_gz(afile)
    firstline   = fh_in.readline()
    fh_in.close()
    if firstline.startswith('>') or len(firstline.split('\t')) != 2 :
        print("\nERROR: File '%s' doesn't seems to be tab-seprated tag-count format" % (afile))
        print("------Please provide correct setting for '-libformat'")
        abool = False
    else:
        abool = True
    return abool


def isfastq(afile):
    '''
    test if file is FASTQ format
    '''
    fh_in = _open_text_maybe_gz(afile)
    line1 = fh_in.readline()
    line2 = fh_in.readline()
    line3 = fh_in.readline()
    line4 = fh_in.readline()
    fh_in.close()
    if not (line1.startswith('@') and bool(line2) and line3.startswith('+') and bool(line4)):
        print("\nERROR: File '%s' doesn't seems to be a FASTQ" % (afile))
        print("------Please provide correct setting for '-libformat'")
        abool = False
    else:
        abool = True
    return abool

def filter_process(alib, out_fas=None, out_sum=None):
    '''
    filter tag count file for mindepth, and write
    to FASTA
    '''
    min_depth = int(_resolve_mindepth())
    countFile, asum = _default_output_paths(alib, out_fas=out_fas, out_sum=out_sum)
    fh_out      = open(countFile,'w')
    fh_in       = _open_text_maybe_gz(alib)
    aread       = fh_in.readlines()
    bcount      = 0 ## tags written
    ccount      = 0 ## tags excluded
    seqcount    = 1 ## To name seqeunces
    for aline in aread:
        atag,acount    = aline.strip("\n").split("\t")
        if int(acount) >= min_depth:
            fh_out.write(">seq_%s|%s\n%s\n" % (seqcount,acount,atag))
            bcount      += 1
            seqcount    += 1
        else:
            ccount+=1
    #print("Library %s - tag written:%s | tags filtered:%s" % (alib,bcount,ccount))
    with open(asum, 'a') as fh_sum:
        fh_sum.write("Library %s - tag written:%s | tags filtered:%s\n" % (alib, bcount, ccount))
    fh_in.close()
    fh_out.close()
    return countFile

def dedup_process(alib, out_fas=None, out_sum=None):
    '''
    To parallelize the process
    '''
    print("#### Fn: De-duplicater #######################")
    afastaL     = dedup_fastatolist(alib)         ## Read
    acounter    = deduplicate(afastaL )           ## De-duplicate
    fastafile   = dedup_writer(acounter, alib, out_fas=out_fas, out_sum=out_sum)     ## Write
    return fastafile

def dedup_fastatolist(alib):
    '''
    New FASTA reader
    '''
    ## Output
    fastaL      = [] ## List that holds FASTA tags
    ## input
    fh_in       = _open_text_maybe_gz(alib)
    print("Reading FASTA file:%s" % (alib))
    read_start  = time.time()
    acount      = 0
    empty_count = 0
    for line in fh_in:
        if line.startswith('>'):
            seq = ''
            pass
        else:
          seq = line.rstrip('\n')
          fastaL.append(seq)
          acount += 1
    read_end    = time.time()
    print("Cached file: %s | Tags: %s | Empty headers: %ss" % (alib,acount,empty_count))
    fh_in.close()
    return fastaL

def deduplicate(afastaL):
    '''
    De-duplicates tags using multiple processes and libraries using multiple cores
    '''
    dedup_start = time.time()
    acounter    = collections.Counter(afastaL)
    dedup_end   = time.time()
    return acounter

def dedup_writer(acounter,alib, out_fas=None, out_sum=None):
    '''
    filter tag counts for 'mindepth' parameter, writes a dict
    pickle and filtered fasta file
    '''
    min_depth = int(_resolve_mindepth())
    print("Writing filtered FASTA for %s" % (alib))
    countFile, sumFile = _default_output_paths(alib, out_fas=out_fas, out_sum=out_sum)
    fh_out      = open(countFile,'w')
    wcount      = 0 ## tags written
    bcount      = 0 ## tags excluded
    seqcount    = 1 ## To name seqeunces
    for atag,acount in acounter.items():
        if int(acount) >= min_depth:
            fh_out.write(">seq_%s|%s\n%s\n" % (seqcount,acount,atag))
            wcount      += 1
            seqcount    += 1
        else:
            bcount+=1
    with open(sumFile, 'w') as fh_sum:
        fh_sum.write("Library %s - tag written:%s | tags filtered:%s\n" % (alib, wcount, bcount))
    #print("Library %s - tag written:%s | tags filtered:%s" % (alib,wcount,bcount))
    fh_out.close()
    return countFile


def fastq_process(alib, out_fas=None, out_sum=None):
    '''
    Converts a preprocessed sRNA FASTQ into de-duplicated FASTA counts.
    Records are streamed and filtered before they enter the tag counter.
    '''
    print("#### Fn: FASTQ Processor #####################")
    print(
        f"[INFO] Streaming preprocessed sRNA FASTQ: {alib} "
        "(progress reports every 1,000,000 reads; raw adapter-containing input is rejected).",
        flush=True,
    )
    seq_counter, stats = count_fastq_tags(alib, progress_callback=_fastq_progress_report)
    count_file = dedup_writer(seq_counter, alib, out_fas=out_fas, out_sum=out_sum)
    _, sum_file = _default_output_paths(alib, out_fas=out_fas, out_sum=out_sum)
    with open(sum_file, "a") as fh_sum:
        values = stats.as_dict(len(seq_counter))
        fh_sum.write(
            "FASTQ reads examined:{reads_examined} | retained:{reads_retained} | "
            "rejected_length:{reads_rejected_length} | rejected_ambiguous:{reads_rejected_ambiguous} | "
            "chopped_at_N:{reads_chopped_at_n} | unique_retained_tags:{unique_retained_tags}\n".format(**values)
        )
    return count_file


def _fastq_progress_report(stats, _delta, final):
    suffix = " complete" if final else ""
    print(
        "[INFO] FASTQ reads examined:{0} | retained:{1} | rejected_length:{2} | "
        "rejected_ambiguous:{3}{4}".format(
            stats.reads_examined,
            stats.reads_retained,
            stats.reads_rejected_length,
            stats.reads_rejected_ambiguous,
            suffix,
        ),
        flush=True,
    )

def merge_processed_fastas(fas_paths, out_dir, out_basename, mindepth):
    """
    Merge multiple processed FASTA count files by summing counts per sequence.
    Accepts either plain `.fas` or canonical `.fas.gz` artifacts.
    Returns the path to the merged plain `.fas`.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_basename}.fas")

    counter = collections.Counter()
    for p in fas_paths:
        for seq, cnt in fas_records(p):
            counter[seq] += cnt

    write_merged_fas(counter, out_path, mindepth)
    return out_path

def fas_records(path):
    """
    Stream a processed FASTA count file produced by the pipeline.
    Header: >seq_<n>|<count>
    Next line: <sequence>
    Yields (sequence, count:int).
    """
    with _open_text_maybe_gz(path) as fh:
        count_val = None
        for line in fh:
            if line.startswith('>'):
                # Example: >seq_123|45
                parts = line.split('|', 1)
                if len(parts) < 2:
                    raise ValueError(f"Malformed header in {path}: {line.strip()}")
                try:
                    count_val = int(parts[1].strip())
                except Exception:
                    raise ValueError(f"Non-integer count in {path}: {line.strip()}")
            else:
                seq = line.rstrip('\n')
                if not seq:
                    continue
                if count_val is None:
                    raise ValueError(f"Sequence without header in {path}")
                yield (seq, count_val)
                count_val = None

def write_merged_fas(seq_counter, out_path, mindepth):
    """
    Write a merged .fas applying mindepth to merged totals.
    Also writes a .sum sidecar like your per-lib writers.
    """
    wcount = 0
    bcount = 0
    seqnum = 1
    with open(out_path, 'w') as out_fh:
        for seq, total in seq_counter.items():
            if int(total) >= int(mindepth):
                out_fh.write(f">seq_{seqnum}|{total}\n{seq}\n")
                wcount += 1
                seqnum += 1
            else:
                bcount += 1
    with open(f"{out_path.rpartition('.')[0]}.sum", 'w') as fh_sum:
        fh_sum.write(
            f"Merged library {os.path.basename(out_path)} - tags written:{wcount} | tags filtered:{bcount}\n"
        )
    return out_path
