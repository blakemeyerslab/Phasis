import os
import time
import collections
import gzip
import phasis.runtime as rt

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
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def isfasta(afile):
    '''
    test if file is fasta format
    '''
    fh_in       = _open_text_maybe_gz(afile)
    firstline   = fh_in.readline()
    fh_in.close()
    if not firstline.startswith('>') and len(firstline.split('\t')) > 1:
        print("\nERROR: File '%s' doesn't seems to be a FASTA" % (afile))
        print("------Please provide correct setting for 'libformat' in 'phasis.set'")
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
        print("------Please provide correct setting for 'libFormat' in 'phasis.set'")
        abool = False
    else:
        abool = True
    return abool

def filter_process(alib):
    '''
    filter tag count file for mindepth, and write
    to FASTA
    '''
    min_depth = int(_resolve_mindepth())
    #print("Writing filtered FASTA for %s" % (alib))
    asum = "%s.sum" % alib.rpartition('.')[0]    # Summary file
    countFile   = "%s.fas" % alib.rpartition('.')[0]  ### Writing in de-duplicated FASTA format
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

def dedup_process(alib):
    '''
    To parallelize the process
    '''
    print("#### Fn: De-duplicater #######################")
    afastaL     = dedup_fastatolist(alib)         ## Read
    acounter    = deduplicate(afastaL )           ## De-duplicate
    fastafile   = dedup_writer(acounter,alib)     ## Write
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

def dedup_writer(acounter,alib):
    '''
    filter tag counts for 'mindepth' parameter, writes a dict
    pickle and filtered fasta file
    '''
    min_depth = int(_resolve_mindepth())
    print("Writing filtered FASTA for %s" % (alib))
    sumFile = "%s.sum" % alib.rpartition('.')[0]    # Summary file

    countFile   = "%s.fas" % alib.rpartition('.')[0]  ### Writing in de-duplicated FASTA format as required for phaster-core
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

def merge_processed_fastas(fas_paths, out_dir, out_basename, mindepth):
    """
    Merge multiple .fas by summing counts per identical sequence.
    Returns the path to the merged .fas.
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
    Stream a .fas produced by your pipeline.
    Header: >seq_<n>|<count>
    Next line: <sequence>
    Yields (sequence, count:int).
    """
    with open(path, 'r') as fh:
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
