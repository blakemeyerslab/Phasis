#!/usr/bin/env python3
"""Supporting external-BAM adapter; clustering/scoring stay in native stages.

Optional dependency: pysam. See docs/external_bam.md for the alignment contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import tempfile

SCHEMA = 2
RC = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reverse_complement(seq):
    return seq.translate(RC)[::-1].upper()


def load_qname_map(path):
    """Explicit original sequence and integer abundance; never infer abundance=1."""
    result = {}
    with open(path) as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'qname', 'sequence', 'abundance'} <= set(reader.fieldnames or []):
            raise ValueError('QNAME mapping needs qname, sequence, abundance columns')
        for row in reader:
            name, seq = row['qname'], row['sequence'].upper()
            abundance = int(row['abundance'])
            if not name or name in result or abundance < 1 or not re.fullmatch('[ACGTN]+', seq):
                raise ValueError(f'Invalid/duplicate QNAME mapping: {name}')
            result[name] = (seq, abundance)
    return result


def abundance_and_sequence(name, records, qmap, abundance_in_qname):
    if qmap is not None:
        if name not in qmap:
            raise ValueError(f'QNAME absent from explicit mapping: {name}')
        seq, abundance = qmap[name]
    else:
        if not abundance_in_qname or not re.search(r'\|[1-9][0-9]*$', name):
            raise ValueError(f'Abundance not established for {name}; supply --qname-map')
        abundance = int(name.rsplit('|', 1)[1])
        seq = None
    # Even secondary SEQ may be omitted; require at least one sequence when no map.
    for record in records:
        if record.query_sequence is None:
            continue
        observed = record.query_sequence.upper()
        if record.is_reverse:
            observed = reverse_complement(observed)
        # Hard-clipped records cannot independently establish the full sequence.
        if record.cigartuples and any(op == 5 for op, _ in record.cigartuples):
            continue
        if seq is None:
            seq = observed
        elif seq != observed:
            raise ValueError(f'SEQ disagrees with original sequence for {name}')
    if seq is None:
        raise ValueError(f'Original sequence unavailable for {name}')
    return seq, abundance


def cigar_is_full_match(cigar, length):
    tokens = re.findall(r'(\d+)([MIDNSHP=X])', cigar)
    return (bool(tokens) and ''.join(n + op for n, op in tokens) == cigar
            and all(op in ('M', '=') and int(n) > 0 for n, op in tokens)
            and sum(int(n) for n, _ in tokens) == length)


def reference_translation(header, fasta, map_path=None):
    mapping = {}
    if map_path:
        with open(map_path) as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not {'old_id', 'clean_id'} <= set(reader.fieldnames or []):
                raise ValueError('Chromosome mapping needs old_id and clean_id columns')
            for row in reader:
                old, clean = row['old_id'], row['clean_id']
                if not old or not clean or old in mapping:
                    raise ValueError(f'Invalid/duplicate chromosome mapping: {old}')
                mapping[old] = clean
    lengths = dict(zip(fasta.references, fasta.lengths))
    translated = {}
    for name, length in zip(header.references, header.lengths):
        # Explicit map takes precedence, preventing ambiguous automatic renaming.
        target = mapping.get(name, name)
        if target not in lengths or lengths[target] != length:
            raise ValueError(f'BAM/reference name or length mismatch: {name} -> {target}')
        if target in translated.values():
            raise ValueError(f'Multiple BAM reference IDs map to {target}')
        translated[name] = target
    if not translated:
        raise ValueError('BAM has no reference dictionary')
    return translated


def placement_cigar(cigar):
    """Unify equivalent ungapped CIGARs before counting reported placements."""
    tokens = re.findall(r'(\d+)([MIDNSHP=X])', cigar or '')
    length = sum(int(n) for n, _ in tokens)
    if cigar_is_full_match(cigar or '', length):
        return f'{length}M'
    return cigar


def expand_group(records, aligner, translation, maxhits, metrics, mismatches=0):
    """Return unique reported placements and their declared multiplicity.

    A placement is (reference, zero-based leftmost start, reverse, CIGAR, NM).
    BWA-backtrack alternatives live in XA rather than secondary SAM records.
    """
    mapped = [r for r in records if not r.is_unmapped]
    if not mapped:
        return [], 0
    for r in mapped:
        if r.is_paired:
            raise ValueError('Paired-end BAMs are unsupported; use collapsed single-end tags')
    allowed = [r for r in mapped if r.flag in (0, 16, 256, 272)]
    metrics['excluded_flag_records'] += len(mapped) - len(allowed)
    if not allowed:
        return [], 0
    count_tag = 'X0' if aligner == 'bwa-aln' else 'NH'
    if aligner != 'bowtie1':
        if any(not r.has_tag(count_tag) for r in allowed):
            raise ValueError(f'{aligner} requires named {count_tag} multiplicity tags')
        counts = {int(r.get_tag(count_tag)) for r in allowed}
        if len(counts) != 1 or min(counts) < 1:
            raise ValueError(f'Inconsistent multiplicity for {allowed[0].query_name}')
        declared = counts.pop()
        if declared >= maxhits:
            return [], declared
        if aligner == 'bwa-aln':
            # X0 counts best hits only; X1 counts aligner's suboptimal hits.
            # Keep the exact-only legacy contract when X1 is absent.
            if mismatches > 0 or any(r.has_tag('X1') for r in allowed):
                if any(not r.has_tag('X1') for r in allowed):
                    raise ValueError('BWA mismatch-enabled input requires consistent X1 tags; '
                                     'regenerate with bwa aln -N and retain X0/X1/XA')
                suboptimal = {int(r.get_tag('X1')) for r in allowed}
                if len(suboptimal) != 1 or min(suboptimal) < 0:
                    raise ValueError('Inconsistent BWA X1 multiplicity')
                declared += suboptimal.pop()
                if declared >= maxhits:
                    return [], declared
    else:
        declared = None
    placements = {}
    for r in allowed:
        if r.reference_name not in translation:
            raise ValueError(f'Unknown BAM reference: {r.reference_name}')
        key = (translation[r.reference_name], r.reference_start, r.is_reverse,
               placement_cigar(r.cigarstring))
        nm = int(r.get_tag('NM')) if r.has_tag('NM') else None
        if key in placements:
            if placements[key] != nm:
                raise ValueError('Duplicate placement with conflicting NM tags')
            metrics['duplicate_alignment_records'] += 1
        placements[key] = nm
        if aligner == 'bwa-aln' and r.has_tag('XA'):
            for entry in r.get_tag('XA').split(';'):
                if not entry:
                    continue
                fields = entry.split(',')
                if len(fields) != 4 or fields[0] not in translation:
                    raise ValueError(f'Malformed BWA XA entry: {entry}')
                chrom, signed, cigar, distance = fields
                if not re.fullmatch(r'[+-][1-9][0-9]*', signed):
                    raise ValueError(f'Invalid XA position: {signed}')
                key = (translation[chrom], abs(int(signed)) - 1, signed[0] == '-',
                       placement_cigar(cigar))
                nm = int(distance)
                if key in placements and placements[key] != nm:
                    raise ValueError('Conflicting XA/record evidence')
                placements[key] = nm
    if declared is None:
        declared = len(placements)
    elif len(placements) != declared:
        raise ValueError(f'Incomplete reported alignments for {allowed[0].query_name}: '
                         f'expected {declared}, found {len(placements)}; check secondary/XA export')
    return [(*key, nm) for key, nm in placements.items()], declared


def accepted_placements(placements, original_sequence, fasta, metrics, mismatches=0):
    accepted = []
    for chrom, start, reverse, cigar, nm in placements:
        if not cigar_is_full_match(cigar or '', len(original_sequence)):
            metrics['rejected_cigar_alignments'] += 1
            continue
        if start < 0 or start + len(original_sequence) > fasta.get_reference_length(chrom):
            # Reject evidence crossing a contig boundary, retaining other valid
            # placements and the original reported-hit denominator.
            metrics['rejected_out_of_bounds_alignments'] += 1
            continue
        seq = reverse_complement(original_sequence) if reverse else original_sequence
        ref = fasta.fetch(chrom, start, start + len(seq)).upper()
        distance = sum(a != b for a, b in zip(seq, ref))
        if (set(seq) - set('ACGT') or set(ref) - set('ACGT')
                or (nm is not None and nm != distance) or distance > mismatches):
            metrics['rejected_mismatch_or_ambiguous_alignments'] += 1
            if nm is not None and nm != distance:
                metrics['rejected_nm_disagreement_alignments'] += 1
            continue
        accepted.append((chrom, start, reverse, seq, distance))
    return sorted(accepted, key=lambda x: (x[0], x[1], x[2]))


def import_bam(bam, reference, output_prefix, *, aligner, qname_map=None,
               abundance_in_qname=False, chrom_map=None, maxhits=25, mindepth=1,
               threads=1, scratch=None, mismatches=0, multimapping='all-reported'):
    """Normalize strict evidence, preserve abundances, and emit native parser inputs.

    Input BAM need not be indexed or sorted. A private query-name sort permits
    bounded per-tag grouping; accepted BAM is coordinate sorted and indexed.
    Native parser dictionaries necessarily scale with retained alignments.
    """
    import pysam
    if maxhits < 2 or mindepth < 1:
        raise ValueError('maxhits must be >=2 and mindepth >=1')
    if not isinstance(mismatches, int) or mismatches < 0:
        raise ValueError('mismatches must be a nonnegative integer')
    if multimapping not in ('all-reported', 'unique-only'):
        raise ValueError('multimapping must be all-reported or unique-only')
    if not Path(str(reference) + '.fai').is_file():
        raise ValueError('Reference must have a .fai index; run samtools faidx first')
    qmap = load_qname_map(qname_map) if qname_map else None
    if qmap is None and not abundance_in_qname:
        raise ValueError('Supply --qname-map or explicitly enable --abundance-in-qname')
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = Counter()
    seen = set()
    accepted_sequences = set()
    nest = defaultdict(lambda: [defaultdict(list)])
    positions = defaultdict(Counter)
    library = prefix.name
    with tempfile.TemporaryDirectory(prefix='phasis-bam-', dir=scratch) as tmp:
        tmp = Path(tmp)
        pysam.quickcheck(str(bam))
        pysam.sort('-n', '-@', str(threads), '-m', '256M', '-T', str(tmp/'sort'),
                   '-o', str(tmp/'names.bam'), str(bam))
        with pysam.FastaFile(str(reference)) as fasta, pysam.AlignmentFile(str(tmp/'names.bam'), 'rb') as inp:
            translation = reference_translation(inp.header, fasta, chrom_map)
            header = {'HD': {'VN': '1.6', 'SO': 'unsorted'},
                      'SQ': [{'SN': n, 'LN': l} for n, l in zip(fasta.references, fasta.lengths)]}
            with pysam.AlignmentFile(str(tmp/'accepted.bam'), 'wb', header=header) as out:
                for name, items in groupby(inp.fetch(until_eof=True), key=lambda r:r.query_name):
                    if name in seen:
                        raise ValueError('QNAME grouping is not contiguous after sort')
                    seen.add(name)
                    records = list(items)
                    seq, abundance = abundance_and_sequence(name, records, qmap, abundance_in_qname)
                    metrics['observed_sequences'] += 1
                    metrics['observed_abundance'] += abundance
                    if abundance < mindepth:
                        metrics['below_mindepth_sequences'] += 1
                        continue
                    placements, declared = expand_group(records, aligner, translation, maxhits,
                                                        metrics, mismatches)
                    if declared:
                        metrics['reported_mapped_sequences'] += 1
                        metrics['reported_mapped_abundance'] += abundance
                    if declared >= maxhits:
                        metrics['rejected_multimapper_sequences'] += 1
                        metrics['rejected_multimapper_abundance'] += abundance
                        continue
                    if multimapping == 'unique-only' and declared > 1:
                        metrics['rejected_unique_only_sequences'] += 1
                        metrics['rejected_unique_only_abundance'] += abundance
                        continue
                    accepted = accepted_placements(placements, seq, fasta, metrics, mismatches)
                    if not accepted:
                        continue
                    # Native ahits uses reported multiplicity before mismatch filtering.
                    hits = declared
                    metrics['accepted_sequences'] += 1
                    metrics['accepted_abundance'] += abundance
                    metrics['accepted_alignment_records'] += len(accepted)
                    metrics['accepted_alignment_weighted_abundance'] += abundance * len(accepted)
                    metrics['unique_sequences' if hits == 1 else 'multimapping_sequences'] += 1
                    if seq in accepted_sequences:
                        raise ValueError('Multiple QNAMEs encode the same accepted collapsed tag; consolidate input first')
                    accepted_sequences.add(seq)
                    canonical_name = name if name.endswith(f'|{abundance}') else f'{name}|{abundance}'
                    for i, (chrom, start, reverse, aligned_seq, distance) in enumerate(accepted):
                        r = pysam.AlignedSegment(out.header)
                        r.query_name = canonical_name
                        r.flag = (16 if reverse else 0) | (256 if i else 0)
                        r.reference_id = out.get_tid(chrom)
                        r.reference_start = start
                        r.mapping_quality = 0  # MAPQ is not comparable across aligners.
                        r.cigarstring = f'{len(seq)}M'
                        r.query_sequence = aligned_seq
                        r.set_tag('NM', distance); r.set_tag('NH', hits)
                        if distance:
                            metrics['accepted_mismatch_alignment_records'] += 1
                        r.set_tag('XC', abundance)
                        out.write(r)
                        key = f'{library}-{chrom}'
                        info = [chrom, 'c' if reverse else 'w', hits, aligned_seq,
                                canonical_name, start + 1, len(seq), abundance]
                        nest[key][0][start + 1].append(info)
                        positions[key][start + 1] += 1
            # Stable native order at ties; no 5-prime coordinate conversion here.
            for value in nest.values():
                for records in value[0].values():
                    records.sort(key=lambda x:(x[4],x[1],x[3]))
        pysam.sort('-@', str(threads), '-m', '256M', '-T', str(tmp/'coord'),
                   '-o', str(prefix)+'.accepted.bam', str(tmp/'accepted.bam'))
        pysam.index(str(prefix)+'.accepted.bam')
    if qmap is not None:
        metrics['input_sequences'] = len(qmap)
        metrics['input_abundance'] = sum(ab for _, ab in qmap.values())
        metrics['input_sequences_absent_from_bam'] = len(set(qmap) - seen)
    counts = {key:OrderedDict(sorted(value.items())) for key,value in positions.items()}
    # Plain dictionaries make these pickles compatible with native stage loaders.
    native_nest = {key:[dict(value[0])] for key,value in nest.items()}
    return native_nest, counts, dict(metrics)


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return value


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bam', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--reference', required=True, help='FASTA matching output names, with .fai')
    p.add_argument('--chrom-map', help='Optional old_id/clean_id TSV mapping BAM to reference IDs')
    p.add_argument('--aligner', required=True, choices=('hisat2','bowtie1','bwa-aln'))
    abundance = p.add_mutually_exclusive_group(required=True)
    abundance.add_argument('--qname-map', help='TSV qname,sequence,abundance')
    abundance.add_argument('--abundance-in-qname', action='store_true', help='Explicitly trust terminal |integer')
    p.add_argument('--phase', type=int, default=24)
    p.add_argument('--output-dir', required=True, help='New empty directory; no implicit cache reuse')
    p.add_argument('--cores', type=int, default=1)
    p.add_argument('--mindepth', type=int, default=1)
    p.add_argument('--maxhits', type=int, default=25, help='Exclusive upper bound, native default: hits <25')
    p.add_argument('--mismatches', type=nonnegative_integer, default=0,
                   help='Maximum substitutions per full-length ungapped alignment [default 0]')
    p.add_argument('--multimapping', choices=('all-reported', 'unique-only'), default='all-reported',
                   help='Retain reported placements below maxhits, or only single-reported-hit tags')
    p.add_argument('--clustbuffer', type=int, default=300)
    p.add_argument('--unique-ratio', type=float, default=0.2)
    p.add_argument('--min-cluster-length', type=int, default=350)
    p.add_argument('--scratch', help='Local temporary directory for BAM sorting')
    return p


def strict_parallel(*args, **kwargs):
    """Do not let native worker-error sentinels become partial candidate output."""
    from phasis.parallel import run_parallel_with_progress
    results = run_parallel_with_progress(*args, **kwargs)
    for item in results or []:
        if isinstance(item, RuntimeError):
            raise item
    return results


def candidate_inventory(paths):
    """Cluster identities, row counts and order-independent row digests."""
    inventory = {}
    for path in paths:
        key = None
        with open(path) as handle:
            for line in handle:
                if line.startswith('>cluster = '):
                    key = line.strip().split(' = ', 1)[1]
                    if key in inventory:
                        raise ValueError(f'Duplicate candidate cluster: {key}')
                    inventory[key] = [0, 0]
                elif line.strip():
                    if key is None or len(line.rstrip('\n').split('\t')) != 18:
                        raise ValueError(f'Malformed native candidate record: {path}')
                    inventory[key][0] += 1
                    inventory[key][1] = (inventory[key][1] + int.from_bytes(
                        hashlib.sha256(line.rstrip('\n').encode()).digest(), 'big')) % (1 << 256)
    return inventory


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', args.library):
        raise ValueError('Library ID must be a safe basename')
    if args.cores < 1:
        raise ValueError('--cores must be positive')
    if args.scratch:
        args.scratch = str(Path(args.scratch).resolve())
        if not Path(args.scratch).is_dir():
            raise ValueError('--scratch must be an existing directory')
    paths = {name:str(Path(getattr(args,name)).resolve()) for name in ('bam','reference')}
    for name in ('qname_map','chrom_map'):
        if getattr(args,name): paths[name] = str(Path(getattr(args,name)).resolve())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty; use a fresh directory to prevent stale caches')
    manifest = {'schema':SCHEMA, 'parameters':vars(args),
                'inputs':{k:{'path':v,'sha256':sha256(v)} for k,v in paths.items()}}
    from phasis import __version__, runtime as rt
    from phasis.cli import build_parser as native_parser, configure_runtime
    from phasis.parallel import coreReserve
    from phasis.stages import cluster_build, cluster_scoring, folder_setup
    manifest['phasis_version'] = __version__
    manifest['adapter_sha256'] = sha256(__file__)
    (out/'import_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previous = Path.cwd()
    previous_build_runner = cluster_build.run_parallel_with_progress
    previous_scoring_runner = cluster_scoring.run_parallel_with_progress
    try:
        cluster_build.run_parallel_with_progress = strict_parallel
        cluster_scoring.run_parallel_with_progress = strict_parallel
        os.chdir(out)
        native = native_parser().parse_args(['-libs', paths['bam'], '-reference', paths['reference'],
                    '-phase', str(args.phase), '-cores', str(args.cores), '-mindepth', str(args.mindepth),
                    '-maxhits', str(args.maxhits), '-mismat',str(args.mismatches), '-clustbuffer',str(args.clustbuffer),
                    '-uniqueRatioCut',str(args.unique_ratio), '-minClusterLength',str(args.min_cluster_length),
                    '-steps','cfind','--reference_id_mode','preserve','--outdir','results'])
        configure_runtime(native)
        rt.ncores = coreReserve(args.cores)
        rt.save_snapshot()
        (out/'resolved_parameters.json').write_text(json.dumps(rt.to_dict(),indent=2)+'\n')
        prefix = out/args.library
        nest, counts, metrics = import_bam(paths['bam'],paths['reference'],prefix,
                 aligner=args.aligner,qname_map=paths.get('qname_map'),
                 abundance_in_qname=args.abundance_in_qname,chrom_map=paths.get('chrom_map'),
                 maxhits=args.maxhits,mindepth=args.mindepth,threads=args.cores,scratch=args.scratch,
                 mismatches=args.mismatches,multimapping=args.multimapping)
        dp,cp = f'{prefix}_{args.phase}.dict',f'{prefix}_{args.phase}.count'
        for path,obj in ((dp,nest),(cp,counts)):
            with open(path,'wb') as handle: pickle.dump(obj,handle,pickle.HIGHEST_PROTOCOL)
        expected_keys = set(counts)
        del nest, counts
        (out/f'{args.library}.sum').write_text(''.join(f'{k}\t{v}\n' for k,v in sorted(metrics.items())))
        (out/'alignment_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        folder = folder_setup.createfolders(str(out))
        clustered = cluster_build.clusterprocess([cp],folder)
        if Counter(item[0] for item in clustered) != Counter({k: 1 for k in expected_keys}):
            raise RuntimeError('Native clustering did not return every input library/chromosome key')
        candidates = cluster_scoring.scoringprocess(libs=[str(prefix)+'.fas'],
                     libs_clustdicts=clustered,libs_nestdict=[dp],clustfolder=folder,concat_mode=False)
        expected = out/f'{args.library}.{args.phase}-PHAS.candidate.clusters'
        if not candidates:
            if clustered:
                raise RuntimeError('Clustering produced work but scoring returned no candidate artifact')
            expected.write_text('')
        if not expected.is_file():
            raise RuntimeError('Expected native candidate file was not produced')
        chunks = [out/f'{args.phase}_scoredClusters'/f'{key}.sRNA_{args.phase}.cluster'
                  for key in sorted(expected_keys)]
        for chunk in chunks:
            if not chunk.is_file():
                raise RuntimeError(f'Native scoring did not produce expected chunk: {chunk}')
        inventory = candidate_inventory(chunks)
        if candidate_inventory([expected]) != inventory:
            raise RuntimeError('Candidate export differs from native scored chunks')
        for entry in manifest['inputs'].values():
            if sha256(entry['path']) != entry['sha256']:
                raise RuntimeError('Input changed during import')
        report = {'status':'PASS','candidate':str(expected),'candidate_sha256':sha256(expected),
                  'clustered_library_chromosome_keys': sorted(expected_keys),
                  'candidate_clusters': len(inventory),
                  'candidate_rows': sum(v[0] for v in inventory.values()),
                  'accepted_bam':str(prefix)+'.accepted.bam',
                  'accepted_bam_sha256':sha256(str(prefix)+'.accepted.bam'),'metrics':metrics}
        (out/'IMPORT_SUCCESS.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        os.chdir(previous)
        cluster_build.run_parallel_with_progress = previous_build_runner
        cluster_scoring.run_parallel_with_progress = previous_scoring_runner
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
