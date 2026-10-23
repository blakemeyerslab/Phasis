"""Opt-in small real-aligner test; requires installed Phasis[bam] and aligners.

Run: python tests/run_external_bam_integration.py --output-dir /tmp/bam-check
Uses no network, private datasets, or scheduler. Never reuses an output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys

import pysam


def native_candidates(work, bam, reference, mismatches=0, maxhits=25):
    from phasis.cli import build_parser, configure_runtime
    from phasis import runtime as rt
    from phasis.parallel import coreReserve
    from phasis.deps_check import require_dependencies
    from phasis.stages import sam_parsing, cluster_build, cluster_scoring, folder_setup

    work.mkdir()
    os.chdir(work)
    (work / 'sample.bam').symlink_to(bam)
    configure_runtime(build_parser().parse_args([
        '-libs', str(work / 'sample.fas'), '-reference', str(reference),
        '-phase', '24', '-cores', '2', '-steps', 'cfind',
        '-mismat', str(mismatches), '-maxhits', str(maxhits)]))
    require_dependencies()
    rt.ncores = coreReserve(2)
    rt.save_snapshot()
    dp, cp = sam_parsing.samparser_streaming((str(work / 'sample.bam'), maxhits, mismatches))
    folder = folder_setup.createfolders(str(work))
    clustered = cluster_build.clusterprocess([cp], folder)
    cluster_scoring.scoringprocess([str(work / 'sample.fas')], clustered, [dp],
                                  folder, concat_mode=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mismatches', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--multimapping', choices=('all-reported', 'unique-only'), default='all-reported')
    parser.add_argument('--reporting-only', action='store_true',
                        help='Only check real BWA best/suboptimal-hit handling on a tiny reference')
    args = parser.parse_args()
    tools = {name: shutil.which(name) for name in
             ('hisat2', 'hisat2-build', 'bowtie', 'bowtie-build', 'bwa', 'samtools')}
    if not all(tools.values()):
        parser.error('Missing executables: ' + ', '.join(k for k, v in tools.items() if not v))
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'scratch').mkdir()
    commands = []

    def run(argv, log, cwd=root, output=None):
        argv = list(map(str, argv))
        commands.append({'argv': argv, 'cwd': str(cwd), 'stdout_file': str(output) if output else log})
        with (root / log).open('w') as logs:
            if output:
                with Path(output).open('w') as stream:
                    subprocess.run(argv, cwd=cwd, stdout=stream, stderr=logs, check=True)
            else:
                subprocess.run(argv, cwd=cwd, stdout=logs, stderr=logs, check=True)
        (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')

    # Exercise X0+X1 with real BWA output, not only hand-constructed BAM tags.
    reporting_reference = root / 'reporting.fa'
    reporting_fastq = root / 'reporting.fastq'
    sequence = 'ACGTGCAATGCTAGCTACGATCGA'
    variant = 'T' + sequence[1:]
    reporting_reference.write_text('>1\n' + sequence + 'N'*100 + variant + '\n')
    reporting_fastq.write_text('@reporting|11\n' + sequence + '\n+\n' + 'I'*24 + '\n')
    pysam.faidx(str(reporting_reference))
    run([tools['bwa'], 'index', reporting_reference], 'reporting_index.log')
    run([tools['bwa'], 'aln', '-n', '1', '-o', '0', '-e', '0', '-l', '1024', '-N', '-q', '0',
         reporting_reference, reporting_fastq], 'reporting_aln.log', output=root/'reporting.sai')
    run([tools['bwa'], 'samse', '-n', '25', reporting_reference, root/'reporting.sai', reporting_fastq],
        'reporting_samse.log', output=root/'reporting.sam')
    run([tools['samtools'], 'view', '-b', '-o', root/'reporting.bam', root/'reporting.sam'],
        'reporting_convert.log')
    with pysam.AlignmentFile(str(root/'reporting.bam'), 'rb') as stream:
        row = next(stream.fetch(until_eof=True))
        assert row.get_tag('X0') == 1 and row.get_tag('X1') == 1
    for limit, policy, accepted in ((0, 'all-reported', 1), (1, 'all-reported', 2), (1, 'unique-only', 0)):
        destination = root/f'reporting_{limit}_{policy}'
        run([sys.executable, Path(__file__).resolve().parents[1]/'support_scripts/bam_to_candidates.py',
             '--bam', root/'reporting.bam', '--reference', reporting_reference, '--library', 'reporting',
             '--aligner', 'bwa-aln', '--abundance-in-qname', '--mismatches', str(limit),
             '--multimapping', policy, '--output-dir', destination], f'reporting_{limit}_{policy}.log')
        success = json.loads((destination/'IMPORT_SUCCESS.json').read_text())
        assert success['metrics'].get('accepted_alignment_records', 0) == accepted
        with pysam.AlignmentFile(str(destination/'reporting.accepted.bam'), 'rb') as stream:
            rows = list(stream.fetch(until_eof=True))
        assert all(row.get_tag('NH') == 2 and row.get_tag('XC') == 11 for row in rows)
        assert sorted(row.get_tag('NM') for row in rows) == ([0, 1] if accepted == 2 else [0] if accepted else [])
    if args.reporting_only:
        (root/'REPORTING_SUCCESS.json').write_text(json.dumps({'status': 'PASS', 'X0': 1, 'X1': 1})+'\n')
        return

    rng = random.Random(1701)
    genome, tags = '', []
    for locus in range(16):
        block = ''.join(rng.choices('ACGT', k=1600))
        offsets = range(0, 960, 24) if locus < 12 else sorted(rng.sample(range(960), 40))
        for i, start in enumerate(offsets):
            tags.append((block[start:start + 24], 10 + i + locus))
        for length, start in [(20, 10), (21, 50), (22, 100), (23, 200)]:
            tags.append((block[start:start + length], 1))
        genome += block + 'N' * 500
    genome += ('N' * 50 + 'A' * 24) * 27
    alternate = next(base for base in 'ACGT' if base != tags[0][0][0])
    tags.extend([('A' * 24, 12), ('C' * 24, 5), (alternate + tags[0][0][1:], 9)])
    reference, fasta, fastq, qmap = (root / n for n in
                                    ('ref.fa', 'sample.fas', 'sample.fastq', 'sample.qnames.tsv'))
    reference.write_text('>1\n' + genome + '\n')
    pysam.faidx(str(reference))
    with fasta.open('w') as fa, fastq.open('w') as fq, qmap.open('w') as mapping:
        writer = csv.writer(mapping, delimiter='\t')
        writer.writerow(['qname', 'sequence', 'abundance'])
        for i, (seq, abundance) in enumerate(tags):
            name = f'seq_{i + 1}|{abundance}'
            fa.write(f'>{name}\n{seq}\n')
            fq.write(f'@{name}\n{seq}\n+\n' + 'I' * len(seq) + '\n')
            writer.writerow([name, seq, abundance])
    for tool in ('hisat2', 'bowtie', 'samtools'):
        run([tools[tool], '--version'], tool + '_version.log')
    version = subprocess.run([tools['bwa']], text=True, capture_output=True)
    (root / 'bwa_version.log').write_text(version.stdout + version.stderr)

    run([tools['hisat2-build'], reference, root / 'hidx'], 'hisat_index.log')
    run([tools['hisat2'], '--no-softclip', '--no-spliced-alignment',
         '--score-min', f'L,{-6 * args.mismatches},0', '--mp', '6,6',
         '-k', '25', '-p', '2', '-x', root / 'hidx', '-f', fasta, '-S', root / 'hisat2.sam'], 'hisat.log')
    run([tools['bowtie-build'], reference, root / 'btidx'], 'bowtie_index.log')
    run([tools['bowtie'], '-f', '-v', str(args.mismatches), '-k', '25', '--best', '-S', '-p', '2',
         root / 'btidx', fasta, root / 'bowtie1.sam'], 'bowtie.log')
    run([tools['bwa'], 'index', reference], 'bwa_index.log')
    run([tools['bwa'], 'aln', '-n', str(args.mismatches), '-o', '0', '-e', '0', '-l', '1024', '-N', '-q', '0',
         '-t', '2', reference, fastq], 'bwa_aln.log', output=root / 'sample.sai')
    run([tools['bwa'], 'samse', '-n', '25', reference, root / 'sample.sai', fastq],
        'bwa_samse.log', output=root / 'bwa-aln.sam')

    metrics = {}
    expected_names = {f'seq_{i + 1}|{ab}' for i, (_, ab) in enumerate(tags[:-3])}
    repeat_name = f'seq_{len(tags) - 2}|12'
    mismatch_name = f'seq_{len(tags)}|9'
    if args.mismatches:
        expected_names.add(mismatch_name)
    for arm in ('hisat2', 'bowtie1', 'bwa-aln'):
        run([tools['samtools'], 'view', '-b', '-o', root / f'{arm}.bam', root / f'{arm}.sam'],
            arm + '_convert.log')
        abundance = ['--abundance-in-qname'] if arm == 'bwa-aln' else ['--qname-map', qmap]
        run([sys.executable, Path(__file__).resolve().parents[1] / 'support_scripts/bam_to_candidates.py',
             '--bam', root / f'{arm}.bam',
             '--aligner', arm, '--reference', reference, '--library', 'sample', *abundance,
             '--mismatches', str(args.mismatches), '--multimapping', args.multimapping,
             '--cores', '2', '--scratch', 'scratch', '--output-dir', root / arm], arm + '_import.log')
        success = json.loads((root / arm / 'IMPORT_SUCCESS.json').read_text())
        assert success['status'] == 'PASS'
        metrics[arm] = success['metrics']
        with pysam.AlignmentFile(str(root / arm / 'sample.accepted.bam'), 'rb') as stream:
            records = list(stream.fetch(until_eof=True))
            observed = {r.query_name for r in records}
            mismatch_records = [r for r in records if r.query_name == mismatch_name]
            if args.mismatches:
                assert mismatch_records and all(r.get_tag('NM') == 1 for r in mismatch_records), arm
            else:
                assert not mismatch_records, arm
        assert observed - {repeat_name} == expected_names, arm
    for arm in ('bowtie1', 'bwa-aln'):
        assert metrics[arm]['rejected_multimapper_sequences'] == 1

    run([sys.executable, Path(__file__).resolve(), '--native', root / 'native',
         root / 'hisat2.bam', reference, str(args.mismatches),
         '2' if args.multimapping == 'unique-only' else '25'], 'native.log')
    for suffix in ('.dict', '.count'):
        with (root / 'native' / ('sample_24' + suffix)).open('rb') as stream:
            expected = pickle.load(stream)
        with (root / 'hisat2' / ('sample_24' + suffix)).open('rb') as stream:
            assert pickle.load(stream) == expected, suffix
    candidate = 'sample.24-PHAS.candidate.clusters'
    native_rows = sorted((root / 'native' / candidate).read_text().splitlines())
    assert native_rows and native_rows == sorted((root / 'hisat2' / candidate).read_text().splitlines())

    call_rows = {}
    for arm in ('native', 'hisat2', 'bowtie1', 'bwa-aln'):
        work = root / (arm + '_calls')
        work.mkdir()
        run([sys.executable, '-c', 'from phasis.cli import main; raise SystemExit(main())',
             '-steps', 'class', '-phase', '24', '-libformat', 'F', '-cores', '2',
             '-mismat', str(args.mismatches),
             '-libs', fasta, '-reference', reference, '--reference_id_mode', 'preserve',
             '-class_cluster_files', root / arm / candidate, '--outdir', 'results'],
            arm + '_calls.log', cwd=work)
        for name in ('24_calls.tsv', '24_all_clusters.tsv', '24_classification_evidence.tsv',
                     '24_PHAS.gff', '24_phasiRNAs.tsv'):
            assert (work / 'results' / name).is_file(), (arm, name)
        assert not (work / 'index').exists(), 'Classification unexpectedly created a mapping index'
        call_rows[arm] = sorted((work / 'results/24_calls.tsv').read_text().splitlines())
        assert len(call_rows[arm]) > 1, arm
    assert call_rows['native'] == call_rows['hisat2']
    from phasis import __version__
    import numpy
    import sklearn
    report = {'status': 'PASS', 'phasis_version': __version__, 'tools': tools,
              'mismatches': args.mismatches, 'multimapping': args.multimapping,
              'bwa_best_suboptimal_reporting_checks': True,
              'python': sys.version, 'numpy': numpy.__version__,
              'scikit_learn': sklearn.__version__, 'pysam': pysam.__version__,
              'adapter_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] /
                                               'support_scripts/bam_to_candidates.py').read_bytes()).hexdigest(),
              'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'native_hisat_import_dictionary_candidate_call_parity': True,
              'standard_outputs_all_arms': True,
              'call_rows': {arm: len(rows) - 1 for arm, rows in call_rows.items()},
              'metrics': metrics,
              'repeat_control': '27 true copies; reported multiplicity is not assumed exhaustive'}
    (root / 'INTEGRATION_SUCCESS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--native':
        native_candidates(*map(Path, sys.argv[2:5]), *map(int, sys.argv[5:7]))
    else:
        main()
