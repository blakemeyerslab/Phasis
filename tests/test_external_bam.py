import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import pysam
except ImportError:
    pysam = None
from support_scripts.bam_to_candidates import import_bam, cigar_is_full_match, main, build_parser


@unittest.skipIf(pysam is None, 'Install Phasis[bam] to test the optional BAM importer')
class ExternalBamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.root/'ref.fa'
        self.ref.write_text('>1\n'+'A'*200+'\n')
        pysam.faidx(str(self.ref))
        self.header = pysam.AlignmentHeader.from_dict({'HD':{'VN':'1.6'}, 'SQ':[{'SN':'1','LN':200}]})

    def record(self, name='tag|7', start=0, flag=0, seq='A'*24, cigar='24M', nh=1, nm=0):
        r = pysam.AlignedSegment(self.header)
        r.query_name=name;r.query_sequence=seq;r.flag=flag
        r.reference_id=0;r.reference_start=start;r.cigarstring=cigar
        if nh is not None:r.set_tag('NH',nh)
        if nm is not None:r.set_tag('NM',nm)
        return r

    def run_import(self, records, aligner='hisat2', mapping=None, **kw):
        bam=self.root/'input.bam'
        with pysam.AlignmentFile(str(bam),'wb',header=self.header) as f:
            for r in records:f.write(r)
        qmap=None
        if mapping is not None:
            qmap=self.root/'q.tsv'
            qmap.write_text('qname\tsequence\tabundance\n'+''.join(f'{n}\t{s}\t{a}\n' for n,s,a in mapping))
        return import_bam(bam,self.ref,self.root/'lib',aligner=aligner,qname_map=qmap,
                          abundance_in_qname=mapping is None,**kw)

    def test_abundance_and_secondary_multimaps(self):
        nest,counts,m=self.run_import([self.record(nh=2),self.record(start=40,flag=256,nh=2)])
        self.assertEqual(m['accepted_abundance'],7)
        self.assertEqual(m['accepted_alignment_weighted_abundance'],14)
        self.assertEqual(nest['lib-1'][0][41][0][2],2)
        self.assertEqual(nest['lib-1'][0][41][0][-1],7)
        self.assertEqual(dict(counts['lib-1']),{1:1,41:1})

    def test_reverse_leftmost_one_based_coordinate(self):
        nest,_,_=self.run_import([self.record(start=20,flag=16)],mapping=[('tag|7','T'*24,7)])
        info=nest['lib-1'][0][21][0]
        self.assertEqual(info[1],'c');self.assertEqual(info[3],'A'*24);self.assertEqual(info[5],21)

    def test_mismatch_rejected_by_reference_even_without_nm(self):
        _,_,m=self.run_import([self.record(seq='C'+'A'*23,nm=None)])
        self.assertEqual(m.get('accepted_sequences',0),0)
        self.assertEqual(m['rejected_mismatch_or_ambiguous_alignments'],1)

    def test_nonzero_named_nm_rejected(self):
        _,_,m=self.run_import([self.record(nm=1)])
        self.assertEqual(m.get('accepted_sequences',0),0)

    def test_clipping_gaps_splicing_and_x_rejected(self):
        for cigar in ('22M2S','12M1I11M','12M1D12M','12M20N12M','12M1P12M','23M1X'):
            with self.subTest(cigar=cigar):
                _,_,m=self.run_import([self.record(cigar=cigar)])
                self.assertEqual(m.get('accepted_sequences',0),0)
        self.assertFalse(cigar_is_full_match('2H22M',24))
        self.assertTrue(cigar_is_full_match('24=',24))

    def test_exclusive_multimapping_limit(self):
        for n in (24,25):
            _,_,m=self.run_import([self.record(start=i,flag=0 if i==0 else 256,nh=n) for i in range(n)])
            self.assertEqual(m.get('accepted_sequences',0),int(n==24))

    def test_user_hit_cutoff_preserves_full_abundance_per_placement(self):
        for cutoff in (2, 3):
            with self.subTest(cutoff=cutoff):
                nest, _, m = self.run_import([
                    self.record(nh=2), self.record(start=40, flag=256, nh=2)], maxhits=cutoff)
                self.assertEqual(m.get('accepted_sequences', 0), int(cutoff == 3))
                if cutoff == 3:
                    self.assertEqual(m['accepted_alignment_weighted_abundance'], 14)
                    self.assertEqual(nest['lib-1'][0][41][0][-1], 7)

    def test_bowtie_cannot_infer_unreported_hits(self):
        _, _, m = self.run_import([self.record(nh=None)], aligner='bowtie1')
        # One supplied placement is all the adapter can observe, even if the
        # generation command omitted other hits. Provenance is indispensable.
        self.assertEqual(m['unique_sequences'], 1)

    def test_qname_map_retains_absent_read_denominator(self):
        _, _, m = self.run_import([self.record()], mapping=[
            ('tag|7', 'A' * 24, 7), ('unmapped', 'C' * 24, 9)])
        self.assertEqual(m['input_abundance'], 16)
        self.assertEqual(m['input_sequences_absent_from_bam'], 1)
        self.assertEqual(m['accepted_abundance'], 7)

    def test_invalid_hit_cutoff_fails(self):
        with self.assertRaisesRegex(ValueError, 'maxhits must be >=2'):
            self.run_import([self.record()], maxhits=1)

    def test_external_mismatches_configurable_and_nonnegative(self):
        args = ['--bam', 'sample.bam', '--library', 'sample', '--reference', 'ref.fa',
                '--aligner', 'bowtie1', '--abundance-in-qname', '--output-dir', 'out']
        self.assertEqual(build_parser().parse_args(args).mismatches, 0)
        self.assertEqual(build_parser().parse_args(args + ['--mismatches', '2']).mismatches, 2)
        with patch('sys.stderr'), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(args + ['--mismatches', '-1'])
        self.assertEqual(error.exception.code, 2)

    def test_mismatch_limit_and_output_nm(self):
        for limit in (0, 1, 2):
            for distance in (1, 2):
                with self.subTest(limit=limit, distance=distance):
                    seq = 'C' * distance + 'A' * (24 - distance)
                    _, _, metrics = self.run_import([self.record(seq=seq, nm=distance)], mismatches=limit)
                    self.assertEqual(metrics.get('accepted_sequences', 0), int(distance <= limit))
                    with pysam.AlignmentFile(str(self.root/'lib.accepted.bam'), 'rb') as bam:
                        records = list(bam.fetch(until_eof=True))
                    if records:
                        self.assertEqual(records[0].get_tag('NM'), distance)
                        self.assertEqual(records[0].query_sequence, seq)

    def test_mismatch_without_nm_is_computed_from_reference(self):
        _, _, m = self.run_import([self.record(seq='C'+'A'*23, nm=None)], mismatches=1)
        self.assertEqual(m['accepted_mismatch_alignment_records'], 1)

    def test_mismatch_nm_disagreement_rejected(self):
        for nm in (0, 2, -1):
            _, _, m = self.run_import([self.record(seq='C'+'A'*23, nm=nm)], mismatches=2)
            self.assertEqual(m.get('accepted_sequences', 0), 0)
            self.assertEqual(m['rejected_nm_disagreement_alignments'], 1)

    def test_reverse_mismatch_keeps_original_abundance(self):
        seq = 'C' + 'A'*23
        nest, _, _ = self.run_import([self.record(seq=seq, flag=16, nm=1)],
                                     mapping=[('tag|7', 'T'*23+'G', 7)], mismatches=1)
        self.assertEqual(nest['lib-1'][0][1][0], ['1', 'c', 1, seq, 'tag|7', 1, 24, 7])

    def test_mismatch_filter_does_not_reduce_reported_hits(self):
        self.ref.write_text('>1\n' + 'A'*40 + 'C'*24 + 'A'*136 + '\n')
        Path(str(self.ref)+'.fai').unlink(); pysam.faidx(str(self.ref))
        rows = [self.record(nh=2), self.record(start=40, flag=256, nh=2, nm=24)]
        nest, _, m = self.run_import(rows, mismatches=1)
        self.assertEqual(m['accepted_alignment_records'], 1)
        self.assertEqual(nest['lib-1'][0][1][0][2], 2)
        self.assertEqual(nest['lib-1'][0][1][0][-1], 7)
        _, _, m = self.run_import(rows, mismatches=1, multimapping='unique-only')
        self.assertEqual(m.get('accepted_sequences', 0), 0)
        self.assertEqual(m['rejected_unique_only_sequences'], 1)

    def test_unique_only_is_single_reported_placement_not_proof_of_uniqueness(self):
        for hits in (1, 2):
            rows = [self.record(nh=hits)]
            if hits == 2:
                rows.append(self.record(start=40, flag=256, nh=hits))
            _, _, m = self.run_import(rows, multimapping='unique-only')
            self.assertEqual(m.get('accepted_sequences', 0), int(hits == 1))

    def test_invalid_mismatch_and_mapping_policy(self):
        for kw in ({'mismatches': -1}, {'mismatches': 1.5}, {'multimapping': 'fractional'}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_import([self.record()], **kw)

    def test_mismatch_allowance_still_rejects_ambiguous_sequence(self):
        _, _, m = self.run_import([self.record(seq='N'+'A'*23, nm=1)], mismatches=2)
        self.assertEqual(m.get('accepted_sequences', 0), 0)
        self.ref.write_text('>1\nN'+'A'*199+'\n')
        Path(str(self.ref)+'.fai').unlink(); pysam.faidx(str(self.ref))
        _, _, m = self.run_import([self.record(nm=1)], mismatches=2)
        self.assertEqual(m.get('accepted_sequences', 0), 0)

    def test_mismatch_allowance_does_not_enable_gaps_or_clipping(self):
        for cigar in ('22M2S', '12M1I11M', '12M1D12M', '12M20N12M', '23M1X'):
            _, _, m = self.run_import([self.record(cigar=cigar)], mismatches=2)
            self.assertEqual(m.get('accepted_sequences', 0), 0)

    def test_bwa_best_and_suboptimal_hits_are_both_counted(self):
        self.ref.write_text('>1\n'+'A'*40+'C'+'A'*159+'\n')
        Path(str(self.ref)+'.fai').unlink(); pysam.faidx(str(self.ref))
        r = self.record(nh=None)
        r.set_tag('X0', 1); r.set_tag('X1', 1); r.set_tag('XA', '1,+41,24M,1;')
        for limit in (0, 1):
            nest, _, m = self.run_import([r], aligner='bwa-aln', mismatches=limit)
            self.assertEqual(m['accepted_alignment_records'], 1+limit)
            self.assertEqual(nest['lib-1'][0][1][0][2], 2)
        _, _, m = self.run_import([r], aligner='bwa-aln', mismatches=1, maxhits=2)
        self.assertEqual(m['rejected_multimapper_sequences'], 1)
        _, _, m = self.run_import([r], aligner='bwa-aln', mismatches=1, multimapping='unique-only')
        self.assertEqual(m['rejected_unique_only_sequences'], 1)

    def test_bwa_mismatch_mode_requires_x1_below_cutoff(self):
        r = self.record(nh=None); r.set_tag('X0', 1)
        with self.assertRaisesRegex(ValueError, 'X1'):
            self.run_import([r], aligner='bwa-aln', mismatches=1)
        r.set_tag('X0', 25)
        _, _, m = self.run_import([r], aligner='bwa-aln', mismatches=1)
        self.assertEqual(m['rejected_multimapper_sequences'], 1)

    def test_bwa_incomplete_or_invalid_suboptimal_evidence_fails(self):
        r = self.record(nh=None); r.set_tag('X0', 1); r.set_tag('X1', 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_import([r], aligner='bwa-aln', mismatches=1)
        r.set_tag('X1', -1)
        with self.assertRaisesRegex(ValueError, 'X1'):
            self.run_import([r], aligner='bwa-aln', mismatches=1)

    def test_mismatch_parser_parity_with_native_on_same_evidence(self):
        from phasis import runtime as rt
        from phasis.stages import sam_parsing
        seq = 'C'+'A'*23
        rows = [self.record(seq=seq, nm=1, nh=2), self.record(seq=seq, nm=1, nh=2, start=40, flag=256)]
        expected, counts, _ = self.run_import(rows, mismatches=1)
        # Native parser expects its HISAT2 tag order; compare identical evidence.
        lines = ['\t'.join([r.query_name, str(r.flag), '1', str(r.reference_start+1), '0',
                            '24M', '*', '0', '0', seq, '*', 'NM:i:1', 'MD:Z:0A23',
                            'AS:i:-6', 'XS:i:-6', 'XN:i:0', 'XM:i:1', 'NH:i:2'])+'\n' for r in rows]
        with patch.object(rt, 'phase', 24, create=True), patch.object(rt, 'norm', False, create=True), \
             patch.object(sam_parsing, '_iter_alignment_lines', return_value=iter(lines)):
            dp, cp = sam_parsing.samparser_streaming((str(self.root/'lib.bam'), 25, 1))
        with open(dp, 'rb') as stream:
            self.assertEqual(pickle.load(stream), expected)
        with open(cp, 'rb') as stream:
            self.assertEqual(pickle.load(stream), counts)

    def test_duplicate_records_not_double_counted(self):
        _,_,m=self.run_import([self.record(),self.record()])
        self.assertEqual(m['accepted_alignment_records'],1)
        self.assertEqual(m['duplicate_alignment_records'],1)

    def test_equivalent_cigars_not_double_counted(self):
        for aligner in ('hisat2', 'bowtie1'):
            with self.subTest(aligner=aligner):
                _, _, m = self.run_import([
                    self.record(cigar='24M'), self.record(cigar='24='),
                    self.record(cigar='12M12=')], aligner=aligner)
                self.assertEqual(m['accepted_alignment_records'], 1)
                self.assertEqual(m['accepted_alignment_weighted_abundance'], 7)
                self.assertEqual(m['duplicate_alignment_records'], 2)

    def test_equivalent_bwa_xa_not_double_counted(self):
        r = self.record(nh=None)
        r.set_tag('X0', 1)
        r.set_tag('XA', '1,+1,24=,0;')
        _, _, m = self.run_import([r], aligner='bwa-aln')
        self.assertEqual(m['accepted_alignment_records'], 1)

    def test_invalid_chromosome_maps_fail(self):
        mapping = self.root / 'chromosomes.tsv'
        for content in ('old_id\tclean_id\n1\tother\n1\t1\n',
                        'old_id\tclean_id\n1\t\n', 'wrong\theader\n1\t1\n'):
            with self.subTest(content=content):
                mapping.write_text(content)
                with self.assertRaisesRegex(ValueError, '[Cc]hromosome mapping'):
                    self.run_import([self.record()], chrom_map=mapping)

    def test_explicit_chromosome_map(self):
        mapping = self.root / 'chromosomes.tsv'
        mapping.write_text('old_id\tclean_id\nChr1\t1\n')
        self.header = pysam.AlignmentHeader.from_dict({'SQ': [{'SN': 'Chr1', 'LN': 200}]})
        _, _, m = self.run_import([self.record()], chrom_map=mapping)
        self.assertEqual(m['accepted_alignment_records'], 1)

    def test_empty_qname_map_entry_fails(self):
        with self.assertRaisesRegex(ValueError, 'Invalid/duplicate QNAME'):
            self.run_import([self.record()], mapping=[('', 'A' * 24, 7)])

    def test_main_resolves_scratch_and_restores_workers_after_failure(self):
        from phasis.stages import cluster_build, cluster_scoring
        self.run_import([self.record()])
        scratch = self.root / 'scratch'
        scratch.mkdir()
        output = self.root / 'work'
        original_cwd = Path.cwd()
        build_runner = cluster_build.run_parallel_with_progress
        scoring_runner = cluster_scoring.run_parallel_with_progress
        with patch('phasis.cli.configure_runtime'), \
             patch('phasis.runtime.save_snapshot'), \
             patch('phasis.parallel.coreReserve', return_value=1), \
             patch('support_scripts.bam_to_candidates.import_bam', side_effect=RuntimeError('test stop')) as importer:
            with self.assertRaisesRegex(RuntimeError, 'test stop'):
                main(['--bam', str(self.root / 'input.bam'), '--reference', str(self.ref),
                      '--library', 'sample', '--aligner', 'hisat2', '--abundance-in-qname',
                      '--output-dir', str(output), '--scratch', os.path.relpath(scratch),
                      '--mismatches', '1', '--multimapping', 'unique-only'])
        self.assertEqual(importer.call_args.kwargs['scratch'], str(scratch.resolve()))
        self.assertEqual(importer.call_args.kwargs['mismatches'], 1)
        self.assertEqual(importer.call_args.kwargs['multimapping'], 'unique-only')
        manifest = json.loads((output/'import_manifest.json').read_text())
        self.assertEqual(manifest['schema'], 2)
        self.assertEqual(manifest['parameters']['mismatches'], 1)
        self.assertEqual(manifest['parameters']['multimapping'], 'unique-only')
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertIs(cluster_build.run_parallel_with_progress, build_runner)
        self.assertIs(cluster_scoring.run_parallel_with_progress, scoring_runner)
        self.assertFalse((output / 'IMPORT_SUCCESS.json').exists())

    def test_supplementary_excluded(self):
        _,_,m=self.run_import([self.record(),self.record(flag=2048,start=40)])
        self.assertEqual(m['accepted_alignment_records'],1)
        self.assertEqual(m['excluded_flag_records'],1)

    def test_unmapped_excluded(self):
        r=self.record(flag=4);r.reference_id=-1;r.reference_start=-1;r.cigarstring=None
        _,_,m=self.run_import([r])
        self.assertEqual(m.get('accepted_sequences',0),0)

    def test_bwa_xa_expansion(self):
        r=self.record(nh=None);r.set_tag('X0',2);r.set_tag('XA','1,+41,24M,0;')
        _,_,m=self.run_import([r],aligner='bwa-aln')
        self.assertEqual(m['accepted_alignment_records'],2)
        self.assertEqual(m['accepted_abundance'],7)

    def test_incomplete_bwa_xa_fails(self):
        r=self.record(nh=None);r.set_tag('X0',2)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            self.run_import([r],aligner='bwa-aln')

    def test_bwa_boundary_xa_rejected_preserves_valid_hit_and_multiplicity(self):
        r=self.record(nh=None);r.set_tag('X0',2);r.set_tag('XA','1,+190,24M,0;')
        nest,_,m=self.run_import([r],aligner='bwa-aln')
        self.assertEqual(m['rejected_out_of_bounds_alignments'],1)
        self.assertEqual(m['accepted_alignment_records'],1)
        self.assertEqual(m['accepted_abundance'],7)
        self.assertEqual(nest['lib-1'][0][1][0][2],2)

    def test_boundary_primary_rejected(self):
        nest,counts,m=self.run_import([self.record(start=190)])
        self.assertFalse(nest);self.assertFalse(counts)
        self.assertEqual(m['rejected_out_of_bounds_alignments'],1)
        self.assertEqual(m.get('accepted_sequences',0),0)

    def test_match_ending_at_reference_boundary_is_valid(self):
        _,_,m=self.run_import([self.record(start=176)])
        self.assertEqual(m['accepted_alignment_records'],1)
        self.assertEqual(m.get('rejected_out_of_bounds_alignments',0),0)

    def test_bwa_reverse_xa_uses_original_orientation(self):
        self.ref.write_text('>1\n'+'A'*40+'T'*24+'A'*136+'\n')
        Path(str(self.ref)+'.fai').unlink();pysam.faidx(str(self.ref))
        r=self.record(nh=None);r.set_tag('X0',2);r.set_tag('XA','1,-41,24M,0;')
        nest,_,m=self.run_import([r],aligner='bwa-aln')
        self.assertEqual(m['accepted_alignment_records'],2)
        info=nest['lib-1'][0][41][0]
        self.assertEqual((info[1],info[3],info[5]),('c','T'*24,41))

    def test_missing_hisat_nh_fails(self):
        with self.assertRaisesRegex(ValueError,'requires named NH'):
            self.run_import([self.record(nh=None)])

    def test_bowtie_counts_all_reported_records(self):
        _,_,m=self.run_import([self.record(nh=None),self.record(start=40,flag=256,nh=None)],aligner='bowtie1')
        self.assertEqual(m['multimapping_sequences'],1)

    def test_missing_abundance_fails(self):
        with self.assertRaisesRegex(ValueError,'Abundance not established'):
            self.run_import([self.record(name='unannotated')])

    def test_explicit_mapping_for_arbitrary_qname(self):
        _,_,m=self.run_import([self.record(name='plain')],mapping=[('plain','A'*24,19)])
        self.assertEqual(m['accepted_abundance'],19)

    def test_sequence_mapping_disagreement_fails(self):
        with self.assertRaisesRegex(ValueError,'SEQ disagrees'):
            self.run_import([self.record()],mapping=[('tag|7','C'*24,7)])

    def test_incomplete_hisat_records_fail(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            self.run_import([self.record(nh=2)])

    def test_reference_length_mismatch_fails(self):
        self.ref.write_text('>1\n'+'A'*199+'\n');Path(str(self.ref)+'.fai').unlink();pysam.faidx(str(self.ref))
        with self.assertRaisesRegex(ValueError,'length mismatch'):
            self.run_import([self.record()])

    def test_paired_end_fails(self):
        with self.assertRaisesRegex(ValueError,'Paired-end'):
            self.run_import([self.record(flag=1)])

    def test_duplicate_collapsed_sequence_fails(self):
        with self.assertRaisesRegex(ValueError,'Multiple QNAMEs'):
            self.run_import([self.record(),self.record(name='other|4',start=40)])
