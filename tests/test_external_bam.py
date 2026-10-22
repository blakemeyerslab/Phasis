import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import pysam
except ImportError:
    pysam = None
from phasis.external_bam import import_bam, cigar_is_full_match, main


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
             patch('phasis.external_bam.import_bam', side_effect=RuntimeError('test stop')) as importer:
            with self.assertRaisesRegex(RuntimeError, 'test stop'):
                main(['--bam', str(self.root / 'input.bam'), '--reference', str(self.ref),
                      '--library', 'sample', '--aligner', 'hisat2', '--abundance-in-qname',
                      '--output-dir', str(output), '--scratch', os.path.relpath(scratch)])
        self.assertEqual(importer.call_args.kwargs['scratch'], str(scratch.resolve()))
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
