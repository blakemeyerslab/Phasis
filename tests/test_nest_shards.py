from __future__ import annotations

import io
import os
import pickle
import random
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from unittest import mock

from phasis import cache, nest_shards
from phasis import runtime as rt
from phasis.stages import cluster_scoring


def _serial_parallel_runner(func, iterable, **_kwargs):
    return [func(item) for item in list(iterable)]


def _random_tags(rng, akeys, phase=21):
    """Parser-style {akey: [(position, taginfo), ...]} with phased and random tags."""
    tags = {}
    for akey in akeys:
        chrom = akey.rsplit("-", 1)[-1]
        entries = []
        for cluster in range(6):
            start = 1000 + cluster * 5000
            for k in range(40):
                strand = rng.choice("wc")
                if rng.random() < 0.7:
                    pos = start + rng.randint(0, 14) * phase - (3 if strand == "c" else 0)
                else:
                    pos = start + rng.randint(0, 320)
                # Mostly phase-length tags so that the clusters pass the scoring pre-filter.
                length = phase if rng.random() < 0.9 else rng.choice([22, 24])
                # Low abundances: the pre-filter needs more phase-length tags than
                # a quarter of the cluster's total abundance.
                abun = rng.choice([1, 1, 1, 2]) if length == phase else 1
                name = f"t{cluster}_{k}|{abun}"
                entries.append((pos, [chrom, strand, rng.choice([1, 1, 2]), "A" * length, name, pos, length, abun]))
        tags[akey] = entries
    return tags


def _legacy_nestdict(tags):
    nestdict = defaultdict(list)
    for akey, entries in tags.items():
        by_position = defaultdict(list)
        for position, taginfo in entries:
            by_position[position].append(taginfo)
        nestdict[akey].append(by_position)
    return nestdict


def _lclust_for(tags, akey):
    """One cluster per 5-kb block of positions, as cluster_build would group them."""
    clusters = defaultdict(set)
    for position, _taginfo in tags[akey]:
        clusters[f"cluster{position // 5000}"].add(position)
    return {cid: sorted(positions) for cid, positions in clusters.items()}


class NestShardFileTests(unittest.TestCase):
    def test_write_then_expand_equals_legacy_structure(self):
        rng = random.Random(11)
        tags = _random_tags(rng, ["libA-1", "libA-2", "libA-Mt"])
        expected = dict(_legacy_nestdict(tags))
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "libA_21.dict")
            consumed = {k: list(v) for k, v in tags.items()}
            index = nest_shards.write_nest_shards(index_path, consumed)

            self.assertEqual(consumed, {})  # emptied while written
            self.assertEqual(sorted(index), sorted(expected))
            self.assertTrue(all(nest_shards.is_ref_list(v) for v in index.values()))
            self.assertLess(os.path.getsize(index_path), 4096)
            self.assertEqual(len(os.listdir(nest_shards.shard_dir_for(index_path))), 3)

            expanded = nest_shards.load_expanded(index_path)
            self.assertEqual(expanded, expected)
            self.assertIsInstance(expanded["libA-1"][0], defaultdict)

    def test_index_changes_when_tags_change(self):
        rng = random.Random(12)
        tags = _random_tags(rng, ["libA-1"])
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "libA_21.dict")
            nest_shards.write_nest_shards(index_path, {k: list(v) for k, v in tags.items()})
            with open(index_path, "rb") as handle:
                first = handle.read()
            tags["libA-1"][0][1][7] += 1  # one abundance differs
            nest_shards.write_nest_shards(index_path, {k: list(v) for k, v in tags.items()})
            with open(index_path, "rb") as handle:
                second = handle.read()
        self.assertNotEqual(first, second)

    def test_references_follow_a_moved_run_directory(self):
        rng = random.Random(13)
        tags = _random_tags(rng, ["libA-1"])
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.path.join(tmpdir, "old")
            os.makedirs(old)
            nest_shards.write_nest_shards(os.path.join(old, "libA_21.dict"), {k: list(v) for k, v in tags.items()})
            new = os.path.join(tmpdir, "new")
            os.rename(old, new)
            expanded = nest_shards.load_expanded(os.path.join(new, "libA_21.dict"))
        self.assertEqual(expanded, dict(_legacy_nestdict(tags)))

    def test_missing_shard_is_reported_and_legacy_files_pass(self):
        rng = random.Random(14)
        tags = _random_tags(rng, ["libA-1", "libA-2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "libA_21.dict")
            nest_shards.write_nest_shards(index_path, {k: list(v) for k, v in tags.items()})
            self.assertTrue(nest_shards.index_is_complete(index_path))
            os.remove(os.path.join(nest_shards.shard_dir_for(index_path), "shard_00002.p"))
            self.assertFalse(nest_shards.index_is_complete(index_path))

            legacy_path = os.path.join(tmpdir, "libB_21.dict")
            with open(legacy_path, "wb") as handle:
                pickle.dump(_legacy_nestdict(tags), handle)
            self.assertTrue(nest_shards.index_is_complete(legacy_path))

    def test_cleanup_removes_shard_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "21_results"))
            lib_dir = os.path.join(tmpdir, "processed_libraries_x")
            os.makedirs(lib_dir)
            index_path = os.path.join(lib_dir, "libA_21.dict")
            nest_shards.write_nest_shards(index_path, {"libA-1": [(5, ["1", "w", 1, "A", "t|1", 5, 21, 1])]})
            with mock.patch.object(cache.rt, "outdir", os.path.join(tmpdir, "21_results"), create=True):
                cache.cleanup(tmpdir)
            self.assertFalse(os.path.exists(index_path))
            self.assertFalse(os.path.exists(nest_shards.shard_dir_for(index_path)))


class ScoringWithShardsTests(unittest.TestCase):
    def _score(self, tmpdir, name, tags, sharded):
        workdir = os.path.join(tmpdir, name)
        os.makedirs(workdir)
        akeys = sorted(tags)
        lclust_inputs = []
        for akey in akeys:
            path = os.path.join(workdir, f"{akey}.lclust")
            with open(path, "wb") as handle:
                pickle.dump(_lclust_for(tags, akey), handle)
            lclust_inputs.append((akey, path))
        dict_path = os.path.join(workdir, "libA_21.dict")
        if sharded:
            nest_shards.write_nest_shards(dict_path, {k: [(p, list(t)) for p, t in v] for k, v in tags.items()})
        else:
            with open(dict_path, "wb") as handle:
                pickle.dump(_legacy_nestdict({k: [(p, list(t)) for p, t in v] for k, v in tags.items()}), handle)
        lib_path = os.path.join(workdir, "libA.fas")
        with open(lib_path, "w", encoding="utf-8") as handle:
            handle.write(">dummy\nTAG\n")
        scored_dir = os.path.join(workdir, "scored")

        old_cwd = os.getcwd()
        os.chdir(workdir)
        captured = io.StringIO()
        try:
            with mock.patch.multiple(
                rt, phase=21, mismat=0, maxhits=25, clustbuffer=300, uniqueRatioCut=0.0,
                memFile=os.path.join(workdir, "phasis.mem"), ncores=1, create=True,
            ), mock.patch.object(
                cluster_scoring, "run_parallel_with_progress", side_effect=_serial_parallel_runner
            ), redirect_stdout(captured):
                outputs = cluster_scoring.scoringprocess(
                    [lib_path], lclust_inputs, [dict_path], workdir,
                    force_rescore=True, verify_outputs=False, scored_dir=scored_dir, concat_mode=False,
                )
        finally:
            os.chdir(old_cwd)

        files = {}
        for fname in sorted(os.listdir(scored_dir)):
            with open(os.path.join(scored_dir, fname), encoding="utf-8") as handle:
                files[fname] = handle.read()
        with open(os.path.join(workdir, outputs[0]), encoding="utf-8") as handle:
            candidates = handle.read()
        return files, candidates, captured.getvalue()

    def test_shards_give_the_same_scored_clusters_as_inline_tags(self):
        rng = random.Random(15)
        tags = _random_tags(rng, ["libA-1", "libA-2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            inline_files, inline_candidates, inline_log = self._score(tmpdir, "inline", tags, sharded=False)
            shard_files, shard_candidates, shard_log = self._score(tmpdir, "shards", tags, sharded=True)

        self.assertTrue(inline_files)
        scored_tags = sum(
            1 for text in inline_files.values() for line in text.splitlines() if line and not line.startswith(">")
        )
        self.assertGreater(scored_tags, 100)
        self.assertEqual(inline_files, shard_files)
        self.assertEqual(inline_candidates, shard_candidates)
        self.assertIn("scoring workers read their own .lclust and tag files", shard_log)
        self.assertNotIn("scoring workers read their own", inline_log)


if __name__ == "__main__":
    unittest.main()
