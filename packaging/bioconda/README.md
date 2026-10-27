# Bioconda recipe preparation

Status: Phasis 2.8.4 was released and the recipe submitted on 2026-10-27:
[Bioconda PR #70852](https://github.com/bioconda/bioconda-recipes/pull/70852).
CI and maintainer approval are pending; public channel availability is not yet
confirmed. The recipe now uses the published tag archive and its verified
SHA256. Candidate packaging was tested on macOS and Linux/Hive. See the
[Hive validation record](../HIVE_VALIDATION_21.md).

## Scope

The recipe installs the native Phasis package and both CLI entry points.
It declares the Python dependencies, HISAT2, and SAMtools. The pure-Python
package is `noarch: python`; actual platform availability still depends on
the compiled dependencies. `matplotlib-base` provides Matplotlib without
unnecessary GUI dependencies on compute nodes.

The auxiliary BAM importer stays in `support_scripts/` in the source checkout.
It is not a native Phasis command and is not included in the wheel. Its
optional `pysam` dependency and Bowtie 1/BWA executables are not required for
standard Phasis installation. For that workflow, obtain the matching source
and follow `docs/external_bam.md`.

## Local testing before a release

Use a disposable recipe copy. Replace its entire `source` block with a local
source archive URL and the archive's actual SHA256, or a local source path.
Do not substitute a moving development branch into the submission recipe.

In a dedicated build environment with `conda-build` installed:

```bash
conda build /path/to/disposable/recipe --no-anaconda-upload \
  --override-channels -c conda-forge -c bioconda
```

Do not upload test packages. Test installation of the resulting
package in a second clean environment and run the recipe tests there.
Successful local `conda build` is useful but does not replace Bioconda CI.

To test installation from a locally indexed output channel, use a package
specification rather than mixing an artifact filename with other specs:

```bash
conda create -p /path/to/new/test-env --override-channels \
  -c conda-forge -c bioconda -c file:///path/to/conda-build-output \
  python=3.12 phasis=2.8.4=py_0 -y
```

The channel path must contain the generated `noarch/repodata.json`. Keep these
test artifacts private. On Hive, a disposable build environment's Conda
launcher used `/usr/bin/env python`, which resolved to the wrong Python after
build PATH changes. Pinning that launcher's interpreter to its own environment
resolved the failure; this is a build-environment repair, not a Phasis patch
or a standard installation requirement. Preserve logs/backups and ask the HPC
administrator before repairing any shared Conda installation.

## Release source and submission checklist

The published `v2.8.4` tag points to commit
`4c1cd8f5252bae21e95c39463df89b34e78d26c3`. Its GitHub tag archive SHA256 is
`3a5b5f54af566a304174151238512e60fec78c8d022c5e0e4b3bb3dfe581dd62`.
This differs from the separately built source-distribution asset, as expected.
The finalized recipe checksum is recorded after tagging to avoid a
self-referential source hash; the release tag has not been changed.

For future submissions:

1. Approve and freeze the final v2.8.4 source after installation and usage tests.
   A GitHub Release page is not technically required by Conda: a stable,
   accessible source archive with a checksum is required. The recipe uses a
   version tag, so that tag/archive must exist before the recipe can
   be submitted as written. Publishing the approved release first is simplest.
2. Download that exact tag archive, compute SHA256, and replace the placeholder.
   Do not use the checksum of a local test ZIP or a differently generated tarball.
3. Recheck the name in Bioconda, conda-forge, and defaults to avoid a collision,
   especially with historical software also named PHASIS. Confirm the package
   points to `blakemeyerslab/Phasis`, not the original implementation.
4. With explicit approval, fork `bioconda/bioconda-recipes`, make a branch,
   and copy `meta.yaml` into `recipes/phasis/`. Follow the current repository's
   contribution instructions, linting, build configuration, and platform tests.
5. With explicit approval, open a pull request. Address CI and maintainer review.
   Acceptance, merge, and channel publication depend on Bioconda maintainers.
6. After publication, verify a fresh channel installation and basic analysis.
   Only then add a working Bioconda installation command to the user README.

References:
[recipe guidelines](https://bioconda.github.io/contributor/guidelines.html),
[submission workflow](https://bioconda.github.io/contributor/workflow.html).
