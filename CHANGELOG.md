# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- One-command dataset download (`scripts/download_data.py`): Figshare is fetched
  automatically; SARTAJ instructions are printed since it needs a Kaggle account.
- Static type checking (`mypy`) as a CI step.

### Changed
- Renamed the CI workflow from `tests.yml` to `ci.yml` (now runs lint, type check, and tests).

### Fixed
- README and RESULTS.md now give the referral result that `uncertainty.py` reports at 80% coverage:
  79.3% to 87.7% (`r5_referral.csv` reads 0.8767 at coverage 0.8001). The figure quoted before was
  the value at about 77.6% coverage.
- README abstract: the leakage gap now reads 5.8 points, the 10-seed result in `r1_summary.csv`;
  the abstract had kept an earlier figure.
- Seed counts in the README, `docs/METHODS.md` and the Makefile: R1 runs over 10 seeds (0 to 9),
  matching the reported result; R2 to R5 use seed 0.

## [1.0.0] - 2026-08-19

First public release: a reproducible audit of a brain-tumour MRI classifier.

### Added
- R1: patient-level leakage, reported over 10 seeds with 95% confidence intervals and a paired t-test.
- Source-independence audit (perceptual-hash near-duplicate check between Figshare and SARTAJ).
- R2: robustness under controlled ImageNet-C-style corruptions.
- R3: diagnosis via Grad-CAM, a feature embedding, and a domain-classifier AUC.
- R4: label-free mitigations (matched normalisation, histogram equalisation, AdaBN).
- R5: temperature scaling and referral-based selective prediction.
- Test suite, continuous integration, methods documentation, result figures and tables.
- Zenodo DOI for citation.
