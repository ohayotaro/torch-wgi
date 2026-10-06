# torch-wgi

Research and development repository for a differentiable analytical forward projector for Whole Gamma Imaging (WGI).

## Current Status

**This repository is currently in the research-planning and validation-specification stage. The simulator, tests, Monte Carlo comparisons, and hardware-optimization experiments have not yet been implemented or executed.**

All numerical thresholds currently documented in this repository are proposed engineering acceptance criteria for this research program. They are not achieved results, external standards, or claims of clinical performance.

## Research Definition

The project models a three-gamma decay as a joint image-space likelihood: the PET response and the Compton response are evaluated for the same source location and integrated over the image basis.

The Cone-conditioned Tube Projector is a low-dimensional approximation of this joint likelihood integral. It is **not** an algebraic solver for geometric LOR–cone intersection coordinates.

Core modeling rules:

- For a finite voxel, multiply the PET and Compton responses at the same sub-voxel location before integrating over the voxel.
- Preserve absolute detection probability, including branching, positron range, attenuation, detector efficiency, rejected/lost events, and coincidence acceptance.
- Keep artificial SDF/soft-binning relaxation widths separate from physical SRF widths.
- Optimize task information under fixed source decays and acquisition time, then validate optimized designs with independent Monte Carlo data and, when available, measurements.
- Treat diagnostic Monte Carlo truth variables as latent variables; do not leak true DOI, interaction order, or emission position into the observable model.

The central source model is

$$
A^J_{ij}(\theta)
=
\beta_3\int
\varphi_j(x)
P_\theta(y_i^P\mid x)
C_\theta(y_i^C\mid x)
\,dx,
$$

with expected observations

$$
\lambda_i
=
\sum_j A^J_{ij}(\theta)f_j+b_i.
$$

## Documentation

- [Engineering & Validation Roadmap](docs/engineering-validation-roadmap.md)
- [Machine-readable validation gates](configs/validation-gates.yaml)
- [Experiment manifest template](configs/experiment-manifest.example.yaml)
- [Research issues](https://github.com/ohayotaro/torch-wgi/issues)

## Development Phases

1. **Phase 1 — Numerical correctness:** reference joint projector, geometry gradients, adjoint consistency, quadrature convergence, SDF/soft-binning continuation.
2. **Phase 2 — Physics cross-validation:** Geant4/GATE comparison of absolute sensitivity, SRF, spatial/energy response, and geometry derivatives.
3. **Phase 3 — End-to-end design optimization:** nuisance-adjusted Poisson Fisher information, constrained geometry optimization, independent Monte Carlo and list-mode MLEM validation.
4. **Phase 4 — Scale, ablation, and dissemination:** full-scale streaming implementation, GPU profiling, ablation studies, reproducible paper/OSS package.

The optimization approximation and the high-accuracy validation reference must remain separate. Every accepted experiment should record the code SHA, configuration hash, data hash, environment, random seeds/quadrature, uncertainty estimate, and reproduction command, with outcomes reported as **PASS**, **FAIL**, **INCONCLUSIVE**, or **NOT RUN**.

## Planned Package Structure

The intended package layout is documented in the roadmap. No installable Python package or completed benchmark is claimed at the current repository state.

OSS licensing, data redistribution rights, archival DOI, and author metadata remain release-time decisions.
