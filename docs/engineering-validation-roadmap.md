# Engineering & Validation Roadmap — torch-wgi

Version: 0.1-proposed  
Date: 2026-10-06  
Status: **PLANNED / NOT RUN**

Research tracking: [Roadmap #5](https://github.com/ohayotaro/torch-wgi/issues/5) → [Phase 1 #1](https://github.com/ohayotaro/torch-wgi/issues/1) → [Phase 2 #2](https://github.com/ohayotaro/torch-wgi/issues/2) → [Phase 3 #3](https://github.com/ohayotaro/torch-wgi/issues/3) → [Phase 4 #4](https://github.com/ohayotaro/torch-wgi/issues/4).

This document defines the implementation and validation plan. It does not report completed experiments. Numerical thresholds are proposed research engineering acceptance criteria, not external standards or achieved performance. Thresholds may be revised after calibration pilots, but must be frozen before final holdout evaluation. Any revision must record the reason, impact, revalidation scope, and a new holdout when required.

## 0. Non-negotiable modeling contracts

### 0.1 Joint image-space observation model

The target model combines PET and Compton observation likelihoods at the same decay location. The Cone-conditioned Tube Projector is a coordinate reduction and marginalization strategy for evaluating that integral; it is not an algebraic LOR–cone intersection solver.

Let \(f_j\) be the expected number of decays during the acquisition in image coefficient \(j\), and let \(\varphi_j\) be a normalized image basis. When conditional independence is appropriate,

$$
A^J_{ij}(\theta)
=
\beta_3
\int
\varphi_j(x)
P_\theta(y_i^P\mid x)
C_\theta(y_i^C\mid x)
\,dx,
$$

$$
\lambda_i
=
\sum_j A^J_{ij}(\theta)f_j+b_i,
\qquad
s_j
=
\int A_j(y;\theta)\,d\nu(y).
$$

For a finite voxel, \(P\) and \(C\) must be multiplied at the same sub-voxel point before voxel integration. In general,

$$
\mathbb E[PC]\neq \mathbb E[P]\mathbb E[C].
$$

Likewise, \((Pf)(Cf)\) is not a valid replacement because it introduces cross-terms between different source locations.

The decay position \(x\) and annihilation position \(z\) are connected by the positron-range model,

$$
P(y^P\mid x)
=
\int r_\beta(z\mid x)p(y^P\mid z)\,dz.
$$

Common timing gates, readout correlations, unknown Compton order, and unmeasured DOI are marginalized as latent variables when required. Monte Carlo truth is diagnostic only and must not leak into the observable model.

Absolute response must not be normalized away in image space. The model must preserve lost/rejected probability, branching probability, detector shadowing, dead channels, and geometry-dependent Jacobians. Sensitivity is an integral over all possible accepted observations, not a sum over the observed list-mode events.

### 0.2 Fair design comparisons

Primary hardware comparisons hold fixed source decays, acquisition time, manufacturing constraints, and explicitly chosen resource budgets. Equal-accepted-count comparisons may be reported only as secondary resolution studies because they remove sensitivity differences.

PET-only, Compton-only, and triple-event channels must be mutually exclusive when their information is combined. A triple event must not be counted again as an independent PET event.

The source-to-observation operator should remain linear in \(f\). Activity-dependent random coincidences, pile-up, or dead-time effects must be represented as separate nonlinear acceptance/background components.

### 0.3 Separate three error classes

1. **Implementation error:** autodiff vs finite difference, adjoint consistency, normalization bugs.
2. **Numerical/approximation error:** quadrature, cone local linearization, SDF relaxation, SRF approximation, support truncation.
3. **Physics-model discrepancy:** scatter multiplicity, Doppler broadening, attenuation, detector readout, background, hardware mismatch.

Agreement between autodiff and finite differences for the same approximation does not validate the approximation or the physics. Conversely, a statistically significant but practically negligible Monte Carlo discrepancy is not automatically a failure: pass/fail is based on preregistered equivalence margins and confidence intervals.

---

# 1. Phase 1 — Minimal mathematical operator and differentiability

## 1.1 Scope

Use 2D only for sign and binning diagnostics. The primary proof of concept should be a simplified 3D problem with \(32^3\)–\(64^3\) images, smooth Gaussian/ball lesions, a small number of planar detectors or a small ring, and initially simplified attenuation and single-scatter response.

Maintain three independent implementations:

- **ReferenceJointProjector:** CPU float64, high-order sub-voxel quadrature, optimized for independence and correctness rather than speed.
- **TubeProjector:** direct finite-width PET × Compton integration in coordinates \(x=m+tu+B\delta\).
- **ConeConditionedTubeProjector:** local approximation \(g(x)\simeq g_0+b^T\delta\) plus conditional Gaussian marginalization across the transverse tube.

For the conditioned model,

$$
S=\sigma_C^2+b^T\Sigma_Lb,
\qquad
\mu_\delta=-\frac{\Sigma_L b\,g_0}{S},
$$

$$
V_\delta
=
\Sigma_L
-
\frac{\Sigma_Lbb^T\Sigma_L}{S}.
$$

The local reduction is accepted only where its value and gradient agree with the independent reference. High-curvature or broad-support regions must fall back to direct tube quadrature.

Detector geometry is generated from constrained differentiable tensors. Material boundaries use an SDF relaxation,

$$
\rho_\tau(r;\theta)
=
\operatorname{sigmoid}\!\left(-\frac{d(r;\theta)}{\tau}\right),
$$

while detector-volume integrals should use reference-domain mappings and Jacobians when possible.

Readout soft binning should use probability-conserving Gaussian CDF differences,

$$
S_k
=
\Phi\!\left(\frac{b_k-u_\theta}{\sigma_{\rm eff}}\right)
-
\Phi\!\left(\frac{a_k-u_\theta}{\sigma_{\rm eff}}\right),
$$

with \(S_{\rm lost}=1-\sum_k S_k\). Artificial geometry/readout relaxation widths must remain distinct from physical detector SRF widths.

Required operator interface:

- project(f, observations, geometry)
- adjoint(weights, observations, geometry)
- sensitivity(geometry, observation_rule)
- geometry JVP/VJP utilities

The implementation must remain matrix-free except for toy dense references.

## 1.2 Geometry-gradient validation

Nondimensionalize design variables as

$$
\theta_k=\theta_{0k}+s_k\eta_k.
$$

For deterministic CPU float64 fixtures and a normalized scalar objective,

$$
g_{\rm AD}=v^T\nabla_\eta L,
\qquad
g_{\rm FD}(h)
=
\frac{L(\eta+hv)-L(\eta-hv)}{2h}.
$$

Scan \(h=10^{-2},\ldots,10^{-7}\). A valid result requires a stable plateau over at least two adjacent step sizes; do not select a single favorable step.

Baseline test set: 50 geometries × 20 directions, including deterministic edge cases near tangency, crystal edges, gaps, thin crystals, broad cones, and near-zero sensitivity.

## 1.3 Adjoint validation

Use the correct image/observation inner products:

$$
e_{\rm adj}
=
\frac{
|\langle Af,y\rangle_Y-\langle f,A^*y\rangle_X|
}{
\|Af\|_Y\|y\|_Y+
\|f\|_X\|A^*y\|_X+\epsilon
}.
$$

If weighted inner products are used,

$$
A^*=M_X^{-1}A^TM_Y.
$$

A VJP-derived \(A^*\) demonstrates internal discrete consistency but is not sufficient evidence of physical correctness; compare against an independently assembled dense toy reference.

## 1.4 Proposed Phase-1 gates

| Gate | Proposed PASS criterion |
|---|---|
| AD vs central FD | relative error p95 ≤ 1e-4, maximum ≤ 1e-3 for nonzero directions |
| Near-zero gradients | nondimensionalized absolute error ≤ 1e-8 |
| Step-size robustness | stable agreement over at least two adjacent h values |
| gradcheck | passes for all regular reference fixtures |
| Adjoint | CPU float64 residual ≤ 1e-10; selected GPU float32 path ≤ 1e-5 |
| Reduced vs independent 3D reference | count-weighted L1 relative error ≤ 0.5%; stable nonzero directional-gradient error ≤ 2% |
| Quadrature refinement | forward change ≤ 0.5%; geometry-gradient change ≤ 2% |
| Probability bookkeeping | accepted + lost residual ≤ 1e-6; no negative probabilities, NaN, or Inf |
| Relaxation convergence | final two levels: objective change ≤ 1%, nonzero-gradient cosine ≥ 0.99, hard/reference sensitivity and objective gaps ≤ 1% |

## 1.5 Relaxation continuation and debugging

Initial artificial relaxation schedule relative to feature pitch:

$$
\tau/p=
0.25,\ 0.125,\ 0.0625,\ 0.03125,\ 0.015625.
$$

Refine quadrature before decreasing \(\tau\). A local spacing \(\Delta\lesssim\tau/4\) is only a starting heuristic; convergence must be demonstrated.

Do not anneal physical position, energy, or Doppler widths.

Debugging rules:

- If gradients vanish, verify boundary sampling, candidate-support refresh, and sigmoid saturation.
- If gradients diverge, separate unresolved boundaries, non-positive covariance, and genuinely nonsmooth hard designs.
- Use stable CDF-tail arithmetic for bin probabilities.
- Use expm1-based formulas for small optical depths.
- Use log-domain accumulation for small probability products.
- Do not hide impossible geometry with arbitrary epsilons.
- Do not treat gradient clipping as evidence that the gradient is correct.

### Phase-1 Definition of Done

Proceed only when all G1 gates pass, failure regions and fallbacks are documented, deterministic tests are committed, experiment manifests are complete, and an independent rerun reproduces the results.

---

# 2. Phase 2 — Physics cross-validation against Geant4/GATE

## 2.1 Scope and benchmark ladder

Build both Monte Carlo and analytical configurations from a shared geometry/material/readout specification wherever practical.

Validation ladder:

- single slab,
- detector pair,
- simplified dual-ring system,
- fixed literature-inspired WGI geometry,
- target prototype geometry once all required dimensions/material/readout parameters are known.

A literature ring diameter alone is not sufficient to claim prototype replication. Crystal composition/dimensions, axial length, DOI, gaps, dead material, thresholds, energy window, and coincidence window must also be documented.

Physics should be added in stages:

- **V0:** ideal same-decay three-gamma source, vacuum, free-electron single scatter, ideal readout.
- **V1:** finite detector volumes, full attenuation/photoelectric absorption, one Compton scatter then absorption, finite position/energy resolution.
- **V2:** material-dependent Doppler broadening, double scatter, unknown interaction order, escape/atomic relaxation.
- **V3:** patient/object attenuation and scatter, positron range, non-collinearity, isotope branching, observable event selection.
- **V4:** randoms, dead time, pile-up, and rate effects if they are inside the intended claim domain.

Do not represent double scatter solely by widening a single-scatter Gaussian cone.

For Geant4/GATE, record the physics list, active processes/models, model energy ranges, production cuts, atomic relaxation, data libraries, and random seeds. Run physics-cut refinement tests.

## 2.2 Truth/observable separation

Maintain separate schemas.

**Observable table:** detector channel, measured energy/time, actually measurable DOI, readout-quality flags, event class.

**Diagnostic truth table:** emission position, true interaction positions/energies, track/parent IDs, process names, scatter multiplicity, true order.

Never feed diagnostic truth into Fisher evaluation or reconstruction.

Use consistent units (mm, keV, ns), event-building rules, branching conventions, and accepted-events-per-decay rules. Absolute efficiency must use generated decays as the denominator.

## 2.3 Physics metrics and proposed equivalence margins

$$
\epsilon=\frac{N_{\rm accepted}}{N_{\rm decay}},
\qquad
p_b=\frac{\lambda_b}{\sum_{b'}\lambda_{b'}},
\qquad
TV=\frac12\sum_b|p_b^{\rm ana}-p_b^{\rm MC}|.
$$

Evaluate rate and normalized shape separately.

| Metric | Proposed equivalence margin |
|---|---|
| Simple slab transmission/detection | ≤ 1% relative |
| Absolute sensitivity / accepted count | ±5% for primary positions/classes |
| Normalized readout shape | TV ≤ 0.05 |
| Primary count strata | ±10% |
| Energy peak centroid | ±1% of reference energy |
| Energy FWHM | ±5% |
| ARM FWHM | ±5% |
| ARM centroid | within 0.1 × reference ARM FWHM |
| Spatial FWHM | ±5% for ideal SRF, ±10% for full response with identical reconstruction |
| FWTM / 90–95% widths | ±15% |
| Preregistered tail probability | absolute difference ≤ 0.02 |
| Physics-cut or quadrature refinement | primary metric change ≤ 1% |

For non-Gaussian responses, FWHM alone is insufficient; include tail probabilities and quantile widths.

Statistical targets: overall MC sensitivity relative standard error ≤ 0.5% and primary-stratum 95% relative half-width ≤ 2%. A simple unweighted rare-event estimate needs roughly 40,000 accepted events for the former, but correlated histories and importance weights require history-level uncertainty estimation.

A signed discrepancy passes only when its entire 95% confidence interval is inside the preregistered equivalence interval. Underpowered results are **INCONCLUSIVE**, not PASS.

## 2.4 Validate geometry derivatives with Monte Carlo

For at least five geometry neighborhoods and at least ten adequately powered nonzero directions,

$$
g_{\rm MC}(v;\delta)
=
\frac{
Q_{\rm MC}(\theta+\delta v)
-
Q_{\rm MC}(\theta-\delta v)
}{
2\delta
}.
$$

Use \(\delta/2\) or another step-refinement check to diagnose curvature bias. Cover every active design variable with at least one identifiable direction.

Initial proposed G2 derivative gate:

- sign agreement ≥ 90%,
- median magnitude relative error ≤ 15%,
- p90 magnitude relative error ≤ 30%,
- unresolved/underpowered directions do not count as passes.

Define two sub-gates:

- **G2A:** defined simplified physics.
- **G2B:** all accepted response components inside the claimed optimization domain.

An omitted accepted component is allowed only if its upper 95% CI fraction is ≤ 1% **and** adding/removing it changes the task metric by ≤ 2%; otherwise add the physics or an observable rejection rule and revalidate.

## 2.5 Runtime comparison and fallback

Compare equal work products at equal accuracy: expected histograms, sensitivity, or task quantities. Do not compare Geant4 transported-events/s directly with analytical likelihood-evaluations/s.

Report separately initialization, calibration Monte Carlo, preprocessing, I/O, forward, backward, CPU core-hours, GPU-hours, peak host RAM, and peak VRAM.

A 10× speedup is a practical target and 100× is a stretch target; neither is a physics-validation criterion.

When discrepancies are large, debug in this order:

1. units,
2. branching/coincidence definitions,
3. material and attenuation,
4. scatter multiplicity,
5. readout mapping,
6. SRF core/tails,
7. reconstruction.

If only G2A passes, constrain the paper claim to the validated simplified/single-scatter domain.

### Phase-2 Definition of Done

G2B value and geometry-derivative gates must pass on held-out geometries before Phase 3 may claim physically meaningful design optimization.

---

# 3. Phase 3 — Task-driven hardware optimization

## 3.1 Objective

Let

$$
f=f_0+\alpha\psi,
\qquad
\lambda_0=A_\theta f_0+b_\theta,
\qquad
D_\alpha=A_\theta\psi.
$$

The Poisson Fisher matrix is

$$
I_{ab}(\theta)
=
\int
\frac{D_a(y)D_b(y)}{\lambda_0(y)}
\,d\nu(y).
$$

For nuisance parameters \(\nu\), use effective information

$$
I_{\rm eff}
=
I_{\alpha\alpha}
-
I_{\alpha\nu}
(I_{\nu\nu}+\Lambda_{\rm prior})^{-1}
I_{\nu\alpha}.
$$

Use linear solves rather than explicit matrix inversion. Prior information is fixed from external assumptions and must not be tuned as a hidden performance knob.

A typical loss is

$$
\mathcal L(\theta)
=
-
\mathbb E[
\log(I_{\rm eff}/I_{\rm ref})
]
+
\gamma C(\theta),
$$

where the expectation may include lesion location, background variation, and manufacturing tolerances.

## 3.2 One-variable optimization

Start with one physically interpretable variable such as scatterer-to-absorber spacing or crystal thickness.

- Define the feasible interval before optimization.
- Compute a high-accuracy reference sweep with an initial 61-point grid and local refinement.
- Run Adam or another local optimizer from at least eight initial conditions.
- Do not assume the optimum is interior; a boundary optimum or monotonic response may be physically correct.
- Use a 1%-near-optimal set when the objective is flat.

Proposed numerical criterion:

$$
\mathrm{regret}
=
\frac{
I_{\rm ref}^{\max}-I(\hat\theta)
}{
I_{\rm ref}^{\max}
}
\le1\%.
$$

Re-evaluate 9–13 prespecified geometries plus the final candidate with independent Monte Carlo.

## 3.3 Multivariable optimization

Use constrained, nondimensionalized parameters for ring radius, detector spacing, thickness, and pitch. Preserve ring closure constraints such as \(Np=2\pi R\) when required by the parameterization.

Use at least 16 Sobol/LHS starts, relaxation continuation, trust regions, and local Adam/L-BFGS refinement.

Discrete channel-count changes should be handled by outer discrete search as the primary method. If continuous occupancy gates are studied, they must be followed by

$$
\text{round/binarize}
\rightarrow
\text{continuous re-optimization}
\rightarrow
\text{independent MC validation of the hard design}.
$$

Keep optimization, model development, and final evaluation data separate. Common random numbers may be used for variance reduction during search, but the final evaluation must use independent seeds/quadrature.

If MC anchors reveal that value or derivative predictions have left the validated domain, shrink the trust region and update the model before continuing.

## 3.4 Reconstruction and detection-task validation

Generate independent list-mode Monte Carlo data for the initial and final hard geometries. Use the same image grid, corrections, initialization, stopping rule, and post-filtering policy.

List-mode MLEM uses

$$
f_j^{(n+1)}
=
\frac{f_j^{(n)}}{s_j(\theta)}
\sum_i
\frac{
A_{ij}(\theta)
}{
\sum_k A_{ik}(\theta)f_k^{(n)}+b_i(\theta)
}.
$$

Sensitivity must come from the full accepted-observation model, not from summing only observed events.

Evaluate central, peripheral, and axial-edge lesions over prespecified sizes/contrasts and background perturbations.

Secondary image metrics:

- CRC,
- ensemble CNR,
- bias/noise curves.

Primary detection evidence should use a fixed lesion-detection score or a properly separated observer such as a channelized Hotelling observer, with \(d'\) or ROC AUC. CRC/CNR alone are insufficient.

Use an initial pilot of about 100 independent noise realizations per class only to estimate variance and finalize a power/precision-based sample size. The pilot count is not itself a power guarantee.

## 3.5 Proposed Phase-3 gates

1. Analytical Fisher and MC reference, under the same readout definition, agree within a ±10% equivalence interval.
2. One-dimensional optimization regret ≤ 1%; all manufacturing constraints satisfied.
3. Final hard geometry reproduces the predicted benefit in independent Monte Carlo.
4. Engineering target: ≥10% point-estimate gain for Fisher or \(d'\); an improvement claim requires the 95% CI lower bound to be > 0.
5. For AUC, use a preregistered absolute margin, for example an initial design target of +0.03, not a relative 10% rule.
6. No >5% degradation in prespecified critical strata unless a separate, justified noninferiority margin was preregistered.
7. Report all starts, failures, compute budgets, and independent re-evaluations.

If improvement disappears under independent MC, the optimization is not considered validated. Diagnose SRF tails, omitted physics, sensitivity bookkeeping, support truncation, background/nuisance modeling, and relaxation-to-hard gaps.

---

# 4. Phase 4 — Scalability, ablation, paper, and OSS

## 4.1 Scale targets and architecture

Benchmark ladder:

- channels: \(10^4,\ 3\times10^4,\ 10^5\),
- image volumes: \(128^3,\ 256^3\),
- streamed observations: \(10^5,\ 10^6,\ 10^7\).

Never materialize a full event × voxel system matrix or the complete PET × Compton channel Cartesian product.

Implementation order:

1. event chunking,
2. image tiling,
3. local material/candidate lookup,
4. hierarchical observation integration for sensitivity/Fisher,
5. two-pass VJP for Fisher accumulation,
6. mixed precision with high-accuracy reductions,
7. compile/profiling,
8. Triton/CUDA custom kernels only if profiling proves they are necessary.

The same response evaluation should process background, lesion, and nuisance templates together when possible.

A custom operator must provide both the image-space adjoint and geometry VJP, then repeat gradcheck, CPU-reference, and adjoint validation.

## 4.2 Proposed scale gates

Initial hardware target: a 24 GB-class GPU should stream a \(256^3\) volume, \(10^5\) channels, and \(10^7\) observations with peak reserved memory ≤ 20 GB.

Other targets:

- 10× more events with fixed chunking should increase peak device memory by ≤ 10%.
- Runtime scaling with event count should have log–log slope ≤ 1.2 over the valid benchmark range.
- Optimized implementation vs high-accuracy reference: Fisher relative difference ≤ 1%.
- Stable nonzero geometry-gradient relative L2 difference ≤ 2%.
- Geometry-gradient cosine ≥ 0.99.
- GPU adjoint residual ≤ 1e-5.

Report cold/warm timing, I/O, compilation, synchronization, median/p95, host/device memory, dtypes, CPU core-hours, and GPU-hours. Use at least 30 repetitions for microbenchmarks when practical.

## 4.3 Required ablations

| Comparison | Scientific question |
|---|---|
| High-accuracy sub-voxel reference vs direct tube vs conditioned tube | Value/gradient bias and speed from dimensional reduction |
| Hard geometry vs fixed soft relaxation vs annealed relaxation | Relaxation gap and boundary-gradient behavior |
| Single Gaussian vs Doppler mixture vs multi-scatter model | Importance of SRF core/tails |
| Joint vs PET-only vs Compton-only | Information contribution under equal resource constraints |
| Analytical AD vs FD vs BO/CMA-ES | Value of gradients independent of the operator itself |
| MC grid/direct search vs MC-evaluated BO | Search quality per MC budget |
| Analytical-only vs MC-anchored trust region | Robustness to model exploitation |
| Absolute sensitivity vs intentionally normalized negative control | Failure induced by removing efficiency information |

For Monte Carlo direct search, fully cover at least feasible 1D/2D spaces. In higher dimensions, compare methods under the same feasible region, evaluation budget, and independent MC final evaluation.

Report both equal-evaluation-count and equal-wall-clock comparisons, including calibration and preprocessing costs.

## 4.4 Paper strategy

The main claim should not be “we can compute an LOR–cone intersection.” WGI and LOR–cone imaging have prior art.

Candidate novelty claims are:

1. an absolute-count voxelwise joint WGI observation operator,
2. an error-controlled low-dimensional projector with validated geometry derivatives,
3. physics/gradient validation on held-out geometries,
4. task-driven hardware optimization confirmed with independent Monte Carlo and, when available, measurements,
5. scalable and reproducible implementation.

Suggested paper structure:

1. motivation and related work,
2. observation/measure contract,
3. mathematical operator and differentiable geometry,
4. preregistered validation protocol,
5. G1 numerical verification,
6. G2 physics and derivative validation,
7. G3 design/reconstruction/detection results,
8. G4 scaling and ablation,
9. error budget and limitations.

For IEEE Transactions on Medical Imaging, emphasize generalizable computational methodology and task validation. For Medical Physics, emphasize physics-response modeling, uncertainty, and phantom validation. An IEEE NSS/MIC submission may report completed G1/G2 and limited G3 results, but planned results must never be written as achieved.

Without measurements, claims must remain simulation-only. Clinical effectiveness must not be claimed without appropriate experimental and clinical evidence.

## 4.5 OSS and reproducibility package

Planned artifacts:

- Python package and documented API,
- explicit unit/measure contracts,
- small deterministic fixtures,
- Monte Carlo generation scripts,
- evaluation CLI,
- CPU unit CI,
- GPU regression tests,
- MC golden-data integration,
- documented failure regions,
- dependency lock/container digest,
- figure-generation scripts from versioned data/configs,
- experiment manifests containing code/data/config hashes.

Large Monte Carlo data should live in an authorized archive; the repository should store manifests and hashes.

Before public release, review OSS license choice, dependency licenses, data redistribution rights, CITATION metadata, author metadata, and archive DOI. Do not store patient-identifying or unauthorized collaborator data in the public repository.

---

# 5. Planned package layout and first implementation steps

This is a target layout, not a claim that the modules already exist.

~~~text
src/torch_wgi/
  geometry/{parameters,sdf,readout}.py
  physics/{transport,decay,energy,doppler,paths}.py
  operators/{reference_joint,tube,conditioned_tube,adjoint}.py
  reconstruction/listmode_mlem.py
  tasks/{fisher,observers}.py
  validation/{gradients,adjoint,statistics,mc_compare}.py
  io/{events,manifest}.py

configs/{geometry,physics,phantoms,experiments}/
tests/{unit,operator,integration}/
mc/{geometry_export,run_reference,event_extract}.py
benchmarks/{operator,memory,optimization}.py
paper/{figures,tables}/
~~~

The first implementation PR should contain only:

- pyproject.toml,
- explicit unit/shape contracts for event and geometry data structures,
- a smooth toy phantom,
- ReferenceJointProjector,
- geometry-gradient tests,
- adjoint tests,
- probability-bookkeeping tests.

The next PR should add SDF/soft-binning, followed by the conditioned Tube Projector, then the complete G1 validation report. Do not start with custom CUDA or full multiple-scatter physics.

A bootstrap environment may use PyTorch, NumPy, SciPy, pytest, and PyYAML, but publication experiments must pin versions and container/environment hashes. Geant4/GATE should use a separate reproducible environment/container.

Every accepted experiment must instantiate configs/experiment-manifest.example.yaml, reference one or more IDs from configs/validation-gates.yaml, and attach result estimates, uncertainty, thresholds, hashes, and reproduction commands.

---

# 6. Reference material

These references motivate modeling or implementation decisions. They are not sources for the roadmap's proposed numerical acceptance thresholds.

- Yoshida et al. (2020), “Whole gamma imaging: a new concept of PET combined with Compton imaging,” Physics in Medicine & Biology 65, 125013. DOI: 10.1088/1361-6560/ab8e89.
- PyTorch documentation, Gradcheck mechanics.
- Geant4 Physics Reference Manual, Compton scattering.
- OpenGATE documentation, physics configuration.
- Barrett, White, Parra (1997), “List-mode likelihood,” JOSA A. DOI: 10.1364/JOSAA.14.002914.
- PyTorch documentation, torch.utils.checkpoint.
- IEEE Transactions on Medical Imaging official site.
- Medical Physics official journal site.
