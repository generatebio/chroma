# Chain-COM translational scaling (ξ) in `BackboneMVNGlobular`

Notes on a units slip in how the globular covariance sets ξ, plus an opt-in fix
and tests. Paper: Ingraham et al., *Illuminating protein space with a
programmable generative model* (Nature 2023), Supplementary Appendix D.3.

## Background

The globular prior factors its covariance square root as (SI D.3.5)

```
R = a · R_center · R_sum · R_init,     R_center = I − (ξ/N)·11ᵀ
```

`R_center` leaves every internal (mean-zero) mode alone and multiplies each
chain's center of mass (COM) by `1 − ξ`. ξ is supposed to make each chain's COM
std (per xyz dimension) equal a target (SI D.3.4):

- **monomer scaling** (`complex_scaling=False`, ChromaBackbone v0): target = 1
- **complex scaling** (`complex_scaling=True`, ChromaBackbone v1, ChromaDesign Multi):
  target = σ_COM, shared by all M chains of a complex (N_c atoms each, N_tot total):

  ```
  Rg²_COM = max( r²·N_tot^(2ν) − (r²/N_tot)·Σ_c N_c^(2ν+1), 1 )
  σ_COM   = √(M / max(M−1, 1)) · √Rg²_COM
  ```

The SI never writes ξ down. It lives only in
`chroma/layers/structure/mvn.py`, in `BackboneMVNGlobular._globular_parameters`
(where `mean_shift_scale` is ξ).

## The finding

The original code sets

```
ξ = 1 − target / x_init_std,       x_init_std = 1/√(1−b²)
```

which treats `x_init_std` as the chain's uncentered COM std. It isn't:

- `x_init_std` is unitless, a per-atom std in units of `a` (a = 1.5587 Å is
  applied later, in `_multiply_R`), while `target` is in Å. So the formula is
  dimensionally inconsistent.
- The uncentered COM std is `a · x_init_std · √(S_N/N²)`, with
  `S_N = Σ_ij b^|i−j| = N(1+b)/(1−b) − 2b(1−bᴺ)/(1−b)²`. This follows from the
  stationary covariance `Cov(x_i, x_j) = a²·b^|i−j|/(1−b²)`.

So the realized COM std is `target · a·√(S_N/N²)`, which falls short:

| residues per chain | 50 | 100 | 200 | 400 | 1000 |
|---|---|---|---|---|---|
| realized / target COM std | 0.877 | 0.811 | 0.749 | 0.692 | 0.624 |

The consistent formula is

```
ξ = 1 − target / (a · x_init_std · √(S_N/N²))
```

Consequences: multi-chain priors are 20–40% too tight in inter-chain placement,
and whole-complex Rg comes out about 5–20% below `r·N^ν`, worse for larger
assemblies. For single chains only the (harmless) COM spread changes. ξ is
usually negative in complex mode, so `R_center` actually *inflates* the COM.
The SI's "mean-deflation" wording only fits the monomer case.

Measured whole-complex Rg / `r·N_tot^ν` under the released ξ: 0.975 for
`[50,50]`, 0.949 for `[100,100]`, 0.894 for `[150,150,150]`.

This does not fix whole-complex Rg exactly. Per-chain Rg from the D.3.1/D.3.2
formula is itself about 3–4% above `r·N^ν`, so with the fix complexes land about
0–2.5% above the law. That is a separate, smaller approximation, and one the SI
flags itself: D.3.1's last step is an `N ≫ 1` limit. It does not actually become
tighter with N the way the SI suggests, because `b(N) → 1` along the scaling law,
so the error plateaus rather than vanishing.

## Why this survived review

ξ appears nowhere in the paper or the SI — D.3.3/D.3.4 describe in words what it
should do and never write the formula — so there is nothing published to check
the code against. The monomer path is a gauge choice in which the released
formula does do the one thing that matters: it exactly cancels the `(1−b²)^(−1/2)`
divergence that D.3.3 exists to remove. Only the complex path makes a
quantitative claim. And the released weights were *trained* under this same Σ, so
everything is self-consistent — `R·R⁻¹ = I`, the log-determinant matches `R`, the
ELBO is a valid ELBO. The rest of `test_mvn.py` checks exactly those
self-consistency properties and passes either way; before these tests, nothing in
the repo compared a realized Rg against `r·N^ν`.

## What changed

- `mvn.py`: new constructor flag `exact_translation_scaling=False`. The default
  reproduces the original numbers exactly, so released weights are unaffected.
  `True` uses the consistent ξ, computed in float64 because the `(1−b)⁻²` terms
  cancel near the `b` clip, and with `S_N/N² := 1` for padded chains (N = 1).
  Nothing downstream of ξ (R, Rᵀ, R⁻¹, log det) needed changes. Only the value of
  `log(1 − ξ)` changes, which shifts the ELBO by a θ-independent constant.
- `tests/layers/structure/test_mvn.py`: two new parametrized tests.

The flag is reachable only by constructing `BackboneMVNGlobular` or
`ConditionalBackboneMVNGlobular` directly. `DiffusionChainCov` builds its MVN
with three explicit kwargs and never forwards its own `**kwargs`;
`SubstructureConditioner` rebuilds the conditional MVN by copying only
`covariance_model` and `complex_scaling` off the base distribution (it drops
`sigma_translation` the same way); and none of `graph_backbone.py`,
`graph_design.py`, `graph_classifier.py` or `procap.py` expose a corresponding
`noise_*` argument. Turning the flag on changes the forward diffusion, so a model
would need retraining with it — which would need that plumbing first. It is not a
drop-in change for existing checkpoints.

Note also that `log_determinant` does not branch on `covariance_model`: it uses
the globular ξ even for `brownian`, where `_multiply_R` never touches ξ. With the
flag off this is unchanged from upstream, but it means the flag would perturb
`log_prob` for a brownian model without changing its `R`.

## Running the tests

Needs torch plus chroma's `requirements.txt` (importing `chroma` loads the whole
package). From the repo root, with no install needed (the source tree is
imported):

```bash
python -m pytest tests/layers/structure/test_mvn.py -v -k translation_scaling
```

It runs on CPU in about 30 s. The tests build the exact covariance `Σ = R Rᵀ` by
pushing basis vectors through `mvn.sample(C, Z=...)`, so there is no sampling
noise, and compute each chain's COM std. Five cases: `[50]` monomer, then
`[100]`, `[50,50]`, `[30,70]`, `[100,100]` with complex scaling.

- `test_exact_translation_scaling_hits_target_com_std`: with the flag on,
  COM std / target = 1 (rtol 1e-3).
- `test_default_translation_scaling_misses_target`: with the flag off, COM std /
  target equals `a·√(S_N/N²)` (rtol 1e-3) and is below 0.95.

All 10 pass. Measured:

| case | default ratio | exact ratio |
|---|---|---|
| [50] | 0.8766 | 1.0 |
| [100] | 0.8109 | 1.0 |
| [50,50] | 0.8766, 0.8766 | 1.0, 1.0 |
| [30,70] | 0.9267, 0.8443 | 1.0, 1.0 |
| [100,100] | 0.8109, 0.8109 | 1.0, 1.0 |

For a regression check, the rest of `tests/layers/structure/test_mvn.py`
exercises the default path. Its `XCS["real"]` cases need
`tests/resources/6wgl.cif`, which is not in the repo because `.gitignore` has a
blanket `*.cif` rule; that is a pre-existing upstream packaging bug (see upstream
PR #66), unrelated to this change, and those cases fail with or without it.

If a test fails:

- A ratio off by a constant factor across cases probably means a units or
  convention mismatch in the test's recomputed target (`_target_com_std`).
- NaNs point to the padded-chain / `b` clip path in `_globular_parameters`. Note
  that a 1-residue chain yields NaN in `B` upstream, independently of this
  change: the discriminant `N^(2(ν−1))·(N²+9) − (a/r)²` is negative for
  N_atoms = 4.
- A dtype error from `conv1d` means `Z` was not float32. The module's masks are
  float32, so `Z` must be float32 too.
