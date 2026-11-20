# Where Remote CME Forecasting Stops: An Empirical Information-Ceiling Study Across Solar Cycles 23–25

**Author:** Zar-Manah
*Independent Researcher*

**Status:** LOCAL DRAFT — not submitted, not published, not deposited.

---

## Abstract

Operational CME arrival-time forecasting has been stuck near a mean absolute error
(MAE) of ten hours for a decade, across more than thirty independent methods that
include three-dimensional MHD simulations running on supercomputers. This paper does
not claim to break that wall. It measures where the wall is, and separates three
effects that are usually conflated: the model, the cohort, and the information
available at launch.

Using a frozen catalogue of 43,559 CDAW CMEs, 610 matched CME–ICME associations and
a held-out Solar Cycle 25 cohort, we evaluate two frozen transit models on two frozen
cohorts, reporting all four cells rather than the most favourable one. The central
result is that the cohort effect dominates the model effect by a factor of five:
on the full Cycle 25 cohort (n = 126) the served model and the benchmark model are
statistically indistinguishable (15.4713 h vs 15.4275 h; paired ΔMAE = +0.0438 h,
95 % CI [−0.8367, +0.9117], p = 0.94), while on the frozen one-to-one association
subset (n = 91) both models improve by four to five hours and the benchmark model
gains a further 1.05 h (10.5336 h vs 11.5792 h; ΔMAE = +1.0456 h, 95 % CI
[+0.0967, +1.9837], p = 0.027). The widely quoted 10.5 h and 15.5 h figures are
therefore not evidence about model quality; they are evidence about cohort
difficulty.

We further report two information ceilings that are demonstrated rather than
assumed. A static 90 % conformal interval cannot be narrower than 69.3126 h
(symmetric) or 64.6250 h (asymmetric) on this cohort at 88 % coverage, against a
design target of 35 h. The oracle-recalibrated Brier skill score for
Dst ≤ −100 nT at launch is 0.1500, against a design target of 0.25. Both targets are
unreachable with white-light coronagraphy and no interplanetary magnetic field
measurement, and we document them as such instead of tuning against them.

Finally, we place these numbers in their correct operational context. The best
public CCMC scoreboard entry in 2023–2024 (NOAA/SWPC ENLIL at 9.72 h) is obtained on
a self-selected sample of 47 CMEs and involves human adjustment of the cone
parameters before submission. HELIOS is fully automatic, has no human in the loop,
runs on a single consumer CPU core, and lands mid-table (11.5792 h
on the clean subset, between BoM at 11.45 h and EAM at 12.04 h). We argue that the
honest contribution is a reproducible map of the observational limits, not another
claim of superiority.

**Keywords:** space weather, coronal mass ejection, arrival time, conformal
prediction, information ceiling, reproducibility, solar cycle 25.

---

## 1. Introduction

Two numbers dominate the CME arrival-time literature: approximately ten hours of
mean absolute error, and approximately twenty hours of standard deviation. Riley
et al. (2018) analysed 724 forecasts from 32 models and found no improvement over
six years. Kay et al. (2024) re-examined the same scoreboard with corrected error
measures and reported an uncertainty of order 10–15 h. A 2025 study of ambient-wind
uncertainty (Raza et al. 2025) starts from the same premise: despite increasingly
complex models, the mean absolute error of CME arrival predictions still exceeds ten
hours.

That plateau is usually explained with a physics argument: a single white-light
viewpoint cannot constrain the along-sight component of the CME trajectory, the
ambient solar wind along the path is unmeasured, and the internal magnetic field is
invisible until the ejecta reach L1. Those are real limitations. They are also
routinely used to justify claims that a particular model has "reached the state of
the art", which is a different and much weaker statement than it appears.

This paper takes the opposite approach. Rather than asking whether a model can be
made to beat a threshold, we ask what the data can support, and we report the
answer in a form that can be checked line by line. The contributions are:

1. A four-cell evaluation design that reports every combination of two frozen
   models and two frozen cohorts, so that no gate can be certified on a favourable
   cell while the unfavourable one stays hidden.
2. A decomposition of the observed error into a cohort component and a model
   component, with paired bootstrap confidence intervals.
3. Two information ceilings — conformal width and Brier skill — derived from the
   data and reported as ceilings rather than as failures.
4. An explicit operational comparison that separates automatic inference from
   human-assisted operational forecasting.

We do not claim state-of-the-art accuracy. We claim that the limits are measurable,
reproducible, and smaller in number than the claims usually built on top of them.

---

## 2. Data

### 2.1 CME catalogue and associations

The CME catalogue is the CDAW SOHO/LASCO list, 43,559 events. CME–ICME associations
come from the Richardson & Cane list (325 one-to-one entries) plus 285 inferred
matches, for 610 matched associations. The Solar Cycle 25 evaluation cohort contains
126 positive events with a launch time, a linear speed, an angular width and a
transit time in the range 12–150 h.

A second, stricter cohort is used for the comparison with the earlier transit
benchmark: the frozen one-to-one association set, n = 91. This is the cohort on
which the v2.5.19 benchmark and the WSA-ENLIL comparison were originally defined.
It is a subset of the 126 and it is systematically easier, as Section 4 shows.

The distinction matters. The 126-event cohort includes the ambiguous associations;
the 91-event cohort does not. Reporting only the second one, as the earlier
benchmark did, inflates the apparent quality of any model by four to five hours.

### 2.2 Ambient solar wind and source magnetism

Ambient conditions come from OMNI2 at one-hour resolution: flow speed, density,
dynamic pressure, Dst, and the prior 24–36 h preconditioning history. Source
region magnetism comes from JSOC SHARP invariants (USFLUX, TOTPOT, R_VALUE,
MEANSHAR, AREA_ACR), with 3,758 SHARP-matched CME records and 38.2 % coverage of
the post-2010 AR-bearing CMEs.

No synthetic magnetic proxy is used anywhere in the v3 pipeline. The legacy v2
proxies (`shear_angle_proxy`, `free_energy_proxy`, `helicity_proxy`,
`flare_flux_proxy`) are explicitly banned by an automated test.

### 2.3 Labels and causality

Every merged feature carries a timestamp at or before the forecast time; the audit
reports zero violations across 41,756 merged rows. Transit labels are the observed
Earth arrival times. No label is generated, interpolated or imputed.

---

## 3. Models

Both transit models are frozen artefacts. Neither was retrained for this paper.

**Served model** (`models/v3/transit_v3.joblib`). A heterogeneous ensemble of
LightGBM, XGBoost, ExtraTrees and ridge members. Three blends (equal weights,
inverse out-of-fold MAE and non-negative least squares) and two target spaces were
compared by pre-2020 out-of-fold MAE; the winner is inverse MAE in log space
(weights 0.251, 0.248, 0.255, 0.246). Features include
launch speed, width, the 20 R⊙ speed, halo flags, drag-based transit priors over a
grid of (w, γ) (Vršnak et al. 2013; Kay et al. 2020; Napoletano et al. 2022),
solar-cycle phase, and a preconditioning group selected by a paired
one-standard-error rule. Trained on 484 pre-2020 events.

**Benchmark model** (`models/v3/transit_v3_bench.joblib`). The same architecture
restricted to the frozen one-to-one association set, with the ambient group selected
instead of preconditioning. Its blend is non-negative least squares (XGBoost 0.276,
ExtraTrees 0.492, ridge 0.232, LightGBM 0.000). Trained on 235 events. This artefact
is **not served**; it exists only because the earlier benchmark was defined on that
cohort.

The served model produces conformal intervals with CQR and per-regime quantiles; the
benchmark model uses absolute conformal residuals. The absolute and normalised variants
were evaluated out-of-fold and the band mode was selected before 2020.

---

## 4. Results

### 4.1 The four cells

All four combinations are reported. The fourth cell — benchmark model on the full
cohort — had never been measured before this study.

| Cohort | Model | n | MAE (h) | Median AE (h) | RMSE (h) | Bias (h) | Coverage@90 | Mean width (h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_n126 | served | 126 | 15.4713 | 12.2499 | 20.7469 | −1.4744 | 0.9206 | 74.0419 |
| full_n126 | bench | 126 | 15.4275 | 11.0394 | 20.8825 | −1.0402 | 0.7857 | 49.2887 |
| frozen_1to1_n91 | served | 91 | 11.5792 | 8.8489 | 16.1284 | −1.4598 | 0.9560 | 71.4393 |
| frozen_1to1_n91 | bench | 91 | 10.5336 | 6.5440 | 14.7272 | −1.4055 | 0.8791 | 49.1864 |

### 4.2 Cohort effect versus model effect

Paired bootstrap, 4,000 resamples, same events for both models:

| Cohort | ΔMAE (served − bench) | 95 % CI | p |
|---|---:|---|---:|
| full_n126 | +0.0438 h | [−0.8367, +0.9117] | 0.94 |
| frozen_1to1_n91 | +1.0456 h | [+0.0967, +1.9837] | 0.027 |

On the full cohort the two models are indistinguishable. On the clean subset the
benchmark model is better by about one hour, with a confidence interval that
excludes zero but barely. The change of cohort, by contrast, moves both models by
four to five hours.

The interpretation is unambiguous: the 10.5336 h and 15.4713 h figures that have
been quoted as evidence of model quality are mostly evidence of cohort difficulty.
Any comparison that reports one without the other is uninformative.

A second result is visible in the coverage column. The benchmark model produces a
narrower interval (49.29 h) but its coverage collapses from 0.8791 on its own cohort
to 0.7857 on the full cohort. The narrow band is not valid out of cohort; it is
overfitted to the one-to-one association set.

### 4.3 The conformal-width ceiling

The design target was a 90 % interval narrower than 35 h at 88 % coverage.

The measured interval is 74.0419 h at 92.06 % coverage. The tightest static band
that can achieve 88 % coverage on this cohort is 69.3126 h if constrained to be
symmetric, and 64.6250 h if allowed to be asymmetric. Neither is close to 35 h.

The reason is arithmetic, not architectural. With a residual dispersion of
σ ≈ 15–21 h, an 88 % interval requires roughly 3.1 σ of width, which is 47–65 h
before any model error is added. A 35 h interval at 88 % coverage would require
σ ≈ 11 h, which this cohort does not have.

We therefore report the gate as information-infeasible and keep the number in the
record. It is not deleted, and it is not redefined after the fact.

### 4.4 The Apex BSS ceiling

The design target was a Brier skill score above 0.25 for Dst ≤ −100 nT at launch.

The measured BSS is 0.0768 (ROC-AUC 0.6980, 22 positive events out of 126). The
oracle-recalibrated BSS — the best score obtainable by recalibrating the existing
probability on the test labels themselves, which is not a legitimate forecast — is
0.1500. The target of 0.25 is above the oracle. No recalibration, no threshold
change and no additional feature from the same white-light observables can reach it.

The physical reason is that the sign and magnitude of the southward field at Earth
are not encoded in white-light imagery. The ceiling is a property of the
observables, not of the learner. We document it and stop.

### 4.5 The tactical phase

Tactical Dst forecasting at L1 is a different problem, and it is where the skill
actually lives:

| Horizon | RMSE (nT) | Reference |
|---|---:|---|
| +1 h | 3.5014 | persistence 4.6172, raw ODE 4.0099 |
| +6 h | 10.6057 | persistence 13.2073, raw ODE 11.9636 |
| +12 h | 15.1372 | — |

Adaptive conformal inference (Gibbs & Candès 2021) holds marginal coverage near 0.90
at every horizon (0.8999 at +1 h, 0.8996 at +6 h, 0.8986 at +12 h), but conditional
coverage on storm hours degrades with lead time: 0.8983 at +1 h, 0.8004 at +6 h and
0.6625 at +12 h. This is consistent with the independent finding (Collado-Villaverde
et al. 2025) that the reliable horizon for L1-driven Dst forecasting is about three
hours, beyond which skill for the main phase degrades sharply and only the recovery
phase remains predictable.

The strategic/tactical split is therefore not a design choice. It is what the data
supports.

---

## 5. Operational context, stated correctly

The CCMC Earth CME Scoreboard for 2023–2024, recomputed for this study directly
from the public API (164 closed CMEs; earliest pre-arrival submission per method),
gives the following mean absolute errors:

| Method | MAE (h) | CMEs |
|---|---:|---:|
| NOAA/SWPC ENLIL | 9.72 | 47 |
| Ensemble ENLIL (M2M) | 10.18 | 51 |
| Cone + HAF | 10.20 | 42 |
| Met Office ENLIL | 10.59 | 91 |
| M2M ENLIL | 10.74 | 153 |
| ELEvo | 11.28 | 26 |
| CMEFM | 11.37 | 74 |
| BoM | 11.45 | 40 |
| EAM | 12.04 | 58 |
| SARM | 12.05 | 71 |
| SIDC | 15.23 | 80 |

Other recomputations of the same scoreboard differ by a few tenths of an hour
depending on the de-duplication rule; the ranking is stable. The numbers above are a
static snapshot of the public API taken on 2026-11-18; the fetching script and its raw
dump are not part of this package.

Two things must be said about this table, and they point in opposite directions.

First, HELIOS at 11.5792 h on the clean subset sits between BoM and EAM, and at
15.4713 h on the full cohort it sits near SIDC. It is mid-table. Any claim that it
is at the top of this table is false.

Second, the ranking is not a like-for-like comparison. The best entry, 9.72 h, is
computed on 47 CMEs that NOAA chose to forecast, with human adjustment of the cone
parameters before submission. The entry with the largest sample, 10.74 h, covers
153 CMEs and is still a human-supervised operational product. HELIOS has no human
in the loop, selects its own features by pre-2020 out-of-fold error, and runs on
one CPU core.

The correct claim is therefore: HELIOS is competitive with the middle of the
operational field while being fully automatic, and it is not competitive with the
best human-assisted entries. That is the claim the abstract makes.

---

## 6. What this paper does not claim

- It does not claim state-of-the-art arrival-time accuracy.
- It does not claim that the benchmark model is served. It is not.
- It does not claim that the southward field can be predicted from white light.
- It does not claim that the 35 h interval or the 0.25 BSS target are achievable.
- It does not use synthetic data, synthetic proxies, or imputed labels.
- It does not tune any threshold against the test cohort.

---

## 7. Reproducibility

All artefacts are frozen. The four-cell table is generated by:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/v3_15_limits_table.py
```

which writes `results/v3_limits_table.json` and `paper/limits_study/RESULTS_TABLE.md`.
The gate audit is regenerated by `scripts/v3_10_verify_all.py`: 16 of 20 gates pass,
and the four design targets discussed in this paper (G2.4, G2.5, G2.7, G2.8) are
recorded as expected failures rather than omitted. The test suite is run by:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p "test_helios_v3.py"
```

Result: 24 tests, `OK (expected failures=2)`. The two expected failures are exactly
the two information-infeasible gates, encoded as tests so that they cannot be
silently "fixed" by loosening a threshold.

---

## 8. Limitations

1. The one-to-one association subset is systematically easier than the full cohort.
   This is measured, not assumed, and it is the main reason published MAEs are not
   comparable across studies.
2. The 126-event Cycle 25 cohort is small. Confidence intervals on MAE are wide;
   the paired design reduces variance but does not eliminate it.
3. Ambient solar wind is taken from OMNI2 observations along the Sun–Earth line,
   not from a forecast of the wind the CME actually traverses.
4. The CDAW mass and speed are single-point catalogue values with unknown
   uncertainty; no independent measurement of the launch speed was available for
   the full cohort.
5. The NASA DONKI API was unreachable during this work (repeated timeouts, zero
   records) and is not a dependency of any result reported here.

---

## 9. Conclusion

The ten-hour wall in CME arrival-time forecasting is real, and it is made of three
different things that are usually reported as one. The model contributes about one
hour, and only on clean events. The cohort contributes about five hours. The
observables contribute the rest, and they set hard ceilings on interval width and
on storm probability skill.

Reporting all four cells, both ceilings and the full operational table is less
exciting than claiming a breakthrough, but it is the only version of this work that
survives a reader who checks the numbers. That is the contribution.

---

## Data and code availability

CME catalogue: CDAW SOHO/LASCO. Solar wind: NASA OMNI2. Source magnetism: JSOC
SHARP. Independent ICME witness: Wind ICME list. Code and frozen artefacts are
distributed with this package; no personal data, credentials, absolute paths or
submission tokens are included in this package.

## References

- Collado-Villaverde, A., Cid, C., & Muñoz-Martínez, P. (2025). The practical horizon
  of L1-based Dst forecasting. *ESWW 2025*, contribution CD5.
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution
  shift. *Advances in Neural Information Processing Systems*, 34.
- Kay, C., Mays, M. L., & Verbeke, C. (2020). Identifying critical input parameters
  for improving drag-based CME arrival time predictions. *Space Weather*, 18,
  e2019SW002382.
- Kay, C., Palmerio, E., Riley, P., Mays, M. L., Nieves-Chinchilla, T., Romano, M.,
  et al. (2024). Updating measures of CME arrival time errors. *Space Weather*, 22,
  e2024SW003951.
- Napoletano, G., Foldes, R., Camporeale, E., de Gasperis, G., Giovannelli, L.,
  Paouris, E., et al. (2022). Parameter distributions for the drag-based modeling of
  CME propagation. *Space Weather*, 20, e2021SW002925.
- Raza, S. A. Z., et al. (2025). The effect of uncertainties in reproducing the
  ambient solar wind at Earth on forecasting CME arrival times. *Space Weather*, 23,
  e2025SW004371.
- Riley, P., Mays, M. L., Andries, J., Amerstorfer, T., Biesecker, D., Delouille, V.,
  et al. (2018). Forecasting the arrival time of coronal mass ejections: Analysis of
  the CCMC CME scoreboard. *Space Weather*, 16, 1245–1260.
- Vršnak, B., Žic, T., Vrbanec, D., Temmer, M., Rollett, T., Möstl, C., et al.
  (2013). Propagation of interplanetary coronal mass ejections: The drag-based model.
  *Solar Physics*, 285, 295–315.
