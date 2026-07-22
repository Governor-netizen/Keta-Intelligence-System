# 3. Results

## 3.1 Catalogue verification and data integrity

Cross-referencing the 23-event catalogue against contemporaneous news reports and
NADMO/IOM assessments led to the correction of five event windows prior to model
training. The most consequential correction concerned the November 2021 disaster,
Ghana's most severe recent coastal flood (~4,000 displaced): the original window
(10–20 November) began three days *after* the surge, which struck at dawn on
7 November 2021. Similar corrections applied to the April 2022 Agavedzi–Salakope
event (previously misdated to May 2022), the January and February 2025 events
(onsets 16 January and 1 February, respectively), and one event for which no
documentary evidence could be found in the assigned year (May 2024) and which was
re-assigned to the documented event of 26 May 2025. Regenerating the FES2022 tidal
boundary conditions over the corrected windows changed the tidal maximum, range,
or spring/neap classification for 10 of 23 events. The corrections produced a
physically coherent catalogue: every news-documented tidal-wave event coincides
with spring-tide conditions (maximum geocentric tide 0.81–0.91 m), whereas all
rainfall- and dam-spillage events fall on neap tides (0.55–0.74 m).

A per-event data-coverage audit across all six satellite and reanalysis sources
identified two further defects that would otherwise have entered training
silently: (i) the soil-moisture source used in earlier pipeline versions
(NASA-USDA SMAP, discontinued August 2022) had supplied a constant fallback value
for the 11 most recent events, and was replaced with SMAP L4 (SPL4SMGP);
(ii) two validation events (February 2024, March 2025) contained no Sentinel-1
acquisition within their windows, which would have produced spurious all-negative
flood labels; both windows were extended by 5 days to capture the next orbital
pass. After correction, all 23 events have complete coverage in all six sources
(Sentinel-1, GPM IMERG, CHIRPS, ERA5, SMAP L4, FES2022).

The corrected driver labels agree closely with the documented character of the
events: the three 2024–2025 tidal surges yield 100% coastal-labelled flood
pixels, the November 2021 surge 69%, while the September 2023 event dominated by
rainfall and the onset of the Akosombo dam spillage yields 99% rain-labelled
pixels.

## 3.2 Flood susceptibility (Model A)

Model A was trained on ten events (2019–2022) and evaluated on four
strictly out-of-time test events (2023) and nine validation events (2024–2026)
(Table 1). Performance was stable across both held-out periods (test: 79.8%
overall accuracy, κ = 0.389; validation: 80.9%, κ = 0.383), with no
degradation from test to validation despite the five-year span, indicating
that the model generalises across years rather than memorising event-specific
conditions. Flood-class recall was 0.66 (test) and 0.53 (validation) at
precisions of 0.41 and 0.47, respectively; the no-flood class exceeded an
F1 of 0.87 in both periods. Feature importance was dominated by distance to
permanent water (gain share 0.83), followed by tidal range, the 3-m elevation
mask, and slope, consistent with flood water propagating from the lagoon and
shoreline into the lowest-lying terrain (Figure X: xgb42_model_a_importance.png).

**Table 1.** Model A flood-susceptibility performance on out-of-time splits.

| Split | Events | n | Accuracy | κ | Flood recall | Flood precision |
|---|---|---|---|---|---|---|
| Test (2023) | 4 | 2,588 | 79.8% | 0.389 | 0.662 | 0.412 |
| Validation (2024–2026) | 9 | 5,823 | 80.9% | 0.383 | 0.535 | 0.470 |

## 3.3 Pixel-scale driver attribution under leakage control

Because the driver labels are constructed from rainfall and coastal indices
(RFI/CII) that share inputs with the feature set, Model B was evaluated at four
nested feature tiers of increasing independence from the labelling rule
(Table 2). The standard tier, intended for operational use, achieved κ = 0.622
(test) and 0.539 (validation); its importance ranking confirms substantial
reliance on rule inputs (freeboard 0.27, tidal range 0.08, 24-h rainfall 0.08),
which motivates the stricter tiers. Removing the two direct rule variables
(no-leak tier) reduced validation performance to κ = 0.301, and removing all six
rule inputs (strict tier) to κ = 0.291. The physical tier — restricted to eleven
independently measured meteorological quantities with no tide-table information —
retained κ = 0.275 (validation) and 0.232 (test). All leak-free tiers therefore
remain well above chance, in contrast to the pre-correction pipeline, in which
the strict tier collapsed to κ = 0.04. We attribute this recovery primarily to
the restoration of genuine soil-moisture variability and the corrected tidal
boundary conditions (Section 3.1).

**Table 2.** Model B pixel-scale driver attribution (rain- vs coastal-dominant)
across leakage tiers.

| Tier | Features | Test acc / κ | Validation acc / κ |
|---|---|---|---|
| Standard | 21 | 84.1% / 0.622 | 77.7% / 0.539 |
| No-leak | 19 | 67.7% / 0.165 | 71.3% / 0.301 |
| Strict | 15 | 70.1% / 0.094 | 71.1% / 0.291 |
| Physical-only | 11 | 75.5% / 0.232 | 70.3% / 0.275 |
| *Strict, pre-correction (v4.0)* | *11* | *52.6% / 0.04* | — |

## 3.4 Event-scale attribution and mechanism ablation

Aggregating flooded-pixel features to event level, a hard driver label was
assigned only where the pixel majority was decisive (≥ 0.60), yielding 14
labelled events (9 rain-dominant, 5 coastal-dominant); eight events were
classed mixed/compound and the October 2023 dam-spillage event was excluded as
fluvial. Under leave-one-out cross-validation (Table 3), the full feature set
achieved 85.7% accuracy (κ = 0.689). A stricter majority threshold (≥ 0.65,
n = 9) left too few events for stable estimation (all tiers degenerate to the
majority class) and is reported only as a robustness boundary.

The mechanism ablation isolates where the leak-free skill originates. A model
restricted to local surge proxies (ERA5 surface pressure and 10-m wind,
including the shore-normal component) shows no skill (57.1%, κ = −0.024),
failing on four of five coastal events. A model restricted to rainfall-related
quantities (IMERG peak intensity and 3-day accumulation, 7-day CHIRPS, solar
radiation, season flag) achieves the best leak-free performance of any
configuration: 92.9% accuracy (95% CI 69–99%), κ = 0.851, misclassifying a
single event. Leak-free driver attribution at Keta therefore operates by
*hydrometeorological elimination* — coastal floods are identified by the
absence of rainfall forcing sufficient to explain the observed inundation —
rather than by direct detection of surge conditions. The absence of a local
surge signal is physically consistent with the documented origin of Keta's
tidal-wave events in long-period swell generated by distant South Atlantic
storms, which is not expressed in local pressure or wind fields. The model is
not simply reading the calendar: the November 2021 surge falls within the minor
wet season (season flag = 1) yet is correctly attributed as coastal from the
measured rainfall amounts. Given the sample size (n = 14), the wide
confidence intervals should be noted; the pixel-scale results of Section 3.3
(n = 2,786) provide the statistically firmer evidence, and the event-scale
analysis is presented as a cross-scale consistency check.

**Table 3.** Event-scale LOOCV attribution on decisive events (n = 14; 9 rain,
5 coastal; Wilson 95% CIs).

| Feature set | Accuracy (95% CI) | κ | Errors |
|---|---|---|---|
| Rain-only (5 features) | 92.9% (69–99%) | 0.851 | 1 |
| Full dynamic (21) | 85.7% (60–96%) | 0.689 | 2 |
| Strict (15) / Physical (11) | 78.6% (52–92%) | 0.512 | 3 |
| Surge-only (6: pressure + wind) | 57.1% (33–79%) | −0.024 | 6 |

## 3.5 Probabilistic attribution and label-noise analysis

The rain-elimination model was used to produce a soft coastal-attribution
probability P(coastal) for all 23 events (Figure Y:
keta_event_attribution_figure.png; LOOCV probabilities for labelled events,
out-of-sample prediction otherwise). The five confirmed coastal events occupy
a distinct band (P = 0.52–0.59) separated by an empty margin (0.28–0.52) from
the confirmed rain events (P = 0.17–0.24), with a single exception
(June 2025 Lawoshime, P = 0.685, discussed below).

Three further observations support the physical validity of the probabilities.
First, the April 2022 event — a news-documented tidal-wave disaster whose
pixel-majority label is nearly ambiguous (55% coastal) — receives P = 0.637,
i.e., the model recovers the documented driver despite the noisy label.
Second, May 2021, an unverified event with a 50:50 pixel split, receives
P = 0.619, constituting a testable prediction that this event was
coastal-driven. Third, the excluded fluvial event (October 2023, Akosombo dam
spillage) receives P = 0.249, correctly rejecting a coastal origin for an
event whose rule-based label (61% coastal) is physically implausible. The
main failure mode is informative: the May 2025 tidal-wave event, whose window
lies in the wet season, receives P = 0.146 — elimination-based attribution
fails when surge and substantial rainfall coincide, which is precisely the
compound-flooding regime and motivates the addition of wave-model forcing
(e.g., significant wave height and swell period) as the natural extension of
this work.
