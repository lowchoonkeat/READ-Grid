"""Assemble paper_package/13_author_manuscript_draft_v3.md from v2 + new prose.

Why a build script rather than an edited copy
---------------------------------------------
The plan (16_section_plan_v2.md §7) is mostly a *relocation*: merge 5.1+5.2, retitle
the 5.x ladder, compress 5.3, keep 5.4 intact, expand §1/§2, rewrite the abstract.
Hand-copying nine thousand words to move them would risk silently altering a number,
and every number in this paper is load-bearing. So the unchanged blocks are sliced
out of v2 by heading and inserted verbatim, the rewritten blocks are written here,
and each targeted edit is an explicit replace() that fails loudly if its anchor
moves.

Citation markers are written as [@key] and numbered later by
analysis/lit/number_citations.py, so no manual renumbering is possible.

Usage
-----
    python analysis/build_manuscript_v3.py
    python analysis/lit/number_citations.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "paper_package" / "13_author_manuscript_draft_v2.md"
OUT = ROOT / "analysis" / "_build" / "v3_marked.md"


# --------------------------------------------------------------------------- #
# slicing
# --------------------------------------------------------------------------- #

def parse_top(text: str) -> dict[str, str]:
    """Split on level-1 headings only, so `##` subsections stay inside their parent."""
    out: dict[str, list[str]] = {"__front__": []}
    cur = "__front__"
    for line in text.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            cur = line[2:].strip()
            out[cur] = []
            continue
        out[cur].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def get_sub(block: str, heading: str) -> str:
    """Return the body of a `## heading` subsection, heading excluded."""
    lines = block.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip().startswith(heading):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"subsection not found: {heading!r}")
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## ") or lines[j].startswith("# "):
            end = j
            break
    return "\n".join(lines[start:end]).strip("\n")


def sub(block: str, heading: str, *pairs: tuple[str, str]) -> str:
    body = get_sub(block, heading)
    for old, new in pairs:
        if old not in body:
            raise SystemExit(f"anchor not found in {heading!r}: {old[:70]!r}")
        body = body.replace(old, new)
    return body


def top(block: str, name: str, *pairs: tuple[str, str]) -> str:
    if name not in block:
        raise SystemExit(f"block not found: {name!r}")
    body = block[name]
    for old, new in pairs:
        if old not in body:
            raise SystemExit(f"anchor not found in {name!r}: {old[:70]!r}")
        body = body.replace(old, new)
    return body



def drop_column(block: str, index: int) -> str:
    """Remove one column from every markdown table row in `block`."""
    out = []
    for line in block.split("\n"):
        if line.startswith("|"):
            cells = line.split("|")
            if len(cells) > index:
                del cells[index]
            line = "|".join(cells)
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# new prose
# --------------------------------------------------------------------------- #

ABSTRACT = """\
Local demand-response execution stays available when the cloud command path fails, but availability is not a claim about quality: it does not show that a learning policy beats a rule. This paper reports a Python feeder simulation of READ-Grid (Resilient Edge-Native Autonomous Demand Response for Smart Grids), a controller whose cloud policy is guarded by a local command-acceptance boundary, and separates command-path availability from policy quality. A first run — 13,965 feeder-coupled records over 21 episodes and 95 faults — put the tuned local rule at 58.57% target achievement against 45.25% for the learned policy, a gap of −13.32 points. Three pipeline defects could produce that gap: intra-minute heartbeat look-ahead, a Bellman target maximised over inadmissible successor actions, and training with neither full-episode replay nor state coverage. An audited, pre-registered pipeline removes all three. Its held-out primary comparison against a rule enumerated over 144 candidates is **inconclusive**: the mean difference is −0.0040, 95% interval [−0.0239, +0.0174], whose half-width of 0.0207 exceeds the pre-registered margin δ = 0.02, so superiority, non-inferiority and equivalence are unestablished. Correcting the pipeline moves the learned arm by +0.1152 [0.0726, 0.1619], locating the shortfall in the training pipeline, not the action-dependent linear representation. The guarded local path adds 0.147 to 0.220 points of outage-window achievement across all four policy pairs, positive throughout though unequal. The audited learner costs 0.96 ms per control step for the whole feeder, 33 times the cloud-only rule and 0.0016% of the 60 s interval."""

S1 = """\
Demand response at the distribution level depends on two things that fail independently. The first is a decision policy: which load groups to curtail, when, and by how much. The second is a path: a command issued at the supervisory layer has to reach the device that actuates it. A cloud-hosted controller couples the two, because losing the backhaul removes the path and the policy together. Executing at the edge restores the path. It does not, by itself, say anything about the quality of the policy that runs there.

These are separate claims, and the literature often runs them together. Work on edge and cloud-edge demand response reports that local execution keeps the service available when the cloud is unreachable [@zhang2021edge; @tao2022hybrid; @jiang2020cloud], and compact learners have been shown to run on device-class hardware without a cloud connection [@gao2025lightweight]. A communication-failure scenario that hands control to a local mode has been evaluated on a PV–BESS microgrid [@pasculescu2026intelligent]. Availability is the outcome these studies measure, and availability is what they establish. The inference that a local rule is therefore the right choice at the edge does not follow, and neither does the converse inference that a learner is preferable because it degrades more gracefully. Both are claims about policy quality, and neither study design isolates it.

The comparison between a rule and a learned controller carries difficulties of its own. Tuning effort is rarely equalised: a rule with three hand-set thresholds is commonly compared against a learner whose hyperparameters were searched, and the search is itself a source of variance the rule does not carry [@dodge2019show; @musgrave2020metric; @melis2018state; @patterson2024empirical; @lucic2018gans]. The evaluation split is often reused for selection, which turns the reported contrast into an order statistic of the selection set rather than an estimate of its generalisation [@dwork2015reusable; @taylor2015statistical; @andrews2024inference; @lee2016exact; @recht2019imagenet; @ioannidis2005most]. Intervals, when present, are frequently computed over the wrong unit, so a difference of unknown precision is read as a ranking [@henderson2018deep; @agarwal2021deep; @colas2018many; @bouthillier2021accounting]. And the boundary that would make a difference interpretable is usually chosen after the difference is seen [@schuirmann1987comparison; @lakens2018equivalence; @lakens2017equivalence; @altman1995statistics; @piaggio2006reporting]. None of these is specific to demand response, and together they explain why the rule-versus-learning question has produced so many confident and inconsistent answers.

This study addresses the two claims separately. READ-Grid is implemented as an executable Python simulation, and command-path availability is measured as an outcome distinct from policy quality. The comparison was run twice, and the two runs are reported separately and never combined. **Run 1** used the training pipeline as first implemented; auditing that pipeline found three defects, which were fixed before **Run 2**, a pre-registered re-run of the same architecture. Run 1's numbers are not revised and are not used to corroborate Run 2. Three questions were fixed before Run 2 was executed.

- **Q1.** In Run 1 the local learned policy trailed the tuned local rule by 13.32 percentage points. Is that shortfall a property of the **method**, or of the **training pipeline**?
- **Q2.** After the training pipeline is corrected, is the learned policy **behind**, **non-inferior to**, **equivalent to**, or **ahead of** a more thoroughly tuned rule — or is the evidence insufficient?
- **Q3.** How much value does the guarded edge architecture provide **during an outage**, and is that value independent of the policy type?

The study makes four contributions, each tied to evidence in Section 5 or Section 6.

- **C1 — A reproducible implementation and a pre-registered re-evaluation.** Source, configuration, ten corrected checkpoints and a per-artefact hash register, with the scripts that regenerate every interval here.
- **C2 — An attribution rather than a comparison.** The 13.32-point shortfall is accounted for by three named pipeline defects, and removing them moves the learned arm by +0.1152 on the held-out split. This localises the shortfall in the pipeline; it does not establish a limit of the action-dependent linear representation.
- **C3 — A measurement of the architecture's value that holds in sign across policy type.** With the policy held fixed and only the local path added or removed, the guarded architecture contributes 0.147 to 0.220 points of outage-window achievement in all four pairs tested. The sign is robust across pairs; the size is not (Section 5.4).
- **C4 — An honest report at a declared resolution.** Against a rule tuned over the same 144-candidate budget the primary comparison is inconclusive, because the interval is wider than the margin, and a parameter-free heuristic matches the enumerated rule. Both are findings, not reasons to re-run at a looser boundary.

Section 2 narrows the claim against prior work and lists what is not claimed. Section 3 gives the system model and value representation, and Section 4 the evidence base. Section 5 reports the results in the order the argument requires, and Sections 6 to 8 state where they stop holding, what design rule follows, and what is established."""

S2 = """\
Local demand-response execution is established prior work, and its availability is not this paper's novelty. Edge and cloud-edge controllers with learned policies have been demonstrated for building demand response [@zhang2021edge], for hybrid cloud-edge coordination [@tao2022hybrid], and for closed-loop price response of aggregated air conditioners, including one that models packet dropouts [@jiang2020cloud]. Compact learners run on device-class hardware without cloud connectivity [@gao2025lightweight]. What this work establishes is availability under a degraded network; what it does not isolate is whether the local policy is preferable to a simpler local alternative, because the comparison is against a cloud controller that has lost its path.

The rule-versus-learned comparison is also established, mostly on building energy and HVAC control [@nagy2023questions; @arroyo2022reinforced; @chen2019gnurl; @blum2021building; @drgona2020need; @vazquezcanteli2019citylearn; @wei2017deep], with fewer studies at the distribution level [@pei2024visibility; @wang2020safe]. Two patterns there shape the design here: the learned controller is compared with a hand-set rule rather than one tuned under an equal budget, and the result is reported as a ranking of point estimates rather than an interval on a difference.

Local fallback under communication loss is likewise not new. Hierarchical microgrid control with seamless transfer between operating modes [@mohamed2011hierarchical], distributed resilient control of storage in cyber-physical microgrids [@deng2021distributed] and resilience analysis of DC microgrids under denial-of-service threats [@liu2019resilience] all keep a system operating as the command path degrades; networked control under packet loss has been analysed for two decades in the parent field [@heemels2010networked]. The contribution is therefore not that a local path exists, but what it is worth against the same policy with and without it.

Distributed systems supply the vocabulary for the authority question, and only the vocabulary. Failure detectors separate the suspicion that a process has failed from the authority to act on it [@chandra1996unreliable]; time-bounded leases make ownership expire instead of requiring a decision [@gray1989leases]; consensus elects a single leader and fences stale ones [@sakic2018response]; distributed lock managers implement exclusion in deployed systems [@yoon2018distributed]. These motivate explicit assumptions about timing and ownership. They do not establish the correctness of the READ-Grid arbiter, whose checks, epochs, sequences and expiry need their own tests (Section 6.1).

The gap this paper fills is in evaluation design: the difficulties of Section 1 are met by three disciplines, each with a stated cost. Tuning effort is equalised in candidate evaluations — 144 per family under one selection objective — and the residual asymmetry is declared: 144 points densely sample a three-parameter rule but sparsely sample a learner's hyperparameter space, so the arrangement works against the learner [@bergstra2012random]. The boundary is fixed at δ = 0.02 before the run and is not raised afterwards [@schuirmann1987comparison; @lakens2018equivalence]. Intervals resample episodes, the unit that was varied, with the small-cluster undercoverage disclosed in Section 6.4 [@cameron2008bootstrap; @field2007bootstrapping]. The validation split produces no interval, because it was consumed by selection [@dwork2015reusable]. The reporting conventions implied are those proposed for models that inform decisions [@kapoor2024reforms], including checking for information that leaks across a split [@kapoor2023leakage].

The physical layer is a boundary of the claim, not a contribution to it. Feeder calculations run on the DSS-Extensions implementation through the `dss-python` package [@dsspython2024]; the IEEE 123-bus example supplies both automatically controlled and explicitly locked regulator configurations [@opendss123], and the distinction matters because the study configures STATIC control mode, not fixed-tap operation. Load shapes come from a public European low-voltage benchmark directory [@lvprofiles] and are not measured household traces.

Four things are therefore not claimed, and each is revisited in Section 6.5: that the architecture is ready for a real feeder; that the reported voltage outcomes constitute an electrical-safety result; that the arbiter resolves ownership across multiple arbiters; and that the workload represents a particular population of households."""

S5_1_HEAD = """\
## 5.1 First evaluation and the three defects

The archived grid establishes the phenomenon that the rest of the paper explains."""

S5_2_BRIDGE = """\
**Three defects, each corrected and pinned by a test.** The gap in Table 1 is not attributed to the model until the pipeline that produced it has been audited. Three defects were found and corrected:

1. **Intra-interval heartbeat look-ahead** (Section 3.1).
2. **Unmasked Bellman target**: the fitted-Q target maximised over inadmissible successor actions.
3. **Training-pipeline incompleteness**: training used a single pass without full-episode replay, and without state coverage."""

S6 = """\
# 6. Validity, audit and limitations

Each subsection states a conclusion this paper relies on, then the condition under which it stops holding. A list of caveats invites a reader to discount all of them equally; a stated failure condition says which claim is affected.

## 6.1 Temporal causality and authority integration

**Conclusion.** The corrected pipeline cannot read the future: arrivals are processed only once their timestamp has occurred, and an integration test mutates post-decision arrivals and asserts the earlier decision is unchanged. Authority transfer and recovery delays are timed on an actual authority *change*; the first grid carried impossible negative delays for policies with no cloud controller (as low as −59.2 s, 1,974 of 1,995 rows), and the corrected grid contains none.

**Failure condition.** The authority update still runs on the 60 s control grid while the configured takeover threshold is 5 s, so no claim of immediate local actuation at 5 s is made; millisecond command-delay parameters are kept separate from whole-minute decision updates. A deployment that must act within seconds of losing its path is untested here.

## 6.2 Source provenance

**Conclusion.** Run 1's hash chain is complete: its configuration hash matches the first main manifest, and its numbers are anchored by the recorded hashes.

**Failure condition.** The source bytes at that run were never archived. The current copies of `src/readgrid/sim.py` and `src/run_stages.py` do not match their recorded first-run hashes, and fifteen of the 21 recorded source hashes differ from the current files, because two defect fixes dated after the first held-out run both touch `sim.py`. Run 1 is therefore **not byte-reproducible** from the current tree. The recorded manifest is not rewritten and no new hash is back-filled onto the old outputs. The corrected programme freezes and snapshots its source for the duration of the long runs.

## 6.3 Metric semantics and workload generalisation

**Conclusion.** Target achievement is defined over eligible demand-response steps on scheduled demand and computed identically for every arm. The archived field named `retention` is controlled scheduled energy over baseline scheduled energy; it is not outage retention, command availability or service preservation, and it has been renamed `energy_ratio` and excluded from every resilience claim.

**Failure condition.** The profile split supplies different benchmark shapes, but their statistical independence and sampling frame are not established, and the profiles are not measured household data. A common shape across all groups imposes spatial synchrony within each episode. Resampling profiles cannot resolve a limitation that lives in the model rather than the sampling. Stronger generalisation needs independently varying group traces and explicitly separated household or calendar-day samples.

## 6.4 Selection, exposure and multiplicity

**Conclusion.** Every interval in this paper is reproducible from the released scripts: 5,000 episode-cluster resamples with bootstrap streams derived deterministically from a comparison key, and Holm correction across secondary contrasts. The two selection families each searched 144 candidates under one objective, and the winner's curse is quantified in Section 5.2 rather than absorbed into the reported score.

**Failure condition.** Three things limit this. First, the 21 validation episodes were consumed by five separate uses — 144 learned candidates, the five-seed retrain of the winner, 144 rule points, a validation gap check and a direct policy search on a training subsample — so their agreement with test is a consistency check rather than a generalisation estimate. Second, the 21 test episodes were read during Run 1, so the held-out comparison is post-hoc with respect to those episodes even though it is pre-registered with respect to the pipeline. Third, the cluster bootstrap undercovers when clusters are few [@cameron2008bootstrap]. If that bias is material the intervals are too narrow, and the consequence cuts both ways. It makes the inconclusive primary comparison **more** secure, since a wider interval cannot turn a half-width above δ into one below it. It puts the contrasts that exclude zero at risk, unequally: the point estimate exceeds its half-width by 1.93× for C3 − C1, 2.04× for C3b − C1, 1.83× for C4v2 − C2v2 and 2.58× for the repair gain, but by only **1.41×** for C4 − C2, so a 40% widening would move that pair across zero and leave the three headline pairs intact. C4 − C2 is not a headline result, but it shows which claim sits closest to the boundary. A fourth limitation belongs here: the replica that scored the candidates computes its switching term from one group's count against the 50-group normaliser, inflating the score by up to 0.0026, an order of magnitude below the margin; under the corrected score every winner keeps its rank, so no selection changed.

## 6.5 Physical and deployment boundary

**Conclusion.** Physical quantities are reported under two definitions that are never interchanged: the controller's own 0.06 pu voltage trigger, never crossed in 59,850 records, and the ANSI reporting band, crossed on the upper side only and identically in every arm including the uncontrolled reference (Section 5.6).

**Failure condition.** The study uses quasi-static power flow only and does not evaluate frequency, its rate of change, protection coordination, islanding, transients or hardware actuation. The model omits appliance comfort, essential-load exclusions, rebound, edge power loss, arbiter restart and multi-arbiter ownership. Bus-number sorting is a reproducible mapping rule, not a verified topology partition, and the assigned line ratings require a source-to-element audit. Because violations are outcomes rather than enforced constraints, the design is not claimed ready for a real feeder, nor the arbiter to resolve consensus across multiple arbiters."""

S7 = """\
# 7. Discussion

Run 1 and Run 2 are the same architecture measured through two pipelines. That is the whole of the causal claim, and it is worth stating in the form that is defensible: correcting the training pipeline moves the learned arm from inferior beyond the declared margin, under the corrected simulator, to a held-out difference of −0.0040 against a rule tuned over the same candidate budget. Run 1's 13.32-point shortfall is therefore attributable to the pipeline, not to the action-dependent linear representation. This is not an improvement of the method, because the method did not change; it is a change in what was being measured.

What follows from the correction is narrower than it first appears. The held-out interval's half-width of 0.0207 exceeds the frozen margin δ = 0.02, so the comparison is inconclusive at the declared resolution: not a win, not a loss, and not an equivalence. A reader who wants a verdict will not get one here, and the margin was not raised to supply it. The substantive repair claim rests on its own interval, which excludes zero, and the two must not be traded against each other — the repair is a statement about pipelines, the primary comparison a statement about policies.

Two rival readings deserve an answer rather than a dismissal. The first is that the repair did not so much improve the learner as make it more like the rule: full-episode replay and state coverage could, in principle, push a linear value function toward the baseline's own behaviour, in which case part of the gain measures convergence rather than a distinct capability. The transfer study is evidence against the strong form of this reading — the never-retrained learner is ahead of a per-condition re-tuned rule at two shifted operating points and behind at a third — but it does not exclude a weaker form, in which some of the gain is convergence. The second is that the primary comparison is underpowered rather than genuinely unresolved, and that a larger episode set would show a difference in one direction. That is possible and untested here. What can be said is that the direction is not predictable from the present interval, which sits nearly centred on zero, and that the pre-registered decision rule is what makes the difference between the two readings reportable at all.

The architecture question resolves separately and more cleanly. Adding a guarded local path to a policy that already exists is worth 0.147 to 0.220 points of outage-window achievement, in every one of the four policy pairs tested and after Holm correction. The mechanism explains why the number is easy to miss: local authority is active for about half a percent of run time, so the value sits in a few high-consequence instants and a whole-episode average dilutes it. The operational reading is a sequencing rule rather than a selection rule — secure the local path first, then decide which policy drives the cloud side, because the first decision is worth more than the second and is not contingent on it.

Three observations would overturn parts of this account, and naming them is more useful than defending it. If a held-out evaluation with enough episodes to bring the half-width below the margin placed the interval entirely outside ±δ, the inconclusive verdict would become a directional one; nothing in the present data predicts which direction. If the workload were reconstructed with independently varying group profiles and the parameter-free heuristic lost its parity with the enumerated rule, then the tuning budget would be buying something after all. The second half of C4 would then fail. And if the architecture were evaluated with several arbiters per feeder, the measured value of the local path could fall, because part of what it buys here is the absence of a competing authority. None of these is a call for more work in general; each is a specific experiment whose outcome would change a specific sentence above."""

S8 = """\
# 8. Conclusion

This study reports READ-Grid as an auditable feeder simulation and evaluates the same architecture twice: in Run 1, and in Run 2 through an audited, pre-registered pipeline.

**What is established.** Correcting the training pipeline moves the learned arm by 0.1152 points on the held-out split, with an interval that excludes zero, and accounts for the 13.32-point shortfall of Run 1 without invoking the action-dependent linear representation. Adding a guarded local path to a policy that already exists contributes 0.147 to 0.220 points of outage-window achievement across four policy pairs, all significant after correction, with a sign that does not depend on the policy type.

**What is not established.** A learning advantage at the nominal operating point: the primary comparison is inconclusive, because its interval half-width exceeds the pre-registered margin and the margin was not raised. Nor is an electrical-safety guarantee, a field-ready deployment, or a value of the architecture that is the same size under every policy. A parameter-free heuristic matching the enumerated rule belongs to the same picture: the tuning budget buys nothing resolvable at this resolution.

**What is released.** The Run 2 artefacts, hashes and analysis scripts, so that the comparison can be repeated on fresh episodes with enough of them to resolve the declared margin. The scripts that regenerate every interval are part of the release rather than a supplement to it. The negative and inconclusive results are released with them, on the view that a stated boundary is a result and a re-run at a looser boundary is not."""


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #

LABELS = [
    ("The study reports two grids.",
     "The study reports two runs of the same architecture, and they are never pooled."),
    ("**Grid 1 (first evaluation, Section 5.1).**", "**Run 1 (Section 5.1).**"),
    ("i.e. 7 \u00d7 21 \u00d7 95. This grid is analysed as archived.",
     "i.e. 7 \u00d7 21 \u00d7 95. It is analysed as recorded, and its numbers are not revised."),
    ("**Grid 2 (corrected evaluation, Sections 5.2\u20135.7).**", "**Run 2 (Sections 5.2\u20135.7).**"),
    ("The v2 policy set has 30 instances:",
     "The Run 2 policy set has 30 instances. Arm identifiers ending in `v2` denote the Run 2 "
     "configuration; the unsuffixed identifier denotes Run 1:"),
    ("The v1 rule (C3)", "The original rule (C3)"),
    ("The v1 state `x`", "The original state `x`"),
    ("Two changes are made in v2.", "Two changes are made in the Run 2 representation."),
    ("The v1 fitted-Q target", "The original fitted-Q target"),
    ("The v1 eight-dimensional representation", "The original eight-dimensional representation"),
    ("The v1 shortfall is therefore", "The shortfall is therefore"),
    ("closes the v1 gap to the v1 rule", "closes the gap to the original rule"),
    ("the *v1 rule*", "the *original* rule"),
    ("the v1 learner's disadvantage to the v1 rule",
     "the original learner's disadvantage to the original rule"),
    ("against the v1 rule", "against the original rule"),
    ("C3 v1 local rule", "C3 original local rule"),
    ("C4 v1 learner", "C4 original learner"),
    ("C2 v1 cloud-only learned", "C2 original cloud-only learned"),
    ("(v1 representation)", "(original representation)"),
    ("(v1 parameters)", "(original parameters)"),
    ("for the v1-style training configuration", "for the original training configuration"),
    ("buys nothing resolvable over the v1 rule", "buys nothing resolvable over the original rule"),
    ("(v1 learner: value of the local fallback)", "(original learner: value of the local fallback)"),
    ("already read during the first evaluation", "already read during Run 1"),
    ("The first evaluation's producing source bytes", "Run 1's producing source bytes"),
    ("the independently recalculated means of the archived grid",
     "the independently recalculated means of Run 1"),
    ("in the corrected grid, four behave", "in Run 2, four behave"),
    ("the only paired intervals in the corrected evaluation are those",
     "the only paired intervals in Run 2 are those"),
    ("with respect to the corrected pipeline, but the episodes",
     "with respect to Run 2, but the episodes"),
    ("The first evaluation's tables derive from `runs/stage_main.jsonl`; the corrected evaluation derives from",
     "Run 1's tables derive from `runs/stage_main.jsonl`; Run 2 derives from"),
    ("The rebuilt model uses an IEEE 123-bus feeder", "The simulation uses an IEEE 123-bus feeder"),
    ("The research archive contains the rebuilt source", "The research archive contains the simulation source"),
    ("The archived grid establishes the phenomenon", "Run 1 establishes the phenomenon"),
    ("the first grid carried impossible negative delays", "Run 1 carried impossible negative delays"),
    ("and the corrected grid contains none", "and Run 2 contains none"),
]



# --------------------------------------------------------------------------- #
# polish pass (nature-polishing: sentence length, dashes, terminology)
# --------------------------------------------------------------------------- #
POLISH = [
    # --- Abstract -----------------------------------------------------------
    ("A first run \u2014 13,965 feeder-coupled records over 21 episodes and 95 faults \u2014 put the "
     "tuned local rule at 58.57% target achievement against 45.25% for the learned policy, a gap "
     "of \u221213.32 points.",
     "A first run covered 13,965 records across 21 episodes and 95 faults. The tuned local rule "
     "reached 58.57% target achievement and the learned policy 45.25%, a gap of \u221213.32 points."),
    ("is **inconclusive**: the mean difference is \u22120.0040, 95% interval [\u22120.0239, +0.0174], "
     "whose half-width of 0.0207 exceeds the pre-registered margin \u03b4 = 0.02, so superiority, "
     "non-inferiority and equivalence are unestablished.",
     "is **inconclusive**. The mean difference is \u22120.0040 (95% interval [\u22120.0239, +0.0174]), "
     "and a half-width of 0.0207 against the pre-registered margin \u03b4 = 0.02 leaves superiority, "
     "non-inferiority and equivalence unestablished."),
    ("and separates command-path availability from policy quality",
     "and separates path availability from policy quality"),
    ("a Bellman target maximised over inadmissible successor actions",
     "a Bellman target maximised over inadmissible successors"),
    ("not the action-dependent linear representation", "not the action-dependent representation"),

    # --- 1. Introduction ----------------------------------------------------
    ("[@zhang2021edge; @tao2022hybrid; @jiang2020cloud], and compact learners have been shown to "
     "run on device-class hardware without a cloud connection [@gao2025lightweight].",
     "[@zhang2021edge; @tao2022hybrid; @jiang2020cloud]. Compact learners have also been run on "
     "device-class hardware without a cloud connection [@gao2025lightweight]."),
    ("were searched, and the search is itself a source of variance the rule does not carry "
     "[@dodge2019show; @musgrave2020metric; @melis2018state; @patterson2024empirical; @lucic2018gans].",
     "were searched [@dodge2019show; @musgrave2020metric; @melis2018state; @patterson2024empirical; "
     "@lucic2018gans]. That search is itself a source of variance the rule does not carry."),
    ("a more thoroughly tuned rule \u2014 or is the evidence insufficient?",
     "a more thoroughly tuned rule, or is the evidence insufficient?"),
    ("**C1 \u2014 A reproducible", "**C1. A reproducible"),
    ("**C2 \u2014 An attribution", "**C2. An attribution"),
    ("**C3 \u2014 A measurement", "**C3. A measurement"),
    ("**C4 \u2014 An honest", "**C4. An honest"),

    # --- 2. Related work ----------------------------------------------------
    ("What this work establishes is availability under a degraded network; what it does not isolate "
     "is whether the local policy is preferable to a simpler local alternative, because the "
     "comparison is against a cloud controller that has lost its path.",
     "What this work establishes is availability under a degraded network. What it does not isolate "
     "is whether the local policy is preferable to a simpler alternative, because the comparison is "
     "against a cloud controller that has lost its path."),
    ("Two patterns there shape the design here: the learned controller is compared with a hand-set "
     "rule rather than one tuned under an equal budget, and the result is reported as a ranking of "
     "point estimates rather than an interval on a difference.",
     "Two patterns there shape the design here. The learned controller is usually compared with a "
     "hand-set rule rather than one tuned under an equal budget. The result is usually reported as "
     "a ranking of point estimates rather than an interval on a difference."),
    ("Tuning effort is equalised in candidate evaluations \u2014 144 per family under one selection "
     "objective \u2014 and the residual asymmetry is declared",
     "Tuning effort is equalised in candidate evaluations (144 per family under one selection "
     "objective), and the residual asymmetry is declared"),
    ("through the `dss-python` package [@dsspython2024]; the IEEE 123-bus example supplies",
     "through the `dss-python` package [@dsspython2024]. The IEEE 123-bus example supplies"),

    # --- 3. System model ----------------------------------------------------
    ("into 60 s windows, selected the latest valid arrival in each window, and passed that "
     "timestamp to the arbiter",
     "into 60 s windows, took the latest valid arrival in each window, and passed its timestamp "
     "to the arbiter"),
    ("could therefore affect an earlier decision \u2014 an intra-interval look-ahead.",
     "could therefore affect an earlier decision, an intra-interval look-ahead."),
    ("A **dwell-age** feature is added \u2014 minutes since the last action-level change, capped at "
     "3 \u2014 because the 180 s dwell constraint",
     "A **dwell-age** feature is added (minutes since the last action-level change, capped at 3) "
     "because the 180 s dwell constraint"),

    # --- 5. Results ---------------------------------------------------------
    ("The limitation is broader than this particular outcome: another set of action offsets could "
     "select a different fixed ranking, so the ablation does not establish that every training "
     "method with a concatenated representation must return the uncontrolled action.",
     "The limitation is broader than this outcome. Another set of action offsets could select a "
     "different fixed ranking, so the ablation does not establish that every "
     "concatenated-representation method must return the uncontrolled action."),
    ("but its greedy variant was *not* an improvement: across the 144 learned candidates",
     "but its greedy variant was *not* an improvement. Across the 144 learned candidates"),
    ("no paired interval is quoted for it; the only paired intervals in Run 2 are those of the "
     "held-out run (Section 5.3).",
     "no paired interval is quoted for it. The only paired intervals in Run 2 are those of the "
     "held-out run (Section 5.3)."),
    ("in the sense defined in `docs/test_exposure_record.md`: it is pre-registered with respect to "
     "Run 2, but the episodes themselves are not fresh.",
     "in the sense defined in `docs/test_exposure_record.md`. It is pre-registered with respect to "
     "Run 2, but the episodes themselves are not fresh."),
    ("The protocol hash, learner configuration hash, split hash, workload index hash, all 20 "
     "source-file hashes and all 10 trained-checkpoint hashes were recorded in the run manifest "
     "and re-verified against the frozen snapshot afterwards with zero discrepancies.",
     "The run manifest records the protocol, learner configuration, split and workload index "
     "hashes, all 20 source-file hashes and all 10 trained-checkpoint hashes. Every one was "
     "re-verified against the frozen snapshot afterwards, with zero discrepancies."),
    ("so the per-episode inference totals are 1,440\u00d7 these values; the two are the same "
     "measurement expressed at different granularity and must not be quoted as two independent "
     "costs.",
     "so a per-episode total is 1,440\u00d7 these values. The two are the same measurement at "
     "different granularity and must not be quoted as two independent costs."),
    ("and its half-width 0.0207 exceeds the frozen margin \u03b4 = 0.02, so the sixth condition "
     "applies: **inconclusive, insufficient power for the declared margin**.",
     "and its half-width 0.0207 exceeds the frozen margin \u03b4 = 0.02. The sixth condition "
     "therefore applies: **inconclusive, insufficient power for the declared margin**."),
    ("(both equivalent): enumerating 144 rule points buys nothing resolvable",
     "(both equivalent). Enumerating 144 rule points buys nothing resolvable"),
    ("The comparison is deliberately not turned into a test: `ceil` is a feasible in-budget value "
     "but is a block-interior point rather than a proven bound, `floor` is an optimistic "
     "discretisation, both see the whole day in advance, and the bracket was computed on the "
     "validation split.",
     "The comparison is deliberately not turned into a test. `ceil` is a feasible in-budget value "
     "but a block-interior point rather than a proven bound; `floor` is an optimistic "
     "discretisation; both see the whole day in advance; and the bracket was computed on the "
     "validation split."),
    ("zero takeovers and zero recoveries, with the authority epoch constant at 0.",
     "zero takeovers and zero recoveries. The authority epoch stays constant at 0."),
    ("The gain is **robust in sign and order of magnitude** across policy type \u2014 the guarded "
     "architecture adds 0.147\u20130.220 points to outage-window achievement whether the cloud policy "
     "is a rule, a tuned rule or a learned model \u2014 but it is **not policy-invariant in size**.",
     "The gain is **robust in sign and order of magnitude** across policy type: the guarded "
     "architecture adds 0.147\u20130.220 points to outage-window achievement whether the cloud policy "
     "is a rule, a tuned rule or a learned model. It is **not policy-invariant in size**."),
    ("The four estimates span 0.0731, which is 3.7 times the 0.02 margin this study uses to judge "
     "differences of this kind, so the architecture gain and the policy contrast are not measured "
     "at the same resolution and the two must not be read as if they were.",
     "The four estimates span 0.0731, 3.7 times the 0.02 margin this study uses to judge "
     "differences of this kind. The architecture gain and the policy contrast are therefore not "
     "measured at the same resolution and must not be read as if they were."),
    ("supports decision B \u2014 the architecture is the contribution",
     "supports decision B: the architecture is the contribution"),
    ("Six conditions were therefore evaluated \u2014 the nominal point, cap relaxed to \u03b4 = 8% and "
     "tightened to \u03b4 = 16%, and budgets of 2%, 4% and 6% \u2014 with the rule either left frozen, "
     "re-tuned per condition, or (for the learner) **never retrained**.",
     "Six conditions were therefore evaluated: the nominal point, cap relaxed to \u03b4 = 8% and "
     "tightened to \u03b4 = 16%, and budgets of 2%, 4% and 6%. For each, the rule was either left "
     "frozen, re-tuned per condition, or (for the learner) **never retrained**."),
    ("Frozen **composite selection score** \u2014 Equation (5), not the achievement rate \u2014 pooled over",
     "Frozen **composite selection score** (Equation (5), not the achievement rate), pooled over"),
    ("(+0.0328 [+0.0194, +0.0451]), and significantly **behind** at budget 2%",
     "(+0.0328 [+0.0194, +0.0451]). It is significantly **behind** at budget 2%"),
    ("*is* breached, but only on the upper side, and the extrema are identical across arms "
     "including the uncontrolled reference:",
     "*is* breached, but only on the upper side. The extrema are identical across arms, including "
     "the uncontrolled reference:"),
    ("is deliberately **not** invoked: this study does not evaluate how long or how often a voltage "
     "sits outside the band",
     "is deliberately **not** invoked. This study does not evaluate how long or how often a "
     "voltage sits outside the band"),
    ("not an effect of demand-response control \u2014 control *reduces* the time spent outside the band",
     "not an effect of demand-response control: control *reduces* the time spent outside the band"),
    ("runs two controllers \u2014 one cloud and one local \u2014 so a cloud-plus-local decision pair",
     "runs two controllers, one cloud and one local, so a cloud-plus-local decision pair"),
    ("**one control step for the whole feeder** \u2014 all 5 domains and all 50 groups \u2014 accumulated "
     "in `sim.py` between the first cloud decision and the last local decision; it is not a "
     "per-controller or per-group latency.",
     "**one control step for the whole feeder**, all 5 domains and all 50 groups, accumulated in "
     "`sim.py` between the first cloud decision and the last local decision. It is not a "
     "per-controller or per-group latency."),
    ("of a 24 h episode, which is the same measurement at a coarser granularity; it is not quoted "
     "separately",
     "of a 24 h episode, the same measurement at a coarser granularity. It is not quoted separately"),

    # --- 6. Validity --------------------------------------------------------
    ("timed on an actual authority *change*; Run 1 carried impossible negative delays",
     "timed on an actual authority *change*. Run 1 carried impossible negative delays"),
    ("fifteen of the 21 recorded source hashes differ from the current files, because two defect "
     "fixes dated after the first held-out run both touch `sim.py`.",
     "fifteen of the 21 recorded source hashes differ from the current files. Two defect fixes "
     "dated after the first held-out run both touch `sim.py`."),
    ("five separate uses \u2014 144 learned candidates, the five-seed retrain of the winner, 144 rule "
     "points, a validation gap check and a direct policy search on a training subsample \u2014 so "
     "their agreement with test is a consistency check rather than a generalisation estimate.",
     "five separate uses: 144 learned candidates, the five-seed retrain of the winner, 144 rule "
     "points, a validation gap check and a direct policy search on a training subsample. Their "
     "agreement with test is therefore a consistency check rather than a generalisation estimate."),
    ("an order of magnitude below the margin; under the corrected score every winner keeps its rank",
     "an order of magnitude below the margin. Under the corrected score every winner keeps its rank"),

    # --- 7. Discussion ------------------------------------------------------
    ("it is worth stating in the form that is defensible: correcting the training pipeline moves",
     "it is worth stating defensibly. Correcting the training pipeline moves"),
    ("must not be traded against each other \u2014 the repair is a statement about pipelines",
     "must not be traded against each other: the repair is a statement about pipelines"),
    ("did not so much improve the learner as make it more like the rule: full-episode replay",
     "did not so much improve the learner as make it more like the rule. Full-episode replay"),
    ("at two shifted operating points and behind at a third \u2014 but it does not exclude a weaker "
     "form",
     "at two shifted operating points and behind at a third. It does not exclude a weaker form"),
    ("rather than a selection rule \u2014 secure the local path first",
     "rather than a selection rule. Secure the local path first"),

    # --- 8. Conclusion ------------------------------------------------------
    ("with an interval that excludes zero, and accounts for the 13.32-point shortfall of Run 1 "
     "without invoking the action-dependent linear representation.",
     "with an interval that excludes zero. It accounts for the 13.32-point shortfall of Run 1 "
     "without invoking the action-dependent representation."),
    ("Adding a guarded local path to a policy that already exists contributes 0.147 to 0.220 points "
     "of outage-window achievement across four policy pairs, all significant after correction, with "
     "a sign that does not depend on the policy type.",
     "Adding a guarded local path to an existing policy contributes 0.147 to 0.220 points of "
     "outage-window achievement across four policy pairs, all significant after correction. The "
     "sign does not depend on the policy type."),
    # --- second pass: sentences still over 30 words -------------------------
    ("The inference that a local rule is therefore the right choice at the edge does not follow, "
     "and neither does the converse inference that a learner is preferable because it degrades "
     "more gracefully.",
     "It does not follow that a local rule is therefore the right choice at the edge, nor that a "
     "learner is preferable because it degrades more gracefully."),
    ("which turns the reported contrast into an order statistic of the selection set rather than "
     "an estimate of its generalisation",
     "turning the reported contrast into an order statistic of that set rather than an estimate of "
     "its generalisation"),
    ("have been demonstrated for building demand response [@zhang2021edge], for hybrid cloud-edge "
     "coordination [@tao2022hybrid], and for closed-loop price response of aggregated air "
     "conditioners, including one that models packet dropouts [@jiang2020cloud].",
     "have been demonstrated for building demand response [@zhang2021edge], hybrid cloud-edge "
     "coordination [@tao2022hybrid] and closed-loop price response of air conditioners "
     "[@jiang2020cloud], one of which models packet dropouts."),
    ("and treats millisecond command-delay assumptions and whole-minute decision updates as "
     "separate.",
     "and keeps millisecond command-delay assumptions separate from whole-minute decision updates."),
    ("The masking is a correction, not an advantage: Section 5.1 reports a negative result",
     "The masking is a correction, not an advantage. Section 5.1 reports a negative result"),
    ("[@ruelens2017residential] and to distribution-level voltage control",
     "[@ruelens2017residential] and distribution-level voltage control"),
    ("The pre-registered decision procedure evaluates five conditions independently and reports the "
     "strongest one that holds: superior (interval entirely above zero), equivalent (interval inside "
     "\u00b1\u03b4), non-inferior (interval lower bound above \u2212\u03b4), inferior (interval entirely below "
     "zero), and inferior beyond the margin (interval upper bound below \u2212\u03b4); if none holds, the "
     "result is inconclusive.",
     "The pre-registered decision procedure evaluates five conditions independently and reports the "
     "strongest one that holds. Superior means an interval entirely above zero; equivalent, an "
     "interval inside \u00b1\u03b4; non-inferior, a lower bound above \u2212\u03b4; inferior, an interval "
     "entirely below zero; and inferior beyond the margin, an upper bound below \u2212\u03b4. If none "
     "holds, the result is inconclusive."),
    ("in a normalised single-group replica that reproduces the full simulator's per-step dynamics.",
     "in a normalised single-group replica reproducing the full simulator's per-step dynamics."),
    ("The two differ because six fault families (F4\u2013F7, F9, F10) are expanded to fifteen variants "
     "each while five (F0\u2013F3, F8) have one, so pooled weighting gives the expanded families 94.7% "
     "of the weight.",
     "Six fault families (F4\u2013F7, F9, F10) expand to fifteen variants each while five (F0\u2013F3, "
     "F8) have one, so pooled weighting gives the expanded families 94.7% of the weight."),
    ("cannot be distinguished from rule++ either as better, as non-inferior, or as equivalent at "
     "the declared resolution.",
     "cannot be distinguished from rule++ as better, non-inferior or equivalent at the declared "
     "resolution."),
    ("so unlike the validation split the two criteria **agree in direction** here; the residual "
     "disagreement is one of magnitude, not of sign.",
     "so unlike the validation split the two criteria **agree in direction** here. The residual "
     "disagreement is one of magnitude, not of sign."),
    ("Both are reported because Equation (5) is a trade-off and a preference between them depends "
     "on the operational valuation of switching and curtailment, which the study does not resolve "
     "by picking the criterion that favours one arm.",
     "Both are reported because Equation (5) is a trade-off, and a preference between them depends "
     "on how switching and curtailment are valued operationally. The study does not resolve that by "
     "picking the criterion that favours one arm."),
    ("The gain is **robust in sign and order of magnitude** across policy type: the guarded "
     "architecture adds",
     "The gain is **robust in sign and order of magnitude** across policy type. The guarded "
     "architecture adds"),
    ("What the four pairs establish is that the local path is worth having whoever drives the cloud "
     "side; they do not establish that its worth is the same in each case.",
     "The four pairs establish that the local path is worth having whoever drives the cloud side. "
     "They do not establish that its worth is the same in each case."),
    ("One boundary must be stated: the corrected learner is **not** better than rule++ on the "
     "outage window itself (outage achievement 0.6646 against 0.7250; paired difference \u22120.0275 "
     "with interval",
     "One boundary needs stating: the corrected learner is **not** better than rule++ on the outage "
     "window itself (outage achievement 0.6646 against 0.7250; paired difference \u22120.0275, "
     "interval"),
    ("and the difference is negative in 15 of the 21 episodes (median \u22120.0385), so it is not "
     "driven by an outlier.",
     "and it is negative in 15 of the 21 episodes (median \u22120.0385), so it is not an outlier "
     "effect."),
    ("so its intervals are narrower than the held-out intervals and are not comparable to them; it "
     "is reported as a within-scenario ranking.",
     "so its intervals are narrower than the held-out ones and not comparable to them. It is "
     "reported as a within-scenario ranking."),
    ("two are undetermined: what the data support is that the nominal ordering is unstable "
     "off-nominal",
     "two are undetermined. What the data support is that the nominal ordering is unstable "
     "off-nominal"),
    ("The intervals above are **post-hoc**: they were computed after the transfer table had been "
     "read, on the same 21 episodes, by resampling episodes with the same frozen bootstrap, and "
     "they are not part of the pre-registered analysis.",
     "The intervals above are **post-hoc**: computed after the transfer table had been read, on the "
     "same 21 episodes, by resampling episodes with the same frozen bootstrap. They are not part of "
     "the pre-registered analysis."),
    ("they describe this seed, not the training distribution, so they bound the uncertainty of the "
     "comparison on this run rather than the variability of the learned arm in general.",
     "they describe this seed, not the training distribution. They bound the uncertainty of this "
     "comparison, not the variability of the learned arm in general."),
    ("this threshold is **never crossed**; the worst deviation is 0.0277 pu on the undervoltage "
     "side, and the solved per-unit minimum stays",
     "this threshold is **never crossed**. The worst deviation is 0.0277 pu on the undervoltage "
     "side, and the per-unit minimum stays"),
    ("not an effect of demand-response control: control *reduces* the time spent outside the band",
     "not an effect of demand-response control. Control *reduces* the time spent outside the band"),
    ("Each value is the time of **one control step for the whole feeder**, all 5 domains and all 50 "
     "groups, accumulated in `sim.py` between the first cloud decision and the last local decision.",
     "Each value times **one control step for the whole feeder**, all 5 domains and all 50 groups, "
     "accumulated in `sim.py` between the first cloud and the last local decision."),
    ("across 55 controller invocations, which is negligible for any processor that could run the "
     "rule.",
     "across 55 controller invocations, negligible for any processor that could run the rule."),
    ("does not constrain deployment at this control rate; the cost of selecting the learned arm is",
     "does not constrain deployment at this control rate. The cost of selecting the learned arm is"),
    ("The authority update still runs on the 60 s control grid while the configured takeover "
     "threshold is 5 s, so no claim of immediate local actuation at 5 s is made; millisecond "
     "command-delay parameters are kept separate from whole-minute decision updates.",
     "The authority update still runs on the 60 s control grid while the configured takeover "
     "threshold is 5 s, so no claim of immediate local actuation at 5 s is made. Millisecond "
     "command-delay parameters are kept separate from whole-minute decision updates."),
    ("The transfer study is evidence against the strong form of this reading \u2014 the "
     "never-retrained learner is ahead",
     "The transfer study is evidence against the strong form of this reading: the never-retrained "
     "learner is ahead"),
    ("What can be said is that the direction is not predictable from the present interval, which "
     "sits nearly centred on zero, and that the pre-registered decision rule is what makes the "
     "difference between the two readings reportable at all.",
     "What can be said is that the direction is not predictable from the present interval, which "
     "sits nearly centred on zero. The pre-registered decision rule is what makes the difference "
     "between the two readings reportable at all."),
    ("and the two must not be traded against each other: the repair is a statement about pipelines",
     "and the two must not be traded against each other. The repair is a statement about pipelines"),
    ("The mechanism explains why the number is easy to miss: local authority is active for about "
     "half a percent of run time, so the value sits",
     "The mechanism explains why the number is easy to miss. Local authority is active for about "
     "half a percent of run time, so the value sits"),
    ("the inconclusive verdict would become a directional one; nothing in the present data "
     "predicts which direction.",
     "the inconclusive verdict would become a directional one. Nothing in the present data "
     "predicts which direction."),
    ("is worth 0.147 to 0.220 points of outage-window achievement, in every one of the four policy "
     "pairs tested and after Holm correction.",
     "is worth 0.147 to 0.220 points of outage-window achievement, in all four policy pairs tested "
     "and after Holm correction."),
    ("to a held-out difference of \u22120.0040 against a rule tuned over the same candidate budget.",
     "to a held-out difference of \u22120.0040 against a rule tuned over the same budget."),
    ("The held-out interval's half-width of 0.0207 exceeds the frozen margin \u03b4 = 0.02, so the "
     "comparison is inconclusive at the declared resolution: not a win, not a loss, and not an "
     "equivalence.",
     "The held-out interval's half-width of 0.0207 exceeds the frozen margin \u03b4 = 0.02. The "
     "comparison is therefore inconclusive at the declared resolution: not a win, not a loss, not "
     "an equivalence."),
    ("toward the baseline's own behaviour, in which case part of the gain measures convergence "
     "rather than a distinct capability.",
     "toward the baseline's own behaviour, in which case part of the gain measures convergence, not "
     "a distinct capability."),
    ("And if the architecture were evaluated with several arbiters per feeder, the measured value "
     "of the local path could fall, because part of what it buys here is the absence of a competing "
     "authority.",
     "If several arbiters served each feeder, the measured value of the local path could fall, "
     "because part of what it buys here is the absence of a competing authority."),
]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
# Only Run 2 figures are embedded; the Run 1 figures are not (D4). Numbering is
# by order of appearance in the assembled text. Each caption states the scope of
# what the figure shows and keeps the claims in the body prose; `v1` cannot
# appear in any of them (D6: "Run 1" / "original" instead). The PNG itself must
# exist in figures/ and is also staged into paper_package/figures/.
FIGURES = [
    ("fig_v2_training_ladder.png",
     "The pre-registered training ladder of twelve configurations, five seeds each, "
     "evaluated in the normalised replica that reproduces the full simulator's "
     "per-step behaviour. Bars are five-seed mean selection scores (Equation 5); "
     "the dashed line marks the original-style reference configuration at 0.4346."),
    ("fig_v2_val_arms.png",
     "Validation-split scores for the Run 2 policy set, with the selection score "
     "(Equation 5) and the raw achievement rate, five-seed means for the learned "
     "arms. The two criteria order the arms differently, which is why both are "
     "reported; the validation split carries no interval, because it was consumed by "
     "candidate selection."),
    ("fig_v2_gap_closure.png",
     "The archived Run 1 shortfall beside the Run 2 validation comparison. The red "
     "bar is the Run 1 learner-minus-rule difference \u22120.1332 with its "
     "episode-clustered 95% interval [\u22120.2038, \u22120.0698]; the two Run 2 "
     "validation differences are point estimates."),
    ("fig_v2_baselines.png",
     "The strengthened validation-split baseline set, comprising the parameter-free heuristic, "
     "the rule tuned over 144 candidates, the original hand-set rule, and both "
     "learned arms as five-seed means. Bars are raw achievement rates; the tuned "
     "rule and the corrected learner each evaluated the same 144 candidates."),
    ("fig_v2_primary_forest.png",
     "Held-out achievement-rate differences with episode-clustered 95% intervals "
     "over 21 episodes (Section 5.3). Panel (a) shows contrasts at the margin, where the "
     "pre-registered equal-weight row is filled and its pooled-weight alternative "
     "open, with the equivalence zone \u00b1\u03b4 = 0.02 shaded. Panel (b) shows "
     "contrasts far beyond \u03b4."),
    ("fig_v2_outage_coverage.png",
     "Demand-response steps by hour of simulated day across the 21 held-out "
     "episodes, with the configured outage windows, the frozen F3 fault "
     "(18:30\u201320:30) and the extension fault F3E (17:00\u201324:00). F3 covers "
     "21.7% of test-split steps and fully covers no episode."),
    ("fig_v2_transfer.png",
     "Transfer study of the learner minus the per-condition re-tuned rule at six "
     "operating points, with post-hoc episode-clustered intervals over the same 21 "
     "episodes. Filled markers mark intervals excluding zero, open markers "
     "intervals crossing zero. The learner is never retrained; one learner seed "
     "is used."),
]

# Sentences that point to the figures. Applied after POLISH so the anchors see
# the final prose; each must hit exactly once.
FIG_REFS = [
    ("once replay, coverage and the corrected state are combined.",
     "once replay, coverage and the corrected state are combined (Figure 1)."),
    ("**The two criteria disagree, and both are reported.**",
     "**The two criteria disagree, and both are reported** (Figure 2)."),
    ("The \u221213.32 point deficit of Section 5.1 is gone.",
     "The \u221213.32 point deficit of Section 5.1 is gone (Figure 3)."),
    ("rather than the selected maximum.",
     "rather than the selected maximum (Figure 4)."),
    ("as better, non-inferior or equivalent at the declared resolution.",
     "as better, non-inferior or equivalent at the declared resolution (Figure 5)."),
    ("**does not exist in the frozen design**.",
     "**does not exist in the frozen design** (Figure 6)."),
    ("**not** that the learner wins off-nominal.",
     "**not** that the learner wins off-nominal (Figure 7)."),
]

# Where each figure goes: the image and caption are inserted directly after the
# anchor, which includes the reference where the two coincide (so this runs
# after FIG_REFS). Number = position in FIGURES. The Table 2 row is anchored on
# its content rather than its column spacing.
FIG_INSERTS = [
    (1, "once replay, coverage and the corrected state are combined (Figure 1)."),
    (2, "| C0 uncontrolled"),
    (3, "The study does not resolve that by choosing the criterion that favours one arm."),
    (4, "rather than the selected maximum (Figure 4)."),
    (5, "as better, non-inferior or equivalent at the declared resolution (Figure 5)."),
    (6, "leaving the frozen protocol and its hash unmodified."),
    (7, "is not guaranteed to improve on test*."),
]


def main() -> None:
    v2 = parse_top(V2.read_text(encoding="utf-8"))

    s3 = top(
        v2, "3. Architecture and controller formulation",
        ("sparsely sample the learner's hyperparameter space [8]",
         "sparsely sample the learner's hyperparameter space [@bergstra2012random]"),
        ("constraint order used at actuation (ramp → dwell → budget).",
         "constraint order used at actuation (ramp → dwell → budget) "
         "[@huang2022closer; @zahavy2018learn]."),
        ("The source fits the model by batch fitted-Q iteration with regularised least squares.",
         "The source fits the model by batch fitted-Q iteration with regularised least squares "
         "[@riedmiller2005neural; @lagoudakis2003least], a family already applied to "
         "thermostatically controlled demand response [@ruelens2017residential] and to "
         "distribution-level voltage control [@xu2020optimal]."),
        ("A regression test pins the behaviour by mutating arrivals after a decision instant "
         "and asserting that the earlier decision is unchanged.",
         "A regression test pins the behaviour."),
        ("These values are new model assumptions. They are not recovered original settings or "
         "measured communication properties.",
         "These are new model assumptions, not recovered original settings or measured "
         "communication properties."),
        ("First, the voltage-deviation feature is removed: it is not a controllable quantity at "
         "decision time and its inclusion risked encoding the physical outcome in the input. "
         "Second, a **dwell-age** feature is added \u2014 the number of minutes since the last "
         "action-level change, capped at 3 \u2014",
         "The voltage-deviation feature is removed: it is not controllable at decision time, "
         "and including it risked encoding the physical outcome in the input. A **dwell-age** "
         "feature is added \u2014 minutes since the last action-level change, capped at 3 \u2014"),
        ("Physical quantities are reported as outcomes. A reward or constraint statement that "
         "implies enforced voltage or thermal safety would therefore misdescribe this "
         "implementation.",
         "Physical quantities are reported as outcomes; a reward or constraint statement "
         "implying enforced voltage or thermal safety would misdescribe this implementation."),
        ("Section 5.2 reports a negative result", "Section 5.1 reports a negative result"),
        ("delays as measured quantities (Section 5.5)",
         "delays as measured quantities (Section 5.4)"),
    )

    s4 = top(
        v2, "4. Experimental implementation and evidence",
        ("IEEE 123-bus feeder [9]", "IEEE 123-bus feeder [@kersting1991radial]"),
        ("from explicitly locked taps [6]", "from explicitly locked taps [@opendss123]"),
        ("(corrected evaluation, Sections 5.2–5.6)",
         "(corrected evaluation, Sections 5.2–5.7)"),
    )

    s5 = v2["5. Results"]

    s51 = sub(s5, "5.1 First evaluation")

    s52 = sub(s5, "5.2 Corrected evaluation: what the v1 shortfall was")
    tail = "The third is the one Q1 is about"
    if tail not in s52:
        raise SystemExit("5.2 tail anchor missing")
    s52_tail = tail + s52.split(tail, 1)[1]

    s53 = sub(s5, "5.3 Corrected evaluation: validation split",
              ("Section 5.4", "Section 5.3"))

    s54 = sub(s5, "5.4 Corrected evaluation: held-out split",
              ("Section 5.5", "Section 5.4"))
    # The inference and safety paragraphs duplicate §5.7 and §5.6. Removing them here,
    # rather than repeating them, is where most of the 226-word compression comes from.
    for para in ("**Inference cost.** On the evaluation host",
                 "**Safety, reported under two distinct definitions.**"):
        i = s54.find(para)
        if i < 0:
            raise SystemExit(f"paragraph not found in 5.4: {para[:40]!r}")
        j = s54.find("\n\n", i)
        s54 = s54[:i] + (s54[j + 2:] if j > 0 else "")

    s55 = sub(s5, "5.5 Outage window")
    s56 = sub(s5, "5.6 Transfer robustness",
              ("Frozen selection score, pooled over the 105 runs of each condition "
               "(21 episodes \u00d7 five deterministic faults).",
               "Frozen **composite selection score** \u2014 Equation (5), not the achievement "
               "rate \u2014 pooled over the 105 runs of each condition (21 episodes \u00d7 five "
               "deterministic faults)."))
    s57 = sub(s5, "5.7 Physical outcomes", ("[10]", "[@ansi2020c84]"))
    s58 = sub(s5, "5.8 Edge inference cost",
              ("The per-episode column is the per-control-step column multiplied by the 1,440 "
               "steps in a 24 h episode. The two columns are the same measurement at different "
               "granularity, and only the per-control-step column is comparable with the 60 s "
               "control interval.",
               "Every value is one control step for the whole feeder. A per-episode total would "
               "be this value multiplied by the 1,440 steps of a 24 h episode, which is the same "
               "measurement at a coarser granularity; it is not quoted separately, because only "
               "the per-control-step figure is comparable with the 60 s control interval. The ratio "
               "column therefore divides per-step means, not totals."))
    # The per-episode column re-introduces 1389.2, the retired per-decision figure at a
    # coarser granularity. Dropping the column removes the number without losing a
    # measurement, since the per-step column is the one the argument uses.
    s58 = drop_column(s58, 4)

    data_avail = top(v2, "Data and code availability")

    body = f"""\
{v2['__front__']}

# Abstract

{ABSTRACT}

Keywords: demand response; local fallback; communication outages; linear value approximation; distribution feeder; reproducibility; pre-registration

# 1. Introduction

{S1}

# 2. Related work and the boundary of the claim

{S2}

# 3. System model and controller formulation

{s3}

# 4. Experimental design and evidence base

{s4}

# 5. Results

{S5_1_HEAD}

{s51}

{S5_2_BRIDGE}

{s52_tail}

## 5.2 Validation split: what the repair moved

{s53}

## 5.3 Held-out comparison: the pre-registered test

{s54}

## 5.4 Architecture value on the outage window

{s55}

## 5.5 Transfer across operating points

{s56}

## 5.6 Physical outcomes

{s57}

## 5.7 Inference cost

{s58}

{S6}

{S7}

{S8}

# Data and code availability

{data_avail}
"""

    # Labels for the two runs. These live in blocks inherited from v2, so they are
    # applied once to the assembled text rather than threaded through every sub() call.
    for old, new in LABELS:
        if old not in body:
            raise SystemExit(f"label anchor missing: {old[:60]!r}")
        body = body.replace(old, new)

    for old, new in POLISH:
        hits = body.count(old)
        if hits != 1:
            raise SystemExit(f"polish anchor hits={hits}: {old[:70]!r}")
        body = body.replace(old, new)

    # --- figures (D4: Run 2 figures only) ------------------------------------
    for old, new in FIG_REFS:
        hits = body.count(old)
        if hits != 1:
            raise SystemExit(f"figure ref anchor hits={hits}: {old[:70]!r}")
        body = body.replace(old, new)

    # The Table 2 C0 row is found by the value it carries, not by copying the
    # column spacing into this file; the §5.3 table has its own C0 row.
    tbl = [l for l in body.split("\n")
           if l.startswith("| C0 uncontrolled") and "0.1105" in l]
    if len(tbl) != 1:
        raise SystemExit(f"F2 Table 2 C0 row not unique: {len(tbl)} matches")
    c0_row = tbl[0]

    for n, anchor in FIG_INSERTS:
        anchor = c0_row if anchor == "| C0 uncontrolled" else anchor
        hits = body.count(anchor)
        if hits != 1:
            raise SystemExit(f"figure insert anchor {n} hits={hits}: {anchor[:70]!r}")
        name, caption = FIGURES[n - 1]
        block = (f'<img src="figures/{name}" style="width:6.3in;" />\n\n'
                 f"*Figure {n}. {caption}*")
        body = body.replace(anchor, anchor + "\n\n" + block)

    # embeds must appear in numbering order and match the figure script's output
    pos = [body.index(f'<img src="figures/{name}"') for name, _ in FIGURES]
    if pos != sorted(pos):
        raise SystemExit("figures embedded out of numbering order")
    reg = json.loads((ROOT / "figures" / "v2_figure_sources.json")
                     .read_text(encoding="utf-8"))
    if sorted(n for n, _ in FIGURES) != sorted(reg["figures"]):
        raise SystemExit("figure set mismatch: build vs figures/v2_figure_sources.json")

    # stage the PNGs beside the manuscript (check all first, then copy)
    pkg = ROOT / "paper_package" / "figures"
    pkg.mkdir(exist_ok=True)
    missing = [n for n, _ in FIGURES if not (ROOT / "figures" / n).is_file()]
    if missing:
        raise SystemExit(f"figure file(s) missing in figures/: {missing}")
    for name, _ in FIGURES:
        shutil.copyfile(ROOT / "figures" / name, pkg / name)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8", newline="\n")

    words = len(re.sub(r"[#*`]|<[^>]+>", " ", body).split())
    markers = re.findall(r"\[(@[^\]]+)\]", body)
    keys = {k.strip() for m in markers for k in m.split(";")}
    print(f"{len(body)} bytes, {words} words -> {OUT}")
    print(f"{len(markers)} citation markers, {len(keys)} distinct keys")
    print(f"{len(FIGURES)} Run 2 figures embedded and staged -> paper_package/figures/")


if __name__ == "__main__":
    main()
