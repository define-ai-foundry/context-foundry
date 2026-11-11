# Cluster-Parallel Fusion

> **What clustering detections geographically before fusion would gain, with one fusion instance per cluster, measured against one instance fusing every site and against each site fused alone.**

**Status:** measured on the far pair and the synthetic boundary case. The near pair is still blocked (see [Inputs](#inputs)).

## Summary

* [Multisite Fusion](multisite-fusion.md) showed that one `FusionEngine` fusing joensuu and alakurtti loses no accuracy. The cost is shared, though: every live track costs about 0.1 ms per event at both sites. So does the late gate: alakurtti lost 19% and 42% of its events when its clock ran 3 s and 5 s behind.
* This page measures what splitting the work buys. It clusters the sensors or detections and runs one engine per cluster, each in its own process, with its own frame, watermark, caps and track-id namespace. The instances' outputs are merged into one picture.
* Three clustering methods are compared: by deployment, by sensor position and by detection position.
* A synthetic boundary case tests what happens where clusters overlap or an object crosses from one cluster into another.
* The result is a recommendation between three options:
  1. engine-level partitioning in one process
  2. a platform-level region router feeding several `fusion` stages
  3. not worth building
* **Result:** clustering by deployment gives each site exactly what it gets fused alone. It removes four cross-site effects of a shared engine: late-gate drops, cost per event, cap refusals, and a newly found one in track confirmation. It does not speed up the busiest deployment under load. A grid by detection position splits tracks at cell edges. [Recommendation](#recommendation): do not cluster fusion. Method A needs a sensor-to-deployment mapping the platform does not provide and does not expect, and sensor or detection position (methods B and C) does not recover deployments or splits tracks. Fix the engine instead: the cost per live track ([context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33)) first, which also removes most of the exercise's late drops, then the confirmation window ([context-foundry#32](https://github.com/define-ai-foundry/context-foundry/issues/32)).

---

## Why Measure This

* **The issue it informs:** the platform's fusion-throughput issue (`rain-iam-platform`, `docs/pipeline-platform/ISSUES.md`, I-394.2). Each association event predicts every live track, and each unmatched track coasts on every event, whichever cell it is in.
  * context-foundry#19 made fusion faster, but on the exercise it still fused fewer detections a second than single sensors sent in bursts (I-394.2).
  * The engine runs on one Python thread, so adding replicas doesn't help.
* **The exercise's peak volume:** the busiest minute came in the senders' catch-up bursts (I-336.2).
* **Late drops on the exercise:** most came from the one `fusion` falling behind its input, so its merge released records early (I-394.3); the rest came from individual senders lagging, running on an offset clock or catching up in bursts (I-394.7, [pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58)).
  * A per-cluster watermark only removes drops caused by one site running behind another, which is neither of these.
  * The recommendation has to weigh that.

---

## Questions

1. **Accuracy:** with one instance per cluster, is each site's picture the same as in Multisite Fusion's E1, and the same as the site fused alone?
2. **Late gate:** are the alakurtti drops from Multisite Fusion's E3 gone?
3. **Cost:** under Multisite Fusion's E4 clutter at joensuu, does alakurtti's per-event time return to its alone value?
4. **Throughput:** how many detections a second does each instance sustain, and how many all instances together on N cores, in proportion to one engine?
5. **Clustering:** what does each clustering method cost, and how does each behave when a sensor registers?
6. **Boundaries:** where clusters overlap or an object crosses between them, how many duplicate tracks and gaps result, and does a stitch step remove them?
7. **Recommendation:** engine-level partitioning, a platform-level split, or not worth it?

---

## Inputs

| Input | Value |
|---|---|
| Engine | context-foundry at the commit the results are produced on, stated with the results |
| Sites | joensuu (9 sensors) and alakurtti (5 sensors), as in Multisite Fusion |
| Scenarios | scenario-foundry `joensuu` and `alakurtti`, `--seed 42`, alakurtti shifted by -900 s as in Multisite Fusion |
| Sensor configs | `config/sensors/*.json`. All four sites' configs are used for the geometry of method B. |
| Ground truth | each scenario's `threat_profiles`, positioned with `Swarm` from `dev/fusion_benchmark.py` |
| Clutter | Multisite Fusion's seeded clutter generator, seed 42 |

**Not available yet:**

* **The near pair (porvoo_oil_refinery and hamina_kotka, about 85 km apart) and all four sites:** scenario-foundry still has no scenario configs for either site, so neither generates.
  * This is the pair where a real geographic boundary would have to be chosen.
  * The rows stay empty until scenario configs exist. No scenario is made up for them here.
* The boundary case below is synthetic and generated in the harness for that reason. It is labelled as synthetic wherever it is reported.

---

## Method

### Clustering

| Method | How a detection reaches a cluster | Cost per detection |
|---|---|---|
| **A. By deployment** | each sensor-network config is one cluster; a detection goes to its sensor's cluster by `nodeId` | a dictionary lookup |
| **B. By sensor position** | single-linkage over sensor positions with a distance threshold; a detection goes to its sensor's cluster | a dictionary lookup, plus reclustering when a sensor registers |
| **C. By detection position** | a fixed lat/lon grid (cell size a parameter, 100 km by default); a detection goes to the cell it lies in | a cell computation; measured |

Method A is the baseline. In the CLI and this harness, sensors arrive as one sensor-network config per deployment, so it needs no geometry. The platform has no such config and does not provide a sensor-to-deployment mapping, and none is expected: see [Recommendation](#recommendation).

**Method B on sensor geometry alone.** Single-linkage over all four sites' sensor configs gives the clusters below. No scenarios are needed for this.

| Linkage threshold | Clusters |
|---|---:|
| 50 km | 7 |
| 100 km | 5 |
| 150 km | 5 |
| 250 km | 2 |
| 300 km | 2 |
| footprints overlap (distance < range of one + range of the other) | 1 |

* No threshold gives back the four sites.
* alakurtti's sensors are up to 274 km apart, but single linkage chains through its middle sensors, so it stays whole at 250 km and splits at 150 km and under.
* hamina_kotka's radar has a 400 km range and sits in Jyväskylä, 210 km from joensuu.
* Linking sensors whose footprints overlap therefore puts all four sites in one cluster.
* So sensor position alone doesn't recover the deployments. Method B is reported as this table, plus the far pair at the thresholds that split alakurtti.

### Process model

* **One OS process per cluster:** a fresh Python subprocess, as in Multisite Fusion. Threads won't work, because the sensor registry and the ENU origin are module-global.
* **Each instance has its own:**
  * frame: the first detection it fuses, as in the pipeline stage today. Multisite Fusion's E2 found no need to re-anchor.
  * watermark: `FusionEngine` keeps one per instance
  * caps: `max_live_tracks` and `max_event_detections` belong to the tracker
  * track-id namespace: the tracker's `id_namespace`, set to the cluster's name
* **Each instance** runs Multisite Fusion's fuse loop (`FusionEngine.process` per event, in delivery order) over its own cluster's events only.
* **Every clustered condition is timed two ways:**
  * **parallel:** the instances run at once, each in its own process. It shows what they cost each other on one machine.
  * **sequential:** the same instances one after another, each in a fresh process, so none competes for a core.
* **From those times:**
  * **option 1**, one engine that predicts and coasts only the tracks of the cluster an event belongs to, is estimated by the instances' summed time on one core
  * **option 2**, a platform-level split, by the longest instance's time, with one core per cluster, or by the instances packed onto 2 cores
* Times are the seconds spent in `FusionEngine.process`. They leave out reading the replay and the harness's own bookkeeping per event.

### Merge

* The instances' fused CoT is concatenated into one picture and scored per site with `fusion_benchmark.score_tracks`, as in Multisite Fusion.
* **Duplicate:** two tracks from different instances, matched to the same swarm, whose lifetimes overlap.
* **Gap:** a stretch at least as long as the coast window in which a swarm has no track, although it is inside some sensor's range.

### Boundary case B1 (synthetic)

* **Setup:** two groups of radars whose footprints overlap.
  * One target flies from the first group's area into the second's.
  * A second target stays in the overlap.
* **Handover options compared:**
  * **none:** the instances' outputs are concatenated as they are
  * **stitch:** a track in one instance is continued by a track in another when:
    * the second starts after the first, and no later than Δt after the first ends
    * it stays within d of the first: of the first's last position if it starts after the first ends, otherwise as a median over the time both exist
  * The stitch therefore joins both handovers and overlapping duplicates. Δt is the coast window (45 s) and d is the scoring's false-track distance (2 km).
  * Each track continues at most one other, and is continued by at most one, the closest first.
* **Run under:** method A (each group a cluster) and method C (a cell boundary between the groups).
* **Reported:** duplicates, gap length, and what the stitch removes and costs.

### Running

As in Multisite Fusion:
* the harness builds the engine as `context_foundry.cli` does and calls `FusionEngine.process` itself
* each run is a fresh subprocess
* each condition runs three times
* wall-clock figures are medians, reported with the machine's CPU and Python version

---

## Metrics

Each site is reported in three runs: the single instance, the clustered run, and the site fused alone.

| Group | Metric | Source |
|---|---|---|
| Accuracy | tracks, peak concurrent, false tracks, estimated objects, median error | `fusion_benchmark.py` scoring, split by site |
| Merge | duplicate tracks, gaps | the merge checks above |
| Losses | events dropped as late, tracks refused at `max_live_tracks`, detections shed at `max_event_detections` | as in Multisite Fusion |
| Cost | ms per event, by site; own-site and other-site ms per live track | as in Multisite Fusion (`fit_cost`) |
| Throughput | detections/s per instance: the detections it was offered, divided by its seconds in `FusionEngine.process` | `time.perf_counter` |
| Throughput | aggregate detections/s: all detections divided by the instances' seconds summed (one core), packed onto 2 cores, or the longest instance's (one core per cluster) | `time.perf_counter` |
| Clustering cost | µs per detection to route (method C); time to recluster when a sensor registers (method B) | `time.perf_counter` |

Rates are compared within one machine, as proportions. The machine is stated. They are not set against the platform's rates (I-394.2) or the exercise's peak (I-336.2), which were measured on other hardware.

---

## Experiments

### P0: Multisite Fusion's open points, settled first

The gains below depend on these two points.

* **Cap timing:** in E4 at 5/s, the cap refused no alakurtti track. Did alakurtti start all its tracks before joensuu filled the cap? If so, a condition is added in which alakurtti starts tracks while joensuu holds the cap. That shows whether the cap refuses tracks across sites, and whether clustering removes that.
* **Clutter confirming into tracks:** a short read of how a fresh track is gated and confirmed. The engine is not changed; the answer is recorded as a finding.

### C1: single against clustered against alone

Multisite Fusion's conditions, for like-for-like numbers:

| Condition | From Multisite Fusion | Run as |
|---|---|---|
| combined | E1 | single, clustered by A, alone; also clustered by C |
| alakurtti 3 s and 5 s behind | E3 | single, clustered by A, alone |
| clutter at joensuu 1, 2 and 5/s | E4 | single, clustered by A, alone; at 5/s also clustered by C |

**Expectations for the far pair:**
* Clustering by A leaves E1's accuracy unchanged.
* It removes E3's alakurtti drops.
* It brings alakurtti's ms/event under E4 back to its alone value.

Under method A, each far-pair cluster gets the same events and the same first-detection frame as its site fused alone, so it must score exactly the same. A difference means a harness bug.

### C2: clustering cost and stability

* **Method B:** the geometry table above. The far pair at the thresholds that split alakurtti. The time to recluster when one sensor registers.
  * When a new sensor joins two clusters, their instances' tracks would have to be merged into one engine. That case is described, not built.
* **Method C:** routing cost per detection, and how many tracks are split or duplicated at cell edges in C1.

### C3: boundary case B1

The synthetic case above, under methods A and C, with and without the stitch.

---

## Results

**Run:**

| Item | Value |
|---|---|
| context-foundry | `ce05775` (the harness commit on this branch) |
| scenario-foundry | `32a7de6` |
| Seeds | scenarios 42, clutter 42 |
| Machine | AMD Ryzen 5 5500U (6 cores, 12 threads), Python 3.12, Linux x86_64 |
| Repeats | 3 per condition. Every condition gave the same tracks, scores, drops and refusals in all three; times are medians. |
| Command | `python dev/cluster_fusion_benchmark.py experiments --scenario-foundry <scenario-foundry checkout> --repeats 3 --p0-delays 600 1200 --save <out>.json` |

**Checks before reading:**
* The refactored harness gave the same tracks, false tracks, error, drops and refusals for Multisite Fusion's E1 and E3 at 3 s as before the refactor.
* Under method A, each far-pair cluster scored exactly as its site fused alone in E1, E3 at 3 s and E3 at 5 s, parallel and sequential: the same scores, event counts, drops, refusals, peak live tracks and gaps.
* The own-site and other-site cost fit was not refitted. Multisite Fusion's fit stands, and the ms per event below shows the same effect.

### P0: Multisite Fusion's open points

**Cap timing.** At 5/s clutter the shared cap is first full at 02:47:10, after alakurtti has started its tracks, which is why E4 refused none of them. With alakurtti's launches moved later:

| alakurtti launches | single: alakurtti tracks refused | single: V3_ARCTIC_HELI_ASSAULT gap | clustered by A: refused | clustered by A: gap |
|---|---:|---:|---:|---:|
| as in E4 | 0 | 45 s | 0 | 45 s |
| 600 s later | 3 | 75 s | 0 | 45 s |
| 1200 s later | 0 | 45 s | 0 | 45 s |

* So the cap does refuse tracks across sites: joensuu's clutter cost alakurtti a 30 s longer gap on one swarm. Whether it happens depends on when the cap is full, not on how late alakurtti starts; at 1200 s the cap had room at the right moments.
* Clustering by A removes it: alakurtti's own cap is never full.

**Clutter confirming into tracks.** This is the confirmation rule as documented at `tracker.py:253-275`: a track confirms on 3 reports within 30 s, and that comment records 65 to 74 ghost tracks from 300 random false reports in two minutes. E4's clutter is 1 to 5 random reports a second, so it confirms into tracks as designed. The harness is not at fault.

**A new cross-site effect: the confirmation window.** E1 showed no difference in tracks, false tracks or error, but the single engine left gaps of 252, 192 and 162 s at the start of three joensuu swarms (W1_DECOY, W2_KMS_PLANT, W2_KMS_RAILWAY) that joensuu fused alone does not have. The cause:
* The tracker checks whether a tentative track has passed its 30 s window only when an event arrives (`_prune_stale_tracks`, `tracker.py:1395-1400`).
* At the start only KOLI-01 sees these swarms, every 18 s, so the third report comes 36 s after the first.
* Fused alone, no event arrives between 30 s and 36 s, so the third report still confirms the track.
* In the single engine, alakurtti's events at 02:45:30.04 and 02:45:34.96 run the check, the tentative track is dropped, and it starts again. That repeats every cycle until a second sensor sees the swarm.
* So whether one site's tracks are shown depends on how busy the other site is. Clustering by A removes it, because each instance only sees its own site's events.

### C1: single against clustered against alone

**Accuracy, E1 (combined).** Tracks, false tracks, median error, and gaps at least as long as the 45 s coast window:

| Site | single | clustered by A | clustered by C | alone |
|---|---|---|---|---|
| joensuu | 6 tracks, 0 false, 21 m; gaps 252, 192, 162, 83 s | 6, 0, 21 m; gaps 48, 54, 48 s | 7, 0, 21 m; gaps 48, 54, 48 + 53 s | 6, 0, 21 m; gaps 48, 54, 48 s |
| alakurtti | 5, 0, 25 m; gaps 245, 74, 45 s | 5, 0, 25 m; gaps 245, 74, 45 s | 8 (7 after the stitch), 0, 27 m; gaps 245, 60, 45 s | 5, 0, 25 m; gaps 245, 74, 45 s |

* Method C splits tracks at cell edges: alakurtti's swarms cross cells, giving 3 more tracks, of which the stitch joins 1. joensuu's W2_KMS_AIRPORT gains a second gap.

**Late gate, E3.** alakurtti's events dropped, of 577:

| alakurtti behind | single | clustered by A | alone |
|---|---|---|---|
| 3 s | 112 (19%); 4 tracks, 27 m | 0; 5 tracks, 25 m | 0; 5 tracks, 25 m |
| 5 s | 240 (42%); 5 tracks, 26 m | 0; 5 tracks, 25 m | 0; 5 tracks, 25 m |

**Cost, E4.** alakurtti's median ms per event with clutter at joensuu (alone: 1.34 ms):

| Clutter at joensuu | single | A, sequential | A, parallel |
|---|---:|---:|---:|
| 1/s | 3.49 | 1.18 | 1.70 |
| 2/s | 5.61 | 1.40 | 1.67 |
| 5/s | 11.64 | 1.42 | 2.10 |

* Clustered, alakurtti's time per event is back at its alone value; in parallel it pays somewhat for sharing the machine.
* joensuu at 5/s: single 1656 tracks, 1601 false, 147 refused; A 1628, 1571 false, 118 refused; C (17 cells) 1670, 1604 false, 0 refused. Each grid cell has its own cap, so none fills; more clutter then confirms.
* joensuu's own swarms at 5/s, sequential: each of its five is uncovered for 413 to 965 s on one engine and under A alike (W1_DECOY 413 s both, W2_KMS_PLANT 632 s both, W2_KMS_AIRPORT 948 s single and 965 s A), with peak live tracks 119 and 120. alakurtti's gaps stay at E1's.

### Throughput

Detections a second through `FusionEngine.process`, from the sequential runs (each instance in its own process, none sharing a core):

| Condition | single | A: one core | A: core per cluster | C: one core | C: core per cluster | Slowest instance |
|---|---:|---:|---:|---:|---:|---|
| E1 combined | 897 | 1196 | 2389 | 1376 | 2633 | alakurtti 983 (A) |
| clutter 1/s | 373 | 365 | 399 | | | joensuu 320 (A) |
| clutter 2/s | 193 | 186 | 191 | | | joensuu 166 (A) |
| clutter 5/s | 80 | 84 | 84 | 100 | 101 | joensuu 79 (A), cell r69c15 85 (C) |

* With 2 clusters, "2 cores" equals "core per cluster".
* **Without clutter**, splitting the far pair raises throughput by a third on one core, because each event predicts fewer tracks, and gives 2.7 times on two.
* **Under load**, the busiest cluster sets the rate. At 5/s clutter, joensuu's instance alone is as slow as the single engine (about 80 a second). 100 km cells lift the total to about 100, but the busiest cell still does only 85.
* **Parallel runs:** two instances at once ran as fast as one after the other. 17 instances at once (C at 5/s) on 12 threads and about 4 GB of memory were each slower, so the core-per-cluster figure for C assumes cores that this machine doesn't have.
* **Noise:** runs of 1 to 2 s (no clutter) vary by up to about 30% between modes; runs of 160 s or more (5/s) vary by a few percent.
* **Against the platform's rates:** these are laptop rates, not the pipeline's hardware, so they show proportions, not headroom. They moved about 3.5 times between two machines (E1 single 897 a second here, 3,162 to 3,541 on another), so they are not set against the platform's rates (I-394.2) or the exercise's peak (I-336.2). What carries over: under clutter the busiest cluster sets the rate, so clustering raises the total only as far as the load spreads over clusters of similar weight, and in E4 it doesn't.

### C2: clustering cost and stability

* **Method B:** the geometry table under [Clustering](#clustering) holds. No threshold gives back the deployments: at 150 km and under alakurtti and hamina_kotka split (single linkage chains through their middle sensors, so neither splits at its 274 km and 204 km spans), and from 250 km up sites merge. Reclustering all 27 sensors when one registers takes 0.3 ms.
* **Routing:** 0.6 to 17 µs a detection for A and 1.4 to 21 µs for C, against 1 to 12 ms per event to fuse: 2% or less in every condition.
* **Method C at cell edges:** see E1 above: 3 extra alakurtti tracks and one extra joensuu gap at 100 km cells.

### C3: boundary case B1 (synthetic)

Two targets, two radar groups whose footprints overlap. Tracks are counted by the group nearest their first position.

| Run | Tracks | Duplicates | Stitch links | After the stitch | Median error | Gaps |
|---|---:|---:|---:|---:|---:|---|
| single | 2 | 0 | 0 | 2 tracks, 0 duplicates | 23 m | none |
| clustered by A | 4 | 2 | 2 | 2 tracks, 0 duplicates | 23 to 25 m | none |
| clustered by C | 4 | 1 | 2 | 2 tracks, 0 duplicates | 31 to 32 m | none |

* Clustered, the overlap target is tracked by both instances, and the crossing target gets a second track when it enters the other group's area. The stitch joins both kinds, so the merged picture has the single engine's 2 tracks again.
* Under C, the cell edge splits each target's detections between instances, so each instance fuses fewer reports per target and the error rises by about 9 m. A doesn't split detections, so its error stays at the single engine's.

---

## Recommendation

**Do not cluster fusion; fix the engine (option 3).** No clustering method is available on the platform: method A needs a sensor-to-deployment mapping the platform does not provide, method B does not recover the deployments, and method C splits tracks at cell edges. The cross-site effects that matter have fixes in the engine that serve every site.

**Why method A is not available.** The `fusion` stage has no sensor config to route by: its sensor registry is learned from its input (rain-iam-platform `DECISIONS.md`, Fusion). Which cell or deployment a sensor belongs to is not known, since every cell sends to the same two listener ports, and the platform does not provide that mapping: the organisers give no table of sensors per cell (I-336.1). `DECISIONS.md` Exercise pipeline also runs every cell on one pipeline with one `fusion`, and rejected a pipeline per cell because it splits the fused picture. Method A holds only where sensors arrive as one sensor-network config per deployment, as in the CLI and this harness.

**What method A would have given.** On the far pair, clustering by deployment is exact: each instance gives what its site gives fused alone, in every condition. It removes four ways one site affects the other in a shared engine, each of which is small on the exercise or fixed in the engine instead:

| Cross-site effect | single | clustered by A | Instead |
|---|---|---|---|
| Late gate (E3) | 19% and 42% of alakurtti's events dropped | none | On the exercise, most late drops came from the one engine falling behind its input, which a cheaper event removes for every site ([context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33)); the rest came from individual senders ([pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58)). |
| Cost per event (E4, 5/s) | alakurtti 11.6 ms | 1.4 ms | A cheaper event in the engine ([context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33)) removes it for every site and also serves throughput (I-394.2), which clustering does not. |
| Track cap (P0, 600 s) | 3 alakurtti tracks refused, a 30 s longer gap | none | Measured here only with synthetic clutter at the other site; a replay of the exercise's recorded data also reached the shared cap (review of this page, [context-foundry#31](https://github.com/define-ai-foundry/context-foundry/pull/31); not rerun here). A site's own clutter still fills its cap under A (E4: 118 refused). |
| Confirmation window (E1) | three joensuu swarms first shown 108 to 216 s later than alone | none | An engine defect, fixed in the engine for every site ([context-foundry#32](https://github.com/define-ai-foundry/context-foundry/issues/32)). |

**Why not method C.** Cells split tracks and raise the error where a target meets a cell edge (B1: 23 m to 32 m), and they fragment the far pair's tracks (alakurtti 5 to 8). The only gain is a cap per cell, which lets more clutter confirm into tracks. The busiest cell is no faster than the busiest deployment.

**Why not method B.** Sensor position does not recover the deployments at any threshold.

**Why not option 1.** Engine-level partitioning needs the same partition key: by deployment it needs the mapping, by position it inherits B's or C's faults. Estimated by the summed time on one core, it gains a third without clutter and 5% at 5/s clutter, and the late gate, the cap and the confirmation window would each need a per-partition version.

**What clustering would not have fixed.**
* **Throughput (I-394.2):** under load the busiest deployment bounds the rate, and clustering does not make it faster.
* **A site's own clutter (E4):** at 5/s, joensuu's swarms go uncovered for 413 to 965 s on one engine and under A alike.
* **The exercise's late drops:** most came from fusion falling behind its input, the rest from individual senders lagging, running on an offset clock or catching up in bursts ([pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58)). Clustering changes the first only as far as it speeds up the busiest deployment, and the second not at all.

**Revisit only if,** after the engine fixes ([context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33)), one `fusion` on the iamrain node cannot keep up with the exercise's peak rate (I-336.2). Splitting is then the way to use more than one core, since the engine runs on one Python thread. With no sensor-to-deployment mapping it needs a partition key by position: a grid with the stitch, which removed border duplicates in B1 (synthetic) but splits tracks at cell edges. `dev/cluster_fusion_benchmark.py` measures one engine against several.

**Problems found and where they are fixed.** Each issue holds the design of its fix; this table holds the order.

| Order | Problem | Found in | Where | Direction |
|---|---|---|---|---|
| 1 | Per-event cost grows with every live track (the coast step of each unmatched track); on the exercise, fusion falling behind caused most late drops | E4, Throughput, [pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58) | [context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33), the engine side of I-394.2 | Profile E4 at 5/s, then cut the per-event work for tracks no detection can reach. |
| 2 | A tentative track is confirmed or dropped depending on unrelated events | E1 | [context-foundry#32](https://github.com/define-ai-foundry/context-foundry/issues/32) | Confirmation depends only on the track's own reports; size the window against the slowest sensor's revisit and measure ghost tracks under clutter. |
| 3 | A site's own clutter leaves its swarms uncovered for 413 to 965 s and fills the track cap | E4 | [context-foundry#35](https://github.com/define-ai-foundry/context-foundry/issues/35) | Decided in that issue, after [context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33)'s profile. |
| 4 | Late drops from individual senders (lagging, offset clock, catch-up bursts) | I-394.7 | [pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58) | Proposed in that issue: an event-time hold of a few seconds in the stage's merge and a clock offset per sender; old bursts stay known losses. |
| — | Cross-site late gate | E3 | none | Covered by [context-foundry#33](https://github.com/define-ai-foundry/context-foundry/issues/33) and [pipeline-stages#58](https://github.com/define-ai-foundry/pipeline-stages/issues/58); no change of its own. |
| — | The near pair (porvoo_oil_refinery, hamina_kotka) is unmeasured | [Inputs](#inputs) | none | Needs scenario configs; only relevant if clustering is revisited. |
| — | Runbook line | Throughput | rain-iam-platform, separate change | Below. |

**Proposed runbook line** (`rain-iam-platform`, `docs/pipelines-fusion-runbook.md`):

> Splitting fusion by deployment raised the rate on the far pair without clutter by about a third on one core and about 2.7 times with a core per deployment; with 5 false reports a second at one site, the busiest deployment's engine bounds the rate and splitting gained about 5%. It would remove cross-site late drops, cost and cap refusals but not speed up the busiest deployment; it is not planned, since the platform has no sensor-to-deployment mapping and none is expected. Absolute rates moved about 3.5 times between two laptops, so headroom needs a measurement on the iamrain node. Source: context-foundry `docs/03-evaluation/cluster-parallel-fusion.md`.

---

## Deliverables

| Item | Location |
|---|---|
| Harness refactor | `dev/multisite_benchmark.py`: `run_condition` is split into building the events (load, align, clutter, lag), fusing one event list with one engine, and scoring. Its output is unchanged. |
| Prototype | `dev/cluster_fusion_benchmark.py`, which provides: <ul><li>clustering methods A, B and C</li><li>routing</li><li>the parallel and sequential runners</li><li>the merge and its duplicate and gap checks</li><li>the B1 generator and the stitch</li><li>an experiments CLI with `--save`</li></ul> It reuses the helpers of `multisite_benchmark.py` and `fusion_benchmark.py` rather than copying them. |
| Tests | `tests/test_dev_cluster_fusion_benchmark.py`, written before the prototype. It covers: <ul><li>single-linkage clusters</li><li>grid routing, including detections on cell edges</li><li>distinct track ids per namespace</li><li>the duplicate and gap checks</li><li>the seeded B1 generator</li><li>the stitch linking and declining to link</li></ul> |
| Results | the tables on this page, with the commits, seeds and commands. Raw JSON stays outside the repository. |

**Checks before results are read:**
* After the refactor, Multisite Fusion's E1 and its E3 at 3 s give the same tracks, false tracks, error, drops and refusals as before.
* Under method A, each far-pair cluster scores exactly as its site fused alone.

---

## Out of Scope

* The near pair and the four-site run, until scenario configs exist for porvoo_oil_refinery and hamina_kotka.
* Production code: no change to `src/`, to the pipeline-stages `fusion` stage or to any pipeline.
* Recorded exercise data: its records cannot be mapped to cells (I-336.1).
