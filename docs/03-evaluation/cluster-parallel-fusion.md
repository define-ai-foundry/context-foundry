# Cluster-Parallel Fusion

> **What clustering detections geographically before fusion would gain, with one fusion instance per cluster, measured against one instance fusing every site and against each site fused alone.**

**Status:** proposal, not yet measured. The near pair is still blocked (see [Inputs](#inputs)).

## Summary

* [Multisite Fusion](multisite-fusion.md) showed that one `FusionEngine` fusing joensuu and alakurtti loses no accuracy. The cost is shared, though: every live track costs about 0.1 ms per event at both sites. So does the late gate: alakurtti lost 19% and 42% of its events when its clock ran 3 s and 5 s behind.
* This page measures what splitting the work buys. It clusters the sensors or detections and runs one engine per cluster, each in its own process, with its own frame, watermark, caps and track-id namespace. The instances' outputs are merged into one picture.
* Three clustering methods are compared: by deployment, by sensor position and by detection position.
* A synthetic boundary case tests what happens where clusters overlap or an object crosses from one cluster into another.
* The result is a recommendation between three options:
  1. engine-level partitioning in one process
  2. a platform-level region router feeding several `fusion` stages
  3. not worth building

---

## Why Measure This

* **The issue it informs:** the platform's fusion-throughput issue (`rain-iam-platform`, `docs/pipeline-platform/ISSUES.md`, I-394.2). Each association event predicts every live track, and each unmatched track coasts on every event, whichever cell it is in.
  * context-foundry#19 brought fusion to about 10 ms per detection, 40 to 48 detections a second fused on the exercise.
  * A single sensor sent bursts of 72 a second.
  * The engine runs on one Python thread, so adding replicas doesn't help.
* **The exercise's peak volume:** the busiest minute of 2026-10-27 carried 329 SAPIENT and 236 CoT records a second, in the senders' catch-up bursts (`docs/pipelines-exercise-runbook.md`, section 9).
* **Late drops on the exercise:** most came from individual senders lagging or catching up in bursts, not from one whole site running behind another (I-394.7).
  * A per-cluster watermark only removes the second kind.
  * The recommendation has to weigh that.

---

## Questions

1. **Accuracy:** with one instance per cluster, is each site's picture the same as in Multisite Fusion's E1, and the same as the site fused alone?
2. **Late gate:** are the alakurtti drops from Multisite Fusion's E3 gone?
3. **Cost:** under Multisite Fusion's E4 clutter at joensuu, does alakurtti's per-event time return to its alone value?
4. **Throughput:** how many detections a second does each instance sustain, and how many all instances together on N cores, measured against I-394.2's rates and the exercise's peak?
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

Method A is the baseline. It is how sensors already arrive (one sensor-network config per deployment), and it needs no geometry.

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
* alakurtti's own sensors are up to 274 km apart, so any threshold under that splits alakurtti.
* hamina_kotka's radar has a 400 km range and sits in Jyväskylä, 210 km from joensuu.
* Linking sensors whose footprints overlap therefore puts all four sites in one cluster.
* So sensor position alone doesn't recover the deployments. Method B is reported as this table, plus the far pair at the thresholds that split alakurtti.

### Process model

* **One OS process per cluster,** started with `multiprocessing` in spawn mode. Threads won't work, because the sensor registry and the ENU origin are module-global.
* **Each instance has its own:**
  * frame: the first detection it fuses, as in the pipeline stage today. Multisite Fusion's E2 found no need to re-anchor.
  * watermark: `FusionEngine` keeps one per instance
  * caps: `max_live_tracks` and `max_event_detections` belong to the tracker
  * track-id namespace: the tracker's `id_namespace`, set to the cluster's name
* **Each instance** runs Multisite Fusion's fuse loop (`FusionEngine.process` per event, in delivery order) over its own cluster's events only.
* **Every clustered condition is timed two ways:**
  * **parallel:** the instances run at once in separate processes. Its wall-clock estimates option 2, a platform-level split.
  * **sequential:** the same instances one after another in one process. Their summed time estimates option 1, one engine that predicts and coasts only the tracks of the cluster an event belongs to.

### Merge

* The instances' fused CoT is concatenated into one picture and scored per site with `fusion_benchmark.score_tracks`, as in Multisite Fusion.
* **Duplicate:** two tracks from different instances, matched to the same swarm, whose lifetimes overlap.
* **Gap:** a stretch longer than the coast window in which a swarm has no track, although it is inside some sensor's range.

### Boundary case B1 (synthetic)

* **Setup:** two groups of radars whose footprints overlap.
  * One target flies from the first group's area into the second's.
  * A second target stays in the overlap.
* **Handover options compared:**
  * **none:** the instances' outputs are concatenated as they are
  * **stitch:** a track that ends in one instance is linked to one that starts in the other within a time window Δt and a distance d (both parameters, reported with the results)
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
| Throughput | detections/s per instance: its detections divided by its fuse-loop seconds | `time.perf_counter` |
| Throughput | aggregate detections/s: all detections divided by the parallel wall-clock, on 2 cores and on one core per cluster | `time.perf_counter` |
| Clustering cost | µs per detection to route (method C); time to recluster when a sensor registers (method B) | `time.perf_counter` |

Throughput is compared with I-394.2's 40 to 48 detections a second fused and 72 a second from one sensor, and with the exercise's busiest minute (329 SAPIENT records a second). Rates are compared within one machine. The machine is stated.

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

*Not measured yet.* This section will give the commits, seeds, machine and commands, and the tables for P0, C1, C2 and C3.

---

## Recommendation

*Written once the results exist.* It will choose and justify one of:

1. **Engine-level partitioning:** one process, with tracks indexed by region so that only nearby tracks are predicted or coasted. Estimated by the sequential timing.
2. **Platform-level split:** a region-router stage feeding several `fusion` stages. Estimated by the parallel timing. It touches:
   * pipeline-stages
   * the pipeline-catalog descriptor
   * pipeline composition in rain-data-pipelines
   * the cell attribution of I-336.1
3. **Not worth it**, with the numbers.

It will:
* name the follow-up issues and repos
* say how much of the exercise's late-gate loss a per-cluster watermark would have removed, given that I-394.7's losses came from individual senders
* propose the measured rates as a line for `docs/pipelines-fusion-runbook.md` in rain-iam-platform, as a separate change

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
* Recorded exercise data, until records can be mapped to cells (I-336.1).
