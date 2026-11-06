# Multisite Fusion

> **How one fusion instance performs when it fuses two unrelated, geographically distant sites at once, measured against each site fused alone.**

**Status:** measured for the far pair (joensuu and alakurtti). The near pair is still blocked. Results are under [Results](#results).

## Key Takeaways

* **Accuracy doesn't suffer from combining at this load and distance.** Each site's tracks, false tracks and error are the same fused together as fused alone, and error doesn't grow with distance from the ENU frame origin up to 600 km.
* **Cost is shared.** Each live track adds about 0.1 ms to every event, whichever site it is at. With clutter at joensuu, each alakurtti event took 6.5 times as long (1.90 to 12.44 ms).
* **The watermark is shared.** With alakurtti's clock 3 s and 5 s behind, the engine dropped 19% and 42% of alakurtti's events, and alakurtti lost a track. Its clock lagging by 2 s or less dropped nothing.
* So the gain from one fusion instance per site would be independent cost and independent late-event gating, not accuracy. Whether the shared 120-track cap refuses tracks at the other site isn't settled yet (see [Open points](#open-points)).
* One `FusionEngine` keeps one track population, one event-time watermark, one ENU frame and one set of caps for every sensor it hears. This page measures what that costs when the sensors sit at sites hundreds of kilometres apart.
* The measurement replays scenario-foundry's joensuu and alakurtti scenarios, alone and merged, through the engine in-process, and scores the result per site with `dev/fusion_benchmark.py`'s scoring.
* Four experiments separate the effects: combining, frame origin placement, clock offset between sites, and load at one site.
* The "near" pair (porvoo_oil_refinery and hamina_kotka, about 85 km apart) is not measured yet: scenario-foundry has no scenario configs for either site.
* The results are meant to say whether clustering detections by site before fusion, with a fusion instance per cluster, is worth building, and what it should be measured against.

---

## Why Measure This

A pipeline that fuses several deployments in one `fusion` stage puts all of them in one engine. The code gives four reasons to expect sites to interfere with each other:

1. **Every live track costs work on every event.** `SapientAsynchronousTracker.process_async_event` predicts every live track on every event. A track that no detection can reach (`_may_gate`) skips association, but it is still predicted, so its cost grows with the event rate at every site. The tracker and engine run on one thread.
2. **Shared caps.** `max_live_tracks` (120) and `max_event_detections` (300) are per engine. A site that fills the track cap refuses new tracks at every other site.
3. **One watermark.** `FusionEngine.process` drops an event more than `LATE_EVENT_TOLERANCE_SECONDS` (2 s) behind the newest event seen from any sensor. A site whose clock or delivery runs behind another site loses those of its events that land too soon after the other site's.
4. **One ENU frame.** Every input path projects with `config.wgs84_to_enu`, which uses one tangent plane.
   * **Where it is centred:** on the first point it ever projects, not on `primary_anchor_node`. That setting only fills `config.ENU_ORIGIN_*`, which is read by the acoustic bearing model and by the fallback in `enu_to_wgs84`. In a replay, and in the pipeline-stages `fusion` stage (which clears the origin when it starts), that first point is the first detection fused, from whichever site it comes from.
   * **For this pair:** the first point is a joensuu target at 62.185N 30.632E, 95 km from joensuu's anchor. The farthest alakurtti sensor is 790 km from there, where the plane is about 49 km above the ground and its "up" axis is tilted about 7°.

The operational question this informs is the platform's fusion-throughput issue, in which each event predicts every live track across all cells (`rain-iam-platform`, `docs/pipeline-platform/ISSUES.md`, I-394.2).

---

## Questions

1. **Accuracy:** how much worse is each site's picture when it is fused together with the other site, compared with each site alone?
2. **Cost:** how much does each event cost, and does that cost grow with the number of live tracks at the *other* site?
3. **Losses:** does the shared watermark drop one site's events, and do the shared caps refuse its tracks?
4. **Frame:** does position error grow with distance from the ENU frame origin, and does moving the origin change it?
5. **Implication:** which of these effects would per-site fusion instances remove, and how much is there to gain?

---

## Inputs

| Input | Value |
|---|---|
| Engine | context-foundry at the commit the results are produced on, stated with the results |
| Sites | joensuu (9 sensors, anchor `FI-MIL-RAD-KOLI-01`) and alakurtti (5 sensors, anchor `FI-MIL-RAD-LUOSTO-01`) |
| Scenarios | scenario-foundry `joensuu` and `alakurtti`, `--seed 42` (commit stated with the results) |
| Sensor configs | `config/sensors/joensuu.json`, `config/sensors/alakurtti.json` |
| Ground truth | each scenario's `threat_profiles`, positioned with `Swarm` from `dev/fusion_benchmark.py` |

Single-site baseline, from `fusion_benchmark.py joensuu alakurtti` at `24cbc2a`, seed 42:

| scenario | swarms | tracks | peak | false | objects | est. peak | err m |
|---|---:|---:|---:|---:|---:|---:|---:|
| joensuu | 6 | 6 | 6 | 0 | 58 | 58 | 21 |
| alakurtti | 3 | 5 | 2 | 0 | 22 | 18 | 25 |

**Not available yet:**

* **The near pair** (porvoo_oil_refinery and hamina_kotka): `config/sensors/` has their sensor configs, but scenario-foundry has no `config/scenarios/` entry for either, so neither scenario generates. The near-pair rows below stay empty until those configs exist. No scenario is made up for them here.
* **All four sites:** blocked for the same reason. (`fusion_benchmark.py` lists all four by default and so fails on a default run; the new script names its sites explicitly.)

---

## Method

### Combined replay

* The two scenarios start 15 minutes apart (joensuu 02:45Z, alakurtti 03:00Z). Every alakurtti timestamp is shifted by -900 s, in both its messages and its ground truth, so the sites really overlap.
* The two message logs are merged into one stream in event-time order.
* The sensor network is the union of both configs (no node id appears in both), as a shared pipeline has.
* The frame is left to its first point, as in the pipeline stage, except in E2, which places it.
* Ground truth is the union of both scenarios' swarms. Each track is scored against all swarms and attributed to the site of the swarm it follows, so a track at the wrong site would count as false.

### Running the engine

* The harness builds the engine as `context_foundry.cli` does (`SapientAsynchronousTracker`, `PublishPolicy`, `TacticalContextAugmentor`, `FusionEngine`) and calls `FusionEngine.process` itself. It does not go through the CLI, because `JsonSapientSource` sorts every event by its timestamp, and that hides the effect of a site clock that is behind.
* Each run is a fresh subprocess, because the sensor registry and the ENU origin are module-global.
* Each condition runs three times. Wall-clock figures are medians, reported with the machine's CPU and Python version. They are compared within one machine, never across machines.

---

## Metrics

Each run is reported per site, next to the same site fused alone.

| Group | Metric | Source |
|---|---|---|
| Accuracy | tracks, peak concurrent, false tracks, estimated objects, median error | `fusion_benchmark.py` scoring, split by site |
| Cost | ms per event and per detection | `time.perf_counter` around `FusionEngine.process` |
| Cost | live tracks over time, by site | tracker state after each event; a track's site is the site of its nearest sensor |
| Cost | per-event time against own-site and other-site live tracks | least-squares fit over all events |
| Losses | events dropped as late, and how far behind they were | `EventResult.skew` |
| Losses | detections refused a track at `max_live_tracks` | the tracker's refused-track count |
| Losses | detections shed at `max_event_detections` | the tracker's shed-detection count |
| Frame | median position error, binned by distance from the frame origin | per-snapshot error from the scoring |

The refused and shed counts are private tracker attributes. The dev script reads them so that this measurement doesn't change the engine.

---

## Experiments

### E1: alone against combined

Each site is fused alone, then both are fused together.

| Pair | Distance | Status |
|---|---|---|
| joensuu + alakurtti | 485 km between anchors | measured |
| porvoo_oil_refinery + hamina_kotka | about 85 km | blocked: no scenario configs |

### E2: frame origin placement

The combined run (frame at the first detection, a joensuu target) is repeated with the frame origin placed by `config.set_reference_origin` before the replay:

* at `FI-MIL-RAD-KOLI-01` (joensuu's anchor)
* at `FI-MIL-RAD-LUOSTO-01` (alakurtti's anchor)
* at the geodesic midpoint of the two anchors (65.049N, 28.456E)

If error follows distance from the origin rather than the site, the frame is the cause.

### E3: clock offset between sites

alakurtti's timestamps are offset by 0, 1, 2, 3 and 5 s, while events are still delivered in true time order, which is how a sensor with a lagging clock reaches the engine. An event is dropped only when some other event in the last moments has pulled the watermark more than 2 s past its stamp. So the share dropped depends on how closely the two sites' events interleave, not on the offset alone. The experiment records that share for each offset and what it does to alakurtti's picture.

A check before this proposal, with alakurtti 3 s behind, gave:

* through the CLI (merged file, sorted by timestamp): 0 of 1082 events dropped
* in-process (true order): 112 of 577 alakurtti events and 0 of 505 joensuu events dropped

### E4: load at one site

Seeded random clutter (detections that follow no object) is added to joensuu's sensors at increasing rates, until the tracker reaches `max_live_tracks`. For each rate it records:

* alakurtti's refused tracks and its accuracy
* per-event cost at both sites
* joensuu's own losses

The engine groups detections per sensor and instant, both in replay and in the pipeline-stages `fusion` stage, so one site's detections never share an event with another site's. `max_event_detections` therefore cannot shed one site's detections because of another's. The experiment reports this as a structural result rather than measuring it.

---

## Results

Measured on the far pair, at context-foundry `c225c7b` and scenario-foundry `32a7de6`, with scenario seed 42 and clutter seed 42. Each condition ran three times in fresh subprocesses, and every run gave the same tracks. Times are the medians on an AMD Ryzen 5 5500U (12 threads), Python 3.12.16, under Linux.

Commands, run from the repository root:

```bash
python dev/multisite_benchmark.py experiments \
  --scenario-foundry ../scenario-foundry --only E1 --repeats 3 --save e1.json
# likewise --only E2, E3 and E4, with the default lags (1 2 3 5) and clutter rates (0.5 1 2 5)
```

Columns:

* **tracks / false / err m:** `fusion_benchmark.py`'s scoring for the site's tracks.
* **dropped:** the site's events dropped as late.
* **ms/event:** the median `FusionEngine.process` time for the site's events.

Timings vary by up to about 40% between conditions that do the same work (alakurtti took 2.70 ms/event alone and 1.90 ms/event combined), so only the larger differences and the fitted slopes are read as results.

### E1: alone against combined

| condition | site | tracks | false | err m | dropped | refused | ms/event |
|---|---|---:|---:|---:|---:|---:|---:|
| alone | joensuu | 6 | 0 | 21 | 0 | 0 | 1.57 |
| alone | alakurtti | 5 | 0 | 25 | 0 | 0 | 2.70 |
| combined | joensuu | 6 | 0 | 21 | 0 | 0 | 1.85 |
| combined | alakurtti | 5 | 0 | 25 | 0 | 0 | 1.90 |

* Combined, each swarm is followed by the same tracks as when its site is fused alone.
* The cost fit over the combined run gives 0.30 ms per event for each own-site live track and 0.32 ms for each other-site live track. At about 10 tracks per site that cost is too small to show in ms/event.

### E2: frame origin placement

| frame origin | joensuu tracks / false / err m | alakurtti tracks / false / err m |
|---|---|---|
| first detection (62.185N 30.632E, a joensuu target) | 6 / 0 / 21 | 5 / 0 / 25 |
| `FI-MIL-RAD-KOLI-01` | 7 / 0 / 19 | 5 / 0 / 25 |
| `FI-MIL-RAD-LUOSTO-01` | 7 / 0 / 21 | 5 / 0 / 25 |
| midpoint 65.049N 28.456E | 7 / 0 / 19 | 5 / 0 / 25 |

Median error by distance of the track from the frame origin, both sites together:

| frame origin | 0–100 km | 100–200 km | 200–300 km | 300–400 km | 400–500 km | 500–600 km |
|---|---|---|---|---|---|---|
| first detection | 20 m (615) | | | | | 24 m (412) |
| KOLI-01 | 20 m (620) | | | | 24 m (412) | |
| LUOSTO-01 | 21 m (191) | 27 m (221) | | | | 20 m (624) |
| midpoint | | 21 m (154) | 21 m (693) | 27 m (185) | | |

The numbers in brackets are track snapshots.

* Error follows the site, not the distance from the frame origin. joensuu is about 20 m both at the origin and 500–600 km from it, and alakurtti stays at 25 m wherever the origin is.
* Placing the origin anywhere other than the first detection, even at joensuu's own anchor, splits joensuu's decoy wave into two tracks. That points to a borderline association that a small numerical change tips over, not to a frame effect.

### E3: clock offset between sites

| alakurtti behind | alakurtti dropped | alakurtti tracks / false / err m | alakurtti tracks per swarm (Iskander, Geran, heli) | joensuu |
|---|---|---|---|---|
| 0 s (E1) | 0 of 577 | 5 / 0 / 25 | 3, 1, 1 | 6 / 0 / 21 |
| 1 s | 0 of 577 | 5 / 0 / 25 | 3, 1, 1 | 6 / 0 / 21 |
| 2 s | 0 of 577 | 5 / 0 / 25 | 3, 1, 1 | 6 / 0 / 22 |
| 3 s | 112 of 577 (19%) | 4 / 0 / 27 | 2, 1, 1 | 6 / 0 / 22 |
| 5 s | 240 of 577 (42%) | 5 / 0 / 26 | 2, 1, 2 | 6 / 0 / 21 |

* Only the lagging site loses events. joensuu dropped none in any run.
* Drops begin above `LATE_EVENT_TOLERANCE_SECONDS` (2 s). Beyond that, the share dropped depends on how the two sites' events interleave.
* At 3 s one Iskander track is lost. At 5 s the helicopter assault is also split into two tracks.

### E4: load at one site

| clutter at joensuu | joensuu tracks (false) | peak live, joensuu | refused, joensuu | alakurtti tracks / false / err m | refused, alakurtti | alakurtti ms/event | own / other ms per track |
|---|---|---:|---:|---|---:|---:|---|
| none (E1 combined) | 6 (0) | 6 | 0 | 5 / 0 / 25 | 0 | 1.90 | 0.30 / 0.32 |
| 0.5/s | 77 (64) | 23 | 0 | 5 / 0 / 25 | 0 | 4.38 | 0.45 / 0.50 |
| 1/s | 213 (188) | 40 | 0 | 5 / 0 / 25 | 0 | 3.93 | 0.11 / 0.12 |
| 2/s | 536 (502) | 65 | 0 | 5 / 0 / 25 | 0 | 5.78 | 0.12 / 0.13 |
| 5/s | 1656 (1601) | 119 | 147 | 5 / 0 / 25 | 0 | 12.44 | 0.11 / 0.09 |

* **Cost:** joensuu's load slows alakurtti. At 5/s an alakurtti event costs 6.5 times what it does without clutter, while alakurtti's own picture is unchanged.
  * From 1/s up, where there are enough live tracks for a stable fit, an other-site live track costs the same per event as an own-site one: about 0.1 ms.
  * The 0.5/s fit has a negative intercept and is not read as a result.
* **Caps:** at 5/s the tracker reached the cap (peak 119 of 120 live tracks) and refused 147 new tracks, all at joensuu. alakurtti was refused none in this run.
* **Shedding:** no run shed a detection, as expected from the per-sensor, per-instant grouping.
* **joensuu's own accuracy under clutter is not read as a result yet** (see [Open points](#open-points)).

### Answers

1. **Accuracy:** no measurable loss from combining the far pair at scenario load (E1). The single frame costs nothing measurable up to 600 km from its origin (E2).
2. **Cost:** every live track costs about 0.1 ms per event at both sites, whichever site it is at (E1, E4). One site's load therefore sets the other site's per-event cost: 6.5 times as long for alakurtti with 5/s of clutter at joensuu.
3. **Losses:**
   * The shared watermark drops a lagging site's events once its clock is more than 2 s behind: 19% at 3 s and 42% at 5 s, losing a track (E3).
   * The shared track cap was reached but refused only joensuu's tracks in this run (E4).
   * Shedding cannot cross sites.
4. **Frame:** error does not grow with distance from the origin up to 600 km, and moving the origin doesn't change it beyond one borderline track split (E2).
5. **Implication:** per-site fusion instances would remove the cross-site cost and the cross-site late-event drops, and keep each site's track cap to itself. They would not improve accuracy at this distance and load, because there is no accuracy loss to remove.

### Open points

* **Clutter confirms into tracks.** Clutter detections are each a one-off point, yet hundreds become confirmed tracks under `DEFAULT_CONFIRM_HITS` (3). Until it is clear whether that reflects the clutter model or the gating, joensuu's accuracy under clutter is not read as a result. alakurtti's numbers and the cost figures don't depend on it.
* **The cap refused no alakurtti track.** It isn't checked yet whether alakurtti's tracks all began before the cap filled. If so, the experiment needs a site that starts tracks while the other one holds the cap.
* **The near pair and all four sites** remain blocked on scenario configs.

### Inputs for Phase B

* **What to compare a clustered run against:** the E1 combined run and the E3 and E4 conditions, with the same commits, seeds and harness. Clustering should leave E1's accuracy unchanged, remove E3's alakurtti drops, and bring alakurtti's ms/event under E4 back to its alone value.
* **Clustering boundary:**
  * The two sites never share a node id, and their tracks never cross between sites, so clustering by site (by sensor network) is enough for this pair.
  * The near pair, about 85 km apart, is where a geographic boundary would have to be chosen, and it is not measurable yet.
* **Frame:** each instance could keep the current first-detection frame. E2 shows no need to re-anchor at these distances.

---

## Deliverables

| Item | Location |
|---|---|
| Harness | `dev/multisite_benchmark.py`, reusing `dev/fusion_benchmark.py`'s scoring helpers rather than copying them |
| Tests | `tests/test_dev_multisite_benchmark.py`, for the timestamp shift, merge order, clock offset and clutter generation, written before the harness |
| Results | the tables on this page, with the context-foundry and scenario-foundry commits, the seeds and the commands used. Raw JSON stays outside the repository. |

---

## Out of Scope

* The near pair and the four-site run, until scenario configs exist for porvoo_oil_refinery and hamina_kotka.
* Recorded exercise data. Archived records are a later input once records can be mapped to cells.
* Clustering and parallel fusion instances themselves: this page provides the measurement they are compared against.
