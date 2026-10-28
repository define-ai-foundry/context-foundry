"""A replay of co-located CoT self-reports through fusion, the way the fusion stage drives it.

The fixture is a cluster of eleven senders reporting themselves: two ATAK phones reporting
from GPS (`m-g`) every one to three minutes, and nine emplaced cameras reporting their
surveyed positions (`m-i`) every two minutes from four sites up to about 270 m apart,
several cameras sharing a site. Fused as one object, they made a single track that jumped
between the sites at 20-40 m/s and was marked suspect or hostile whenever no phone's
declaration was fresh. Each sender reporting itself is an object of its own.

The records are fused as the stage fuses them: in event-time order, one sensor's reports
of one instant together, with the stage's default settings: three reports within 30 s
confirm a track, and a track is published at most every 5 s. No sender here reports that
often, so each is published only because a self-report confirms its track at once.
"""

import json
import math
from pathlib import Path

from context_foundry.fusion import config
from context_foundry.fusion.cot_input import CotRecordReader
from context_foundry.fusion.engine import FusionEngine
from context_foundry.fusion.publish import PublishPolicy
from context_foundry.fusion.sapient_raw import cot_object_ulid, fusion_node_id
from context_foundry.fusion.tracker import DEFAULT_CONFIRM_HITS, SapientAsynchronousTracker

FIXTURE = Path(__file__).parent / "fixtures" / "self_reports_cluster.jsonl"
NODE = fusion_node_id("p-replay", "fusion")
EARTH_RADIUS_M = 6371000.0


def _records():
    with FIXTURE.open() as lines:
        records = [json.loads(line) for line in lines]
    return sorted(records, key=lambda record: record["event_time"])


def replay(confirm_hits=DEFAULT_CONFIRM_HITS):
    """Every published track record as a dict, in publish order."""
    config.reset_registry()
    config.set_reference_origin(None, None, None)
    records = _records()
    uid_of = {cot_object_ulid(record["uid"]): record["uid"] for record in records}
    reader = CotRecordReader(own_node_id=NODE)
    engine = FusionEngine(
        SapientAsynchronousTracker(confirm_hits=confirm_hits, id_namespace=NODE),
        PublishPolicy("window", window_seconds=5.0),
    )
    published = []

    def fuse(key, detections):
        result = engine.process(key[0], detections)
        assert not result.dropped
        for track, tactical in result.published:
            sources = result.sources[str(track.id)]
            published.append(
                {
                    "track": str(track.id),
                    "time": tactical.timestamp,
                    "lat": tactical.latitude,
                    "lon": tactical.longitude,
                    "speed": tactical.speed_mps,
                    "threat": tactical.threat_level,
                    "class": tactical.classification,
                    "uids": {uid_of[source["object_id"]] for source in sources},
                }
            )

    key, frame, objects = None, [], set()
    for record in records:
        made = reader.detection(record)
        assert made is not None
        timestamp, sensor, detection = made
        uid = detection.metadata["objectId"]
        if frame and (key != (timestamp, sensor) or uid in objects):
            fuse(key, frame)
            frame, objects = [], set()
        key = (timestamp, sensor)
        frame.append(detection)
        objects.add(uid)
    fuse(key, frame)
    return published


def _metres(a, b):
    """Great-circle distance between two published positions."""
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlat, dlon = lat2 - lat1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _jump_speeds(published):
    """Speed between each pair of consecutive records of one track, in m/s."""
    last, speeds = {}, []
    for record in published:
        previous = last.get(record["track"])
        if previous is not None and record["time"] > previous["time"]:
            seconds = (record["time"] - previous["time"]).total_seconds()
            speeds.append(_metres(previous, record) / seconds)
        last[record["track"]] = record
    return speeds


def test_co_located_phones_and_cameras_are_each_their_own_friendly_track():
    published = replay()
    senders = {record["uid"]: record["type"] for record in _records()}
    assert len(senders) == 11
    tracks = {}
    for record in published:
        assert len(record["uids"]) == 1, "a record names exactly the sender behind it"
        tracks.setdefault(record["track"], set()).update(record["uids"])
    # One track per sender: none takes two senders' reports, none is missing.
    assert len(tracks) == 11
    assert sorted(uid for uids in tracks.values() for uid in uids) == sorted(senders)
    assert all(record["speed"] < 20.0 for record in published)
    assert all(speed < 20.0 for speed in _jump_speeds(published))
    assert {record["threat"] for record in published} == {"friend"}
    for record in published:
        (uid,) = record["uids"]
        assert record["class"] == senders[uid]
