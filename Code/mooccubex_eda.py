"""Complete, memory-conscious EDA for the MOOCCubeX dataset in Google Drive.

Colab usage:
    !python /content/drive/MyDrive/DataCon/mooccubex_eda.py

All persistent outputs are written to:
    /content/drive/MyDrive/DataCon/eda_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import ijson
except ImportError as exc:
    raise SystemExit("Install ijson first: pip install ijson") from exc


DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/DataCon/MOOCCubeX")
DEFAULT_OUTPUT_ROOT = Path("/content/drive/MyDrive/DataCon/eda_outputs")
ID_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])([A-Za-z]+_[A-Za-z0-9_-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete streaming EDA for MOOCCubeX")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--sample-records", type=int, default=5)
    return parser.parse_args()


def configure_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mooccubex_eda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_root / "eda.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def first_nonspace_byte(path: Path) -> bytes:
    with path.open("rb") as stream:
        while True:
            value = stream.read(1)
            if not value or not value.isspace():
                return value


def iter_json_records(path: Path) -> Iterator[Any]:
    """Stream JSON Lines, a top-level JSON array, or a top-level JSON object."""
    first = first_nonspace_byte(path)
    if first == b"[":
        with path.open("rb") as stream:
            yield from ijson.items(stream, "item")
        return

    if first != b"{":
        raise ValueError(f"Unsupported JSON format: {path}")

    # Most MOOCCubeX files are one JSON object per line. Test the first useful line.
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first_line = next((line.strip() for line in stream if line.strip()), "")
    try:
        json.loads(first_line)
        json_lines = True
    except json.JSONDecodeError:
        json_lines = False

    if json_lines:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    else:
        with path.open("rb") as stream:
            for key, value in ijson.kvitems(stream, ""):
                if isinstance(value, dict) and not any(
                    candidate in value for candidate in ("id", "user_id", "video_id", "course_id")
                ):
                    value = {"key": key, **value}
                elif not isinstance(value, dict):
                    value = {"key": key, "value": value}
                yield value


def flatten_keys(value: Any, prefix: str = "", depth: int = 0) -> List[str]:
    if depth > 2 or not isinstance(value, dict):
        return []
    result = []
    for key, child in value.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        result.append(full)
        if isinstance(child, dict):
            result.extend(flatten_keys(child, full, depth + 1))
    return result


def pick(record: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, "", []):
            return record[name]
    return None


def extract_identifier(record: Dict[str, Any], kind: str) -> Optional[str]:
    aliases = {
        "user": ("user_id", "uid", "id", "_id"),
        "video": ("video_id", "vid", "id", "_id"),
        "concept": ("concept_id", "cid", "id", "_id"),
        "course": ("course_id", "cid", "id", "_id"),
    }
    value = pick(record, aliases[kind])
    return str(value) if value is not None else None


def json_file_profile(
    path: Path, entity_kind: str, sample_limit: int, progress_every: int, logger: logging.Logger
) -> Tuple[Dict[str, Any], set, List[Dict[str, Any]]]:
    start = time.time()
    key_counts: Counter = Counter()
    missing_id = 0
    duplicate_id = 0
    ids: set = set()
    samples: List[Dict[str, Any]] = []
    records = 0

    for record in iter_json_records(path):
        records += 1
        if isinstance(record, dict):
            key_counts.update(flatten_keys(record))
            identifier = extract_identifier(record, entity_kind)
            if identifier is None:
                missing_id += 1
            elif identifier in ids:
                duplicate_id += 1
            else:
                ids.add(identifier)
            if len(samples) < sample_limit:
                samples.append(record)
        if records % progress_every == 0:
            logger.info("%s: %,d records", path.name, records)

    summary = {
        "file": str(path),
        "records": records,
        "unique_ids": len(ids),
        "missing_ids": missing_id,
        "duplicate_ids": duplicate_id,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    return summary, ids, samples


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def describe(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def analyse_user_video(
    path: Path, output_root: Path, progress_every: int, top_k: int, logger: logging.Logger
) -> Tuple[Dict[str, Any], set, set]:
    logger.info("Analysing user-video interactions: %s", path)
    started = time.time()
    user_ids: set = set()
    observed_video_ids: set = set()
    video_user_count: Counter = Counter()
    video_event_count: Counter = Counter()
    video_segment_count: Counter = Counter()
    video_watch_seconds: Counter = Counter()
    speed_count: Counter = Counter()
    videos_per_user: List[int] = []
    segments_per_user: List[int] = []
    watch_seconds_per_user: List[float] = []
    invalid_segments = 0
    missing_video_ids = 0
    duplicate_users = 0
    min_timestamp: Optional[int] = None
    max_timestamp: Optional[int] = None
    record_count = 0

    for record in iter_json_records(path):
        record_count += 1
        if not isinstance(record, dict):
            continue
        user_id = extract_identifier(record, "user")
        if user_id:
            if user_id in user_ids:
                duplicate_users += 1
            user_ids.add(user_id)

        sequence = pick(record, ("seq", "sequence", "videos", "interactions")) or []
        if isinstance(sequence, dict):
            sequence = [sequence]
        current_videos: set = set()
        current_segments = 0
        current_watch = 0.0

        for event in sequence if isinstance(sequence, list) else []:
            if not isinstance(event, dict):
                continue
            video_id = extract_identifier(event, "video")
            if not video_id:
                missing_video_ids += 1
                continue
            observed_video_ids.add(video_id)
            current_videos.add(video_id)
            video_event_count[video_id] += 1
            segments = pick(event, ("segment", "segments", "watch_segments")) or []
            if isinstance(segments, dict):
                segments = [segments]

            for segment in segments if isinstance(segments, list) else []:
                if not isinstance(segment, dict):
                    invalid_segments += 1
                    continue
                start_point = safe_float(pick(segment, ("start_point", "start", "begin")))
                end_point = safe_float(pick(segment, ("end_point", "end", "finish")))
                speed = safe_float(pick(segment, ("speed", "playback_speed"))) or 1.0
                timestamp = safe_float(pick(segment, ("local_start_time", "timestamp", "time")))
                if start_point is None or end_point is None or end_point < start_point or speed <= 0:
                    invalid_segments += 1
                    continue
                # Actual elapsed viewing time after accounting for playback speed.
                watched = (end_point - start_point) / speed
                current_segments += 1
                current_watch += watched
                video_segment_count[video_id] += 1
                video_watch_seconds[video_id] += watched
                speed_count[round(speed, 2)] += 1
                if timestamp is not None:
                    timestamp_int = int(timestamp)
                    min_timestamp = timestamp_int if min_timestamp is None else min(min_timestamp, timestamp_int)
                    max_timestamp = timestamp_int if max_timestamp is None else max(max_timestamp, timestamp_int)

        for video_id in current_videos:
            video_user_count[video_id] += 1
        videos_per_user.append(len(current_videos))
        segments_per_user.append(current_segments)
        watch_seconds_per_user.append(current_watch)

        if record_count % progress_every == 0:
            logger.info("user-video.json: %,d user records", record_count)

    def utc_string(value: Optional[int]) -> Optional[str]:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat() if value else None

    top_rows = []
    for video_id, users in video_user_count.most_common(top_k):
        top_rows.append(
            {
                "video_id": video_id,
                "unique_users": users,
                "events": video_event_count[video_id],
                "segments": video_segment_count[video_id],
                "watch_seconds": round(video_watch_seconds[video_id], 2),
            }
        )
    pd.DataFrame(top_rows).to_csv(output_root / "top_videos_by_users.csv", index=False)
    pd.DataFrame(speed_count.most_common(), columns=["playback_speed", "segments"]).to_csv(
        output_root / "playback_speed_distribution.csv", index=False
    )

    summary = {
        "user_records": record_count,
        "unique_users": len(user_ids),
        "duplicate_user_records": duplicate_users,
        "unique_watched_videos": len(observed_video_ids),
        "video_events": sum(video_event_count.values()),
        "valid_watch_segments": sum(video_segment_count.values()),
        "invalid_segments": invalid_segments,
        "missing_video_ids": missing_video_ids,
        "total_elapsed_watch_hours": round(sum(video_watch_seconds.values()) / 3600, 2),
        "videos_per_user": describe(videos_per_user),
        "segments_per_user": describe(segments_per_user),
        "elapsed_watch_seconds_per_user": describe(watch_seconds_per_user),
        "earliest_activity_utc": utc_string(min_timestamp),
        "latest_activity_utc": utc_string(max_timestamp),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return summary, user_ids, observed_video_ids


def split_relation_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    if "," in line:
        return [part.strip().strip('"') for part in next(csv.reader([line])) if part.strip()]
    return line.split()


def identify_prefixed(parts: Sequence[str], prefix: str) -> Optional[str]:
    prefix = prefix.upper() + "_"
    for part in parts:
        if part.upper().startswith(prefix):
            return part
    return None


def analyse_concept_video(
    path: Path, known_concepts: set, known_videos: set, output_root: Path, top_k: int
) -> Tuple[Dict[str, Any], set]:
    concept_degree: Counter = Counter()
    video_degree: Counter = Counter()
    linked_videos: set = set()
    duplicate_edges = 0
    malformed = 0
    unknown_concepts = 0
    unknown_videos = 0
    edge_count = 0
    seen_edges: set = set()

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = split_relation_line(line)
            concept_id = identify_prefixed(parts, "K") or identify_prefixed(parts, "C")
            video_id = identify_prefixed(parts, "V")
            if concept_id is None or video_id is None:
                malformed += 1
                continue
            edge = (concept_id, video_id)
            if edge in seen_edges:
                duplicate_edges += 1
                continue
            seen_edges.add(edge)
            edge_count += 1
            concept_degree[concept_id] += 1
            video_degree[video_id] += 1
            linked_videos.add(video_id)
            if known_concepts and concept_id not in known_concepts:
                unknown_concepts += 1
            if known_videos and video_id not in known_videos:
                unknown_videos += 1

    pd.DataFrame(concept_degree.most_common(top_k), columns=["concept_id", "video_count"]).to_csv(
        output_root / "top_concepts_by_video_count.csv", index=False
    )
    return {
        "unique_edges": edge_count,
        "duplicate_edges": duplicate_edges,
        "malformed_lines": malformed,
        "unique_linked_concepts": len(concept_degree),
        "unique_linked_videos": len(video_degree),
        "unknown_concept_references": unknown_concepts,
        "unknown_video_references": unknown_videos,
        "videos_per_concept": describe(list(concept_degree.values())),
        "concepts_per_video": describe(list(video_degree.values())),
    }, linked_videos


def analyse_generic_relation(path: Path) -> Dict[str, Any]:
    rows = 0
    malformed = 0
    left_values = set()
    right_values = set()
    sample = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = split_relation_line(line)
            if len(parts) < 2:
                malformed += 1
                continue
            rows += 1
            left_values.add(parts[0])
            right_values.add(parts[1])
            if len(sample) < 5:
                sample.append(parts[:4])
    return {
        "file": str(path),
        "rows": rows,
        "malformed_lines": malformed,
        "unique_left_values": len(left_values),
        "unique_right_values": len(right_values),
        "sample": sample,
    }


def create_plots(output_root: Path) -> None:
    top_path = output_root / "top_videos_by_users.csv"
    if top_path.exists() and top_path.stat().st_size:
        frame = pd.read_csv(top_path).head(20).iloc[::-1]
        if not frame.empty:
            plt.figure(figsize=(10, 7))
            plt.barh(frame["video_id"], frame["unique_users"], color="#377eb8")
            plt.xlabel("Unique viewers")
            plt.ylabel("Video ID")
            plt.title("Top 20 videos by unique viewers")
            plt.tight_layout()
            plt.savefig(output_root / "top_videos.png", dpi=160)
            plt.close()

    concept_path = output_root / "top_concepts_by_video_count.csv"
    if concept_path.exists() and concept_path.stat().st_size:
        frame = pd.read_csv(concept_path).head(20).iloc[::-1]
        if not frame.empty:
            plt.figure(figsize=(10, 7))
            plt.barh(frame["concept_id"], frame["video_count"], color="#4daf4a")
            plt.xlabel("Connected videos")
            plt.ylabel("Concept ID")
            plt.title("Top 20 concepts by connected video count")
            plt.tight_layout()
            plt.savefig(output_root / "top_concepts.png", dpi=160)
            plt.close()

    speed_path = output_root / "playback_speed_distribution.csv"
    if speed_path.exists() and speed_path.stat().st_size:
        frame = pd.read_csv(speed_path).sort_values("playback_speed")
        if not frame.empty:
            plt.figure(figsize=(9, 5))
            plt.bar(frame["playback_speed"].astype(str), frame["segments"], color="#984ea3")
            plt.xlabel("Playback speed")
            plt.ylabel("Watch segments")
            plt.title("Playback-speed distribution")
            plt.tight_layout()
            plt.savefig(output_root / "playback_speeds.png", dpi=160)
            plt.close()


def build_inventory(data_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and not path.name.endswith(".aria2"):
            rows.append(
                {
                    "relative_path": str(path.relative_to(data_root)),
                    "size_mb": round(path.stat().st_size / 1024**2, 2),
                    "extension": path.suffix.lower(),
                    "download_complete": not Path(str(path) + ".aria2").exists(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    logger = configure_logging(output_root)
    logger.info("Starting MOOCCubeX EDA")
    logger.info("Data root: %s", data_root)
    logger.info("Output root: %s", output_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    required = {
        "video": data_root / "entities/video.json",
        "user": data_root / "entities/user.json",
        "course": data_root / "entities/course.json",
        "concept": data_root / "entities/concept.json",
        "user_video": data_root / "relations/user-video.json",
        "concept_video": data_root / "relations/concept-video.txt",
        "video_ccid": data_root / "relations/video_id-ccid.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    inventory = build_inventory(data_root)
    inventory.to_csv(output_root / "file_inventory.csv", index=False)
    logger.info("Dataset files: %d; total size: %.2f GB", len(inventory), inventory["size_mb"].sum() / 1024)

    entity_summaries = []
    entity_ids: Dict[str, set] = {}
    schema_rows = []
    samples = {}
    for kind in ("video", "user", "course", "concept"):
        logger.info("Profiling %s entities", kind)
        summary, ids, entity_samples = json_file_profile(
            required[kind], kind, args.sample_records, args.progress_every, logger
        )
        summary["entity"] = kind
        entity_summaries.append(summary)
        entity_ids[kind] = ids
        samples[kind] = entity_samples
        key_counter = Counter()
        for sample in entity_samples:
            key_counter.update(flatten_keys(sample))
        for key, count in key_counter.most_common():
            schema_rows.append({"entity": kind, "field": key, "sample_presence": count})

    pd.DataFrame(entity_summaries).to_csv(output_root / "entity_summary.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(output_root / "sample_schema.csv", index=False)
    write_json(output_root / "sample_records.json", samples)

    user_video_summary, interaction_users, watched_videos = analyse_user_video(
        required["user_video"], output_root, args.progress_every, args.top_k, logger
    )
    concept_video_summary, concept_linked_videos = analyse_concept_video(
        required["concept_video"], entity_ids["concept"], entity_ids["video"], output_root, args.top_k
    )
    video_ccid_summary = analyse_generic_relation(required["video_ccid"])

    validation = {
        "interaction_users_missing_from_user_entities": len(interaction_users - entity_ids["user"]),
        "interaction_videos_missing_from_video_entities": len(watched_videos - entity_ids["video"]),
        "entity_videos_never_watched": len(entity_ids["video"] - watched_videos),
        "watched_videos_without_concept_link": len(watched_videos - concept_linked_videos),
        "concept_linked_videos_never_watched": len(concept_linked_videos - watched_videos),
        "video_entity_coverage_by_interactions_percent": round(
            100 * len(entity_ids["video"] & watched_videos) / max(1, len(entity_ids["video"])), 2
        ),
        "watched_video_coverage_by_concept_links_percent": round(
            100 * len(watched_videos & concept_linked_videos) / max(1, len(watched_videos)), 2
        ),
    }

    prerequisite_summaries = {}
    for name in ("psy", "cs", "math"):
        path = data_root / f"prerequisites/{name}.json"
        if path.exists():
            count = 0
            sample = []
            keys = Counter()
            for record in iter_json_records(path):
                count += 1
                if isinstance(record, dict):
                    keys.update(flatten_keys(record))
                if len(sample) < args.sample_records:
                    sample.append(record)
            prerequisite_summaries[name] = {
                "records": count,
                "common_fields": keys.most_common(20),
                "sample": sample,
            }

    complete_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "inventory": {
            "files": len(inventory),
            "total_size_gb": round(inventory["size_mb"].sum() / 1024, 2),
            "incomplete_downloads": inventory.loc[~inventory["download_complete"], "relative_path"].tolist(),
        },
        "entities": entity_summaries,
        "user_video": user_video_summary,
        "concept_video": concept_video_summary,
        "video_id_ccid": video_ccid_summary,
        "cross_file_validation": validation,
        "prerequisites": prerequisite_summaries,
    }
    write_json(output_root / "eda_report.json", complete_report)

    metric_rows = []
    for section, values in complete_report.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                if not isinstance(value, (dict, list)):
                    metric_rows.append({"section": section, "metric": metric, "value": value})
    pd.DataFrame(metric_rows).to_csv(output_root / "key_metrics.csv", index=False)

    create_plots(output_root)

    logger.info("EDA completed successfully")
    logger.info("Main report: %s", output_root / "eda_report.json")
    logger.info("Inventory: %s", output_root / "file_inventory.csv")
    logger.info("Charts and CSV tables are in: %s", output_root)


if __name__ == "__main__":
    main()
