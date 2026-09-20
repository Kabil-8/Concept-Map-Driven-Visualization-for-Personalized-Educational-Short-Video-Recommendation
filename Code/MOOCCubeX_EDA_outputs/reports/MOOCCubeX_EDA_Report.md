# MOOCCubeX EDA Report

Generated: 2026-09-12T05:15:53.192998+00:00

## Dataset scale
- Users in profile data: 3,330,294
- Videos/caption records: 59,581
- Courses: 3,781
- Concepts: 637,572
- Users with video behaviour: 310,360
- Video viewing events: 21,295,747
- Watch segments: 25,709,799

## Knowledge graph
- Video-ID-to-caption-ID links: 2,798,892
- Concept-video edges: 624,683
- Concepts connected to videos: 217,038
- Detected video key in concept relations: ccid

## Short-video recommendation readiness
- Video entities no longer than 10 minutes: 34,230
- Initial candidates with metadata, concepts and at least two users: 27,032
- All automated quality checks passed: False

## Recommended preprocessing stage
1. Filter valid users and videos using the saved candidate table.
2. Convert watch segments into implicit engagement labels.
3. Join video IDs to caption IDs and concept IDs.
4. Build chronological train, validation and test splits.
5. Construct the concept graph from concept-video and prerequisite relations.
6. Save model-ready Parquet tables under `/content/drive/MyDrive/DataCon/processed`.
