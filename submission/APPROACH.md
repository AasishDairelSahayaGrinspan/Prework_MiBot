# Store entry/exit counting note

## Approach

The video is a fixed 1270x720 overhead retail camera. I use the open-source YOLO26 medium detector, restricted to the COCO `person` class, and Ultralytics ByteTrack for frame-to-frame identities. The doorway threshold was moved to the lower-right physical door, using a line from `(900,720)` to `(1260,500)`, rather than the interior of the mat. The bottom-center foot point of each box is used for crossing decisions because it is a better approximation of where a person is walking than the box centre. A hysteresis band, multi-frame stable-side requirement, two-second initial-occupancy warm-up, and per-track cooldown are applied before a crossing is accepted. Every accepted crossing is written as `timestamp,person_id,event,headcount`; the overlay shows the same running state.

## Edge cases

- **Occlusion:** ByteTrack bridges short detector gaps using motion and association. The hysteresis band and stable-side dwell prevent a partially hidden box from immediately flipping sides. Long complete occlusions can still cause an ID switch and are reported as a limitation.
- **Groups entering together:** each YOLO person detection gets its own ByteTrack ID, and crossings are evaluated per track rather than per blob. This handles side-by-side groups; severe full overlap can still undercount until people separate.
- **Re-entry:** crossing state is maintained per track, and the cooldown suppresses rapid jitter. A genuine `OUT` followed later by `IN` is accepted as a new event, including when ByteTrack assigns a new ID after a long disappearance.
- **Staff vs customers:** the left cashier/checkout area is configured as a staff-only polygon (`0,0;210,0;210,500;0,500`). A track must be observed there persistently before it is labeled `STAFF`; those tracks are orange and excluded from customer IN/OUT totals. All other tracks are `CUSTOMER` (green). This is a zone/behavior rule, not a claim that clothing can identify staff; customers standing in that area can be false positives, so a staff roster or trained appearance/ReID classifier is the production upgrade.

## What failed / limitations

The camera is wide-angle and people overlap near the entrance, so perfect identity continuity is not possible. A line-only counter also cannot recover a person fully hidden behind a group, and a staff polygon can misclassify a customer who enters that area. The clip has no ground-truth annotations, so accuracy should be reported only after manually annotating a short representative sample; the CSV is event-level output, not a claimed benchmark.

## More time

I would annotate several entrance-heavy segments, tune the line and confidence on a validation split, compare ByteTrack with BoT-SORT/ReID, add an entrance-region tracklet stitcher, and use a camera-specific detector fine-tuned for overhead shoppers. I would also add a calibrated staff roster/zone and a review export for low-confidence or overlapping crossings.
