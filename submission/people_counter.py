#!/usr/bin/env python3
"""Count doorway crossings in a CCTV video using YOLO + ByteTrack."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Point = Tuple[int, int]


@dataclass
class TrackState:
    last_center: Optional[Point] = None
    last_side: Optional[float] = None
    last_event_frame: int = -10_000
    counted: bool = False
    staff: bool = False
    seeded_initial: bool = False
    sides: list[float] = field(default_factory=list)
    confirmed_side: int = 0
    candidate_side: int = 0
    candidate_frames: int = 0
    observed_frames: int = 0
    staff_zone_frames: int = 0


def signed_side(point: Point, line_a: Point, line_b: Point) -> float:
    return (line_b[0] - line_a[0]) * (point[1] - line_a[1]) - (
        line_b[1] - line_a[1]
    ) * (point[0] - line_a[0])


def point_in_polygon(point: Point, polygon: Optional[np.ndarray]) -> bool:
    return polygon is not None and cv2.pointPolygonTest(polygon, point, False) >= 0


def parse_polygon(value: Optional[str]) -> Optional[np.ndarray]:
    if not value:
        return None
    points = []
    for pair in value.split(";"):
        x, y = pair.split(",")
        points.append((int(x), int(y)))
    if len(points) < 3:
        raise ValueError("staff polygon needs at least three x,y points")
    return np.asarray(points, dtype=np.int32)


def draw_label(frame: np.ndarray, text: str, origin: Point, color: Tuple[int, int, int]) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-video", default=Path("output/annotated.mp4"), type=Path)
    parser.add_argument("--output-csv", default=Path("output/events.csv"), type=Path)
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--confidence", default=0.38, type=float)
    parser.add_argument("--iou", default=0.50, type=float)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--line",
        default="900,720;1260,500",
        help="doorway line as x1,y1;x2,y2 in source pixels",
    )
    parser.add_argument(
        "--staff-polygon",
        default="0,0;210,0;210,500;0,500",
        help="staff-only polygon x,y;x,y;...; persistent zone presence marks STAFF",
    )
    parser.add_argument("--min-crossing-frames", default=3, type=int)
    parser.add_argument("--event-cooldown-frames", default=26, type=int)
    parser.add_argument("--warmup-seconds", default=2.0, type=float)
    parser.add_argument("--line-margin-px", default=28.0, type=float)
    args = parser.parse_args()

    line_a, line_b = [
        tuple(map(int, item.split(","))) for item in args.line.split(";")
    ]
    staff_polygon = parse_polygon(args.staff_polygon)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not open H.264 output writer; check OpenCV video codecs")

    model = YOLO(args.model)
    states: Dict[int, TrackState] = {}
    headcount = 0
    total_in = 0
    total_out = 0
    frame_index = 0
    warmup_frames = max(1, int(args.warmup_seconds * fps))
    initialized_headcount = False

    with args.output_csv.open("w", newline="") as csv_file:
        event_writer = csv.writer(csv_file)
        event_writer.writerow(["timestamp", "person_id", "event", "headcount"])

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            visible_ids: set[int] = set()
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                for box, track_id, confidence in zip(boxes, ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, y2)
                    visible_ids.add(track_id)
                    state = states.setdefault(track_id, TrackState())
                    state.observed_frames += 1
                    if staff_polygon is not None and point_in_polygon(center, staff_polygon):
                        state.staff_zone_frames += 1
                    if (
                        state.staff_zone_frames >= max(10, int(fps * 0.75))
                        and state.staff_zone_frames / state.observed_frames >= 0.35
                    ):
                        state.staff = True
                    side = signed_side(center, line_a, line_b)
                    state.sides.append(side)
                    state.sides = state.sides[-8:]
                    line_length = math.hypot(line_b[0] - line_a[0], line_b[1] - line_a[1])
                    normalized_side = side / max(line_length, 1.0)
                    stable_side = (
                        1 if normalized_side > args.line_margin_px
                        else -1 if normalized_side < -args.line_margin_px
                        else 0
                    )
                    if stable_side:
                        if stable_side == state.candidate_side:
                            state.candidate_frames += 1
                        else:
                            state.candidate_side = stable_side
                            state.candidate_frames = 1

                    event = None
                    if (
                        state.confirmed_side == 0
                        and stable_side
                        and state.candidate_frames >= args.min_crossing_frames
                    ):
                        state.confirmed_side = stable_side
                    if (
                        state.confirmed_side
                        and stable_side
                        and stable_side != state.confirmed_side
                        and state.candidate_frames >= args.min_crossing_frames
                        and frame_index >= warmup_frames
                    ):
                        crossed_in = state.confirmed_side > 0 and stable_side < 0
                        crossed_out = state.confirmed_side < 0 and stable_side > 0
                        if frame_index - state.last_event_frame >= args.event_cooldown_frames:
                            if crossed_in:
                                event = "IN"
                            elif crossed_out:
                                event = "OUT"
                        state.confirmed_side = stable_side
                    if event and not state.staff and frame_index >= warmup_frames:
                        state.last_event_frame = frame_index
                        if event == "IN":
                            total_in += 1
                            headcount += 1
                        else:
                            total_out += 1
                            headcount = max(0, headcount - 1)
                        event_writer.writerow(
                            [f"{frame_index / fps:.3f}", track_id, event, headcount]
                        )
                    state.last_side = side
                    state.last_center = center

                    color = (0, 165, 255) if state.staff else (0, 220, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(
                        frame,
                        f"ID {track_id} {'STAFF' if state.staff else ''} {confidence:.2f}",
                        (x1, max(28, y1)),
                        color,
                    )

            cv2.line(frame, line_a, line_b, (0, 0, 255), 4)
            cv2.circle(frame, line_a, 7, (0, 0, 255), -1)
            cv2.circle(frame, line_b, 7, (0, 0, 255), -1)
            if staff_polygon is not None:
                cv2.polylines(frame, [staff_polygon], True, (0, 165, 255), 3)
                cv2.putText(
                    frame,
                    "staff exclusion zone",
                    tuple(staff_polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
            if frame_index == warmup_frames and not initialized_headcount:
                headcount = sum(
                    1 for state in states.values()
            if state.confirmed_side < 0 and not state.staff
                )
                initialized_headcount = True
            hud_x = width - 390
            cv2.rectangle(frame, (hud_x, 12), (width - 12, 150), (0, 0, 0), -1)
            staff_count = sum(
                1 for track_id in visible_ids if states[track_id].staff
            )
            customer_count = sum(
                1 for track_id in visible_ids
                if not states[track_id].staff and states[track_id].confirmed_side < 0
            )
            cv2.putText(frame, f"IN: {total_in}", (hud_x + 16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {total_out}", (hud_x + 16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2)
            cv2.putText(frame, f"INSIDE: {headcount}", (hud_x + 16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, f"CUSTOMERS: {customer_count}  STAFF: {staff_count}", (hud_x + 16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
            cv2.putText(frame, f"YOLO26m conf {args.confidence:.2f}", (hud_x + 16, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
            cv2.putText(
                frame,
                f"frame {frame_index + 1}/{frame_count}",
                (width - 210, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
            frame_index += 1

    capture.release()
    writer.release()
    print(f"processed {frame_index} frames at {fps:.2f} FPS")
    print(f"events: IN={total_in}, OUT={total_out}, final_headcount={headcount}")


if __name__ == "__main__":
    main()
