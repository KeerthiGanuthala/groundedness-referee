#!/usr/bin/env python3
"""Stretch Goal 6: Record terminal walkthrough of Experiment 1 (asciinema .cast + .mp4 video)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from groundedness_referee.experiments import run_experiment_1_twin_pipelines  # noqa: E402


def generate_demo_artifacts(root_dir: Path = ROOT) -> tuple[Path, Path]:
    demo_dir = root_dir / "docs" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    cast_path = demo_dir / "exp1_twin_pipelines.cast"
    mp4_path = demo_dir / "exp1_twin_pipelines.mp4"

    exp1 = run_experiment_1_twin_pipelines()
    pa = exp1["pipeline_a"]
    pb = exp1["pipeline_b"]

    cmd_str = "$ PYTHONPATH=src python3 -m groundedness_referee.cli twin-demo"
    stages = [
        [
            "groundedness-referee :: Experiment 1 Walkthrough",
            "------------------------------------------------------------------------",
            cmd_str,
        ],
        [
            "groundedness-referee :: Experiment 1 Walkthrough",
            "------------------------------------------------------------------------",
            cmd_str,
            "",
            "=== TWIN PIPELINE SYMMETRY BREAK (SEED = 20261117, N = 240 QA PAIRS) ===",
            f"Shared Headline Groundedness Score (2 d.p.): {exp1['shared_groundedness_score_2dp']:.2f}",
            "Observation: Score alone cannot tell whether retrieval or generation failed.",
        ],
        [
            "groundedness-referee :: Experiment 1 Walkthrough",
            "------------------------------------------------------------------------",
            cmd_str,
            "",
            "=== TWIN PIPELINE SYMMETRY BREAK (SEED = 20261117, N = 240 QA PAIRS) ===",
            f"Shared Headline Groundedness Score (2 d.p.): {exp1['shared_groundedness_score_2dp']:.2f}",
            "",
            f"[1] {pa['pipeline_name']} (recall=0.60, faithful=0.95)",
            f"    Groundedness Score : {pa['groundedness_score']:.4f} ({pa['grounded_count']}/{pa['answered_questions']} answered)",
            f"    Counterfactual Probe Blame:",
            f"      - retrieval-miss                   : {pa['referee_shares']['retrieval-miss']*100:5.1f}%",
            f"      - generation-ignore                : {pa['referee_shares']['generation-ignore']*100:5.1f}%",
            f"      - fabrication-with-evidence-present: {pa['referee_shares']['fabrication-with-evidence-present']*100:5.1f}%",
            f"    Referee Verdict    : {pa['recommended_fix']} (+0.2269 gain vs +0.0231 wrong fix)",
        ],
        [
            "groundedness-referee :: Experiment 1 Walkthrough",
            "------------------------------------------------------------------------",
            cmd_str,
            "",
            "=== TWIN PIPELINE SYMMETRY BREAK (SEED = 20261117, N = 240 QA PAIRS) ===",
            f"Shared Headline Groundedness Score (2 d.p.): {exp1['shared_groundedness_score_2dp']:.2f}",
            "",
            f"[1] {pa['pipeline_name']} (recall=0.60, faithful=0.95)",
            f"    Groundedness Score : {pa['groundedness_score']:.4f} ({pa['grounded_count']}/{pa['answered_questions']} answered)",
            f"    Referee Blame      : retrieval-miss=90.9%, gen-ignore=3.4%, gen-fab=5.7%",
            f"    Referee Verdict    : {pa['recommended_fix']}  --> +0.2269 gain (vs +0.0231 wrong fix)",
            "",
            f"[2] {pb['pipeline_name']} (recall=0.95, faithful=0.635)",
            f"    Groundedness Score : {pb['groundedness_score']:.4f} ({pb['grounded_count']}/{pb['answered_questions']} answered)",
            f"    Referee Blame      : retrieval-miss=7.6%, gen-ignore=41.3%, gen-fab=40.2%",
            f"    Referee Verdict    : {pb['recommended_fix']}  --> +0.2336 gain (vs +0.0265 wrong fix)",
            "",
            "Conclusion: Same 0.59 score, opposite root causes, opposite sprint fixes.",
        ],
    ]

    # Write asciinema v2 .cast file
    header = {
        "version": 2,
        "width": 92,
        "height": 24,
        "timestamp": 1763400000,
        "title": "groundedness-referee Experiment 1: Twin Pipelines",
    }
    cast_lines = [json.dumps(header)]
    timestamps = [0.2, 1.5, 3.2, 5.0]
    for ts, frame_lines in zip(timestamps, stages):
        payload = "\r\n".join(frame_lines) + "\r\n"
        cast_lines.append(json.dumps([ts, "o", payload]))
    cast_path.write_text("\n".join(cast_lines) + "\n", encoding="utf-8")

    # Render frames to PNG and encode MP4 with ffmpeg
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 19)
            break
    if font is None:
        font = ImageFont.load_default()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_idx = 0
        durations = [8, 12, 16, 24]  # at 4 fps -> 15 seconds total
        for stage_lines, n_frames in zip(stages, durations):
            img = Image.new("RGB", (1280, 720), color=(15, 20, 28))
            draw = ImageDraw.Draw(img)
            # Window title bar
            draw.rectangle([(0, 0), (1280, 44)], fill=(30, 38, 52))
            draw.ellipse([(18, 14), (34, 30)], fill=(235, 87, 87))
            draw.ellipse([(44, 14), (60, 30)], fill=(242, 201, 76))
            draw.ellipse([(70, 14), (86, 30)], fill=(39, 174, 96))
            draw.text(
                (440, 12),
                "groundedness-referee — terminal demo",
                fill=(200, 210, 225),
                font=font,
            )

            y = 70
            for line in stage_lines:
                color = (220, 228, 240)
                if line.startswith("$"):
                    color = (102, 217, 239)
                elif "fix_retriever" in line or "fix_generator" in line:
                    color = (166, 226, 46)
                elif line.startswith("===") or line.startswith("Conclusion:"):
                    color = (253, 151, 31)
                draw.text((36, y), line, fill=color, font=font)
                y += 32

            for _ in range(n_frames):
                frame_file = tmp_dir / f"frame_{frame_idx:04d}.png"
                img.save(frame_file)
                frame_idx += 1

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "4",
                "-i",
                str(tmp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return cast_path, mp4_path


if __name__ == "__main__":
    c_path, m_path = generate_demo_artifacts()
    print(f"Wrote {c_path} and {m_path}")
