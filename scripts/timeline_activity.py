#!/usr/bin/env python3
"""
Generate activity timeline visualization from Claude Code and Codex transcripts.
Shows when actual messages/events occurred, not just session start/end times.

Usage:
    # With config file (recommended for reuse)
    uv run --with matplotlib --with numpy scripts/timeline_activity.py timeline_config.json

    # Without config (uses hardcoded SESSIONS below)
    uv run --with matplotlib --with numpy scripts/timeline_activity.py

Output:
    docs/timeline_activity.png
    docs/timeline_activity.svg

The script reads JSONL transcript files from ~/.claude and ~/.codex directories,
extracts timestamps, and creates a visualization showing actual activity patterns.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np

# Colors for each agent type
CLAUDE_COLOR = "#6366f1"  # indigo
CODEX_COLOR = "#f97316"   # orange

# Session definitions with their transcript paths and metadata
SESSIONS = {
    "Claude Code #1": {
        "path": Path.home() / ".claude/projects/-Users-stefansko-conductor-workspaces-jax-js-mcmc-lyon/0b9b0665-46df-40fd-b2d0-10df07d451b3.jsonl",
        "type": "claude",
    },
    "Claude Code #2": {
        "path": Path.home() / ".claude/projects/-Users-stefansko-conductor-workspaces-jax-js-mcmc-kyoto/4045d60b-5016-4c3b-b25d-9174ef216086.jsonl",
        "type": "claude",
    },
    "Claude Code #2.2": {
        "path": Path.home() / ".claude/projects/-Users-stefansko-conductor-workspaces-jax-js-mcmc-kyoto/d87c1e7e-bfb3-473b-9f11-ec436618f965.jsonl",
        "type": "claude",
    },
    "Claude Code #4": {
        "path": Path.home() / ".claude/projects/-Users-stefansko-conductor-workspaces-jax-js-mcmc-cairo/3d2b0069-5075-488e-9614-ad30db3f8c9b.jsonl",
        "type": "claude",
    },
    "Claude Code #6": {
        "path": Path.home() / ".claude/projects/-Users-stefansko-jax-js-mcmc/c2f46c1d-6ecd-4e41-a0af-d1ca3b2cf4e8.jsonl",
        "type": "claude",
    },
    "Codex #2.1": {
        "path": Path.home() / ".codex/sessions/2026/01/17/rollout-2026-01-17T00-08-07-019bc910-e873-7f91-b5e9-659c99dfa485.jsonl",
        "type": "codex",
    },
    "Codex #3": {
        "path": Path.home() / ".codex/sessions/2026/01/16/rollout-2026-01-16T16-46-56-019bc77d-000e-7252-99f9-f3d45926c791.jsonl",
        "type": "codex",
    },
    "Codex #3.1": {
        "path": Path.home() / ".codex/sessions/2026/01/16/rollout-2026-01-16T23-53-58-019bc903-f3de-71f0-bbff-67bd1983c4b3.jsonl",
        "type": "codex",
    },
    "Codex #5a": {
        "path": Path.home() / ".codex/sessions/2026/01/16/rollout-2026-01-16T23-25-23-019bc8e9-c9b9-7b80-9906-aed4c2a8027b.jsonl",
        "type": "codex",
    },
    "Codex #5b": {
        "path": Path.home() / ".codex/sessions/2026/01/17/rollout-2026-01-17T00-38-20-019bc92c-9234-7c31-b59e-0937c69da930.jsonl",
        "type": "codex",
    },
}


def parse_timestamp(ts_str):
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle Z suffix
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        # Handle milliseconds
        if '.' in ts_str and '+' in ts_str:
            base, tz = ts_str.rsplit('+', 1)
            if '.' in base:
                main, frac = base.split('.')
                frac = frac[:6]  # Truncate to microseconds
                ts_str = f"{main}.{frac}+{tz}"
        return datetime.fromisoformat(ts_str)
    except Exception as e:
        return None


def extract_timestamps(path):
    """Extract all timestamps from a JSONL transcript file."""
    timestamps = []
    path = Path(str(path).replace("~", str(Path.home())))

    if not path.exists():
        print(f"    Warning: {path} not found")
        return timestamps

    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Get timestamp from top level
                if 'timestamp' in data and data['timestamp']:
                    ts = parse_timestamp(data['timestamp'])
                    if ts:
                        timestamps.append(ts)
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    return sorted(timestamps)


def create_activity_plot(sessions_data, output_path, title="Session Activity Timeline"):
    """Create activity timeline visualization."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Collect all timestamps to determine time range
    all_timestamps = []
    for name, data in sessions_data.items():
        all_timestamps.extend(data['timestamps'])

    if not all_timestamps:
        print("No timestamps found!")
        return

    min_time = min(all_timestamps)
    max_time = max(all_timestamps)

    # Sort sessions by start time within each type
    def get_start_time(name):
        ts = sessions_data[name]['timestamps']
        return min(ts) if ts else datetime.max

    claude_sessions = sorted(
        [n for n, d in sessions_data.items() if d['type'] == 'claude'],
        key=get_start_time
    )
    codex_sessions = sorted(
        [n for n, d in sessions_data.items() if d['type'] == 'codex'],
        key=get_start_time
    )

    # Create y-positions for each session (bottom to top within each section)
    y_positions = {}
    y = 1
    for name in codex_sessions:
        y_positions[name] = y
        y += 1
    y += 0.5  # Gap between sections
    divider_y = y - 0.25
    for name in claude_sessions:
        y_positions[name] = y
        y += 1

    # Plot activity for each session
    for name, data in sessions_data.items():
        timestamps = data['timestamps']
        if not timestamps:
            continue

        y_pos = y_positions[name]
        color = CLAUDE_COLOR if data['type'] == 'claude' else CODEX_COLOR

        # Plot individual events as small vertical lines
        for ts in timestamps:
            ax.plot([ts, ts], [y_pos - 0.35, y_pos + 0.35],
                   color=color, alpha=0.6, linewidth=0.5)

        # Add density plot (activity intensity)
        if len(timestamps) > 1:
            time_range = (max(timestamps) - min(timestamps)).total_seconds()
            if time_range > 0:
                bin_minutes = max(1, int(time_range / 60 / 30))  # ~30 bins
                bins = max(10, int(time_range / 60 / bin_minutes) + 1)

                start = min(timestamps)
                minutes = [(t - start).total_seconds() / 60 for t in timestamps]

                hist, edges = np.histogram(minutes, bins=bins)

                if hist.max() > 0:
                    hist_norm = hist / hist.max() * 0.3
                    for i in range(len(hist)):
                        if hist[i] > 0:
                            t_start = start + np.timedelta64(int(edges[i]), 'm')
                            t_end = start + np.timedelta64(int(edges[i+1]), 'm')
                            ax.fill_between(
                                [t_start, t_end],
                                [y_pos - hist_norm[i], y_pos - hist_norm[i]],
                                [y_pos + hist_norm[i], y_pos + hist_norm[i]],
                                color=color, alpha=0.4
                            )

    # Add session labels on the right side
    for name, data in sessions_data.items():
        timestamps = data['timestamps']
        if not timestamps:
            continue
        y_pos = y_positions[name]
        color = CLAUDE_COLOR if data['type'] == 'claude' else CODEX_COLOR
        ax.text(max_time + np.timedelta64(5, 'm'), y_pos, name,
               fontsize=9, fontweight='bold', color=color, va='center')

    # Add section divider and labels
    ax.axhline(y=divider_y, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(min_time - np.timedelta64(5, 'm'), y - 0.5, 'Claude Code',
           fontsize=11, fontweight='bold', color='#6366f1', ha='right', va='center')
    ax.text(min_time - np.timedelta64(5, 'm'), len(codex_sessions) / 2 + 0.5, 'Codex',
           fontsize=11, fontweight='bold', color='#f97316', ha='right', va='center')

    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

    plt.xticks(rotation=0)
    # Dynamic date label
    date_str = min_time.strftime('%b %d')
    if min_time.date() != max_time.date():
        date_str = f"{min_time.strftime('%b %d')}-{max_time.strftime('%d')}"
    ax.set_xlabel(f'Time (UTC) — {date_str}, {min_time.year}', fontsize=11)
    ax.set_ylabel('')
    ax.set_title(f'{title}\nActual message/event activity (vertical lines = events, shading = intensity)',
                fontsize=13, fontweight='bold', pad=15)

    # Remove y-axis ticks
    ax.set_yticks([])

    # Set limits with padding for labels
    ax.set_xlim(min_time - np.timedelta64(60, 'm'),
               max_time + np.timedelta64(90, 'm'))
    ax.set_ylim(0.3, len(sessions_data) + 1.5)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-')
    ax.grid(True, axis='x', which='minor', alpha=0.15, linestyle=':')

    # Legend
    legend_elements = [
        Patch(facecolor='#6366f1', alpha=0.6, label='Claude Code'),
        Patch(facecolor='#f97316', alpha=0.6, label='Codex'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save as SVG for the markdown
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {svg_path}")

    plt.close()


def load_config(config_path):
    """Load sessions from a JSON config file."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def main():
    # Check for config file argument
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.exists():
            print(f"Loading config from {config_path}...")
            config = load_config(config_path)
            sessions = config.get('sessions', {})
            title = config.get('title', 'Session Activity Timeline')
            output_dir = Path(config.get('output_dir', '.')).expanduser()
            # Resolve relative paths from config file location
            if not output_dir.is_absolute():
                output_dir = config_path.parent / output_dir
        else:
            print(f"Config file not found: {config_path}")
            print("Using hardcoded SESSIONS...")
            sessions = SESSIONS
            title = "Vibe Engineering Activity Timeline"
            output_dir = Path(__file__).parent.parent / "docs"
    else:
        sessions = SESSIONS
        title = "Vibe Engineering Activity Timeline"
        output_dir = Path(__file__).parent.parent / "docs"

    print("Extracting timestamps from transcripts...")

    sessions_data = {}
    for name, info in sessions.items():
        print(f"  Processing {name}...")
        timestamps = extract_timestamps(info['path'])
        print(f"    Found {len(timestamps)} events")
        sessions_data[name] = {
            'timestamps': timestamps,
            'type': info['type'],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "timeline_activity.png"

    print("\nCreating visualization...")
    create_activity_plot(sessions_data, output_path, title)

    print("\nDone!")


if __name__ == "__main__":
    main()
