#!/usr/bin/env python3
"""
Minimal example:
- Load a chosen CARLA town
- Set a birds-eye-view spectator
"""

import argparse
import carla
import sys

def p(msg):
    print(f"[info] {msg}", flush=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town05",
                    help="Town map name, e.g. Town03, Town05, Town07")
    return ap.parse_args()

def main():
    args = parse_args()

    p(f"Connecting to CARLA at {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.load_world(args.town)
    except Exception as e:
        p(f"Failed to connect/load world: {e}")
        sys.exit(1)

    p(f"Loaded: {args.town}")

    # ---------------------------
    # Bird’s-eye-view spectator
    # ---------------------------
    spectator = world.get_spectator()
    # High altitude straight-down
    location = carla.Location(x=-50, y=0, z=260)
    rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
    spectator.set_transform(carla.Transform(location, rotation))

    p("Spectator moved to bird’s eye position.")
    p("Ready.")

if __name__ == "__main__":
    main()
