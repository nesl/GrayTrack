import carla
import time
import random
import sys, os, subprocess
import shutil
import json
from queue import Queue, Empty
import signal
from agents.navigation.basic_agent import BasicAgent
import util
from util import make_agent_ignore_traffic_lights
import numpy as np
import pandas as pd
import argparse
import datetime

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="Random seed")
args, _ = parser.parse_known_args()
SEED = args.seed if args.seed is not None else 42

random.seed(SEED)
np.random.seed(SEED)
TARGET_SPEED = random.SystemRandom().randint(20, 32)  # unseeded; one speed per run
print(f"Target speed: {TARGET_SPEED}")

ROUTE_POINTS = 150
MIN_FRAMES = 260
# Encode a fixed IP-only GOP: 1 I-frame followed by 10 P-frames.
P_FRAMES_PER_I_FRAME = 10
GOP = P_FRAMES_PER_I_FRAME + 1

TM_PORT = 8000  # port for traffic manager
TOWN_NAME = "Town05"

CAMERA_CONFIGS = util.CAMERA_CONFIGS

PREFIX_DIR = "/media/ubuntu/samsung_scratch/carla"  # Set this to a folder path string (e.g., "/tmp/my_data/") or leave as "" for current directory
now = datetime.datetime.now()
ROOT_DIR = os.path.join(PREFIX_DIR, now.strftime("%Y_%m_%d_%H_%M_%S") + f"_{SEED}")

print(f"Root directory: {ROOT_DIR}")

# add debugging for print statements
import builtins, traceback
_real_print = builtins.print
def hook(*a, **k):
    s = " ".join(map(str, a))
    if "deque index out of range" in s:
        traceback.print_stack(limit=12)
    _real_print(*a, **k)
builtins.print = hook

def handle_sigterm(signum, frame):
    raise KeyboardInterrupt  # convert SIGTERM into KeyboardInterrupt

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)

def load_town(client):
    print("loading town...")

    world = client.load_world(TOWN_NAME)

    spectator = world.get_spectator()
    # High altitude straight-down
    location = carla.Location(x=-50, y=0, z=260)
    rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
    spectator.set_transform(carla.Transform(location, rotation))

    print("Spectator moved to bird's eye position.")
    print("loaded town")

def worker(client, world, camera_data, per_camera_vehicle_position):

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    
    w_map = world.get_map()

    # -------------------------------------------
    # Set synchronous mode
    # -------------------------------------------
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable sync mode
    settings.fixed_delta_seconds = 1.0 / util.FPS 
    world.apply_settings(settings)

    # -----------------------------
    # Generate route
    # -----------------------------
    spawns = w_map.get_spawn_points()
    # Keep waypoints near cameras 1,2,3,6,8,9,13,14,15,16,17.
    # Bounds = camera-center AABB expanded by ~40m (≈ mean FOV footprint
    # circumradius at those camera heights), so the car stays under this
    # cluster without routing across the rest of Town05.
    VICINITY_X = (-90.0, 115.0)
    VICINITY_Y = (-190.0, 185.0)
    # Cameras 4 and 5 sit on the outer loop at (35, -210) and (27.5, 212.5).
    # Exclude spawn points within ~40m of either camera so route sampling stays
    # away from the north/south outer-loop approach.
    OUTER_LOOP_CAMERAS = ((35.0, -210.0), (27.5, 212.5))
    OUTER_LOOP_EXCLUSION_RADIUS = 40.0
    OUTER_LOOP_EXCLUSION_RADIUS_SQ = OUTER_LOOP_EXCLUSION_RADIUS ** 2
    inside_spawns = [
        sp for sp in spawns
        if VICINITY_X[0] <= sp.location.x <= VICINITY_X[1]
        and VICINITY_Y[0] <= sp.location.y <= VICINITY_Y[1]
        and all(
            (sp.location.x - cam_x) ** 2 + (sp.location.y - cam_y) ** 2
            > OUTER_LOOP_EXCLUSION_RADIUS_SQ
            for cam_x, cam_y in OUTER_LOOP_CAMERAS
        )
    ]
    if not inside_spawns:
        raise RuntimeError(
            f"No spawn points in camera vicinity "
            f"x={VICINITY_X}, y={VICINITY_Y}"
        )

    # We can change the number of waypoints internally.
    route_points = [random.choice(inside_spawns) for _ in range(ROUTE_POINTS)]

    print("Spawn at:", route_points[0].location)

    # -------------------------------------------
    # Spawn the vehicle
    # -------------------------------------------
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.toyota.prius")

    print("Spawning vehicle...")
    vehicle = world.try_spawn_actor(vehicle_bp, route_points[0])

    if vehicle is None:
        print("Failed to spawn vehicle.")
        return 0

    world.player = vehicle 
    # vehicle.set_autopilot(False, TM_PORT)  # important! BehaviorAgent controls it manually
    vehicle.set_autopilot(False)

    # -----------------------------
    # Initialize the agent
    # -----------------------------
    # agent = BehaviorAgent(vehicle, ignore_traffic_light=True, behavior="normal")
    agent = BasicAgent(vehicle, target_speed=TARGET_SPEED)
    make_agent_ignore_traffic_lights(agent)

    print("Starting route...")

    # Tick the world a few times so everything initializes
    for _ in range(5):
        world.tick()

    # -------------------------------------------
    # Simulation loop: destination-only, distance-based heuristic per waypoint
    # -------------------------------------------
    total_frames = 0
    failed = False
    try:
        print_interval = 20  # print every 20 ticks (~1 second if tick = 0.05s)

        for t in route_points[1:]:

            # Reject waypoint if behind the car or would cause route planner to U-turn
            loc = vehicle.get_location()
            transform = vehicle.get_transform()
            forward = transform.get_forward_vector()
            dx = t.location.x - loc.x
            dy = t.location.y - loc.y
            dist_to_wp = (dx * dx + dy * dy) ** 0.5
            if dist_to_wp >= 1e-3:
                dir_x = dx / dist_to_wp
                dir_y = dy / dist_to_wp
                dot = forward.x * dir_x + forward.y * dir_y
                # dot < 0: behind; dot < 0.5: sharp turn / potential U-turn (~60° off heading)
                if dot < 0.77:
                    print("Skipping waypoint (behind vehicle or would cause U-turn):", t.location)
                    continue
            else:
                print("Skipping waypoint (already at waypoint):", t.location)
                continue

            print("changing to new destination:", t.location)
            # Changed: BasicAgent just needs the target location
            # Changed: Older BasicAgent requires a tuple (x, y, z), not a Location object
            agent.set_destination((t.location.x, t.location.y, t.location.z))

            for _ in range(5):
                world.tick()

            tick_counter = 0 # purely for printing
            prev_dist = None
            dist_stall_count = 0

            while True:

                # Changed: BasicAgent handles its own queue state elegantly
                if agent.done():
                    break

                control = agent.run_step()
                vehicle.apply_control(control)
                world.tick()

                total_frames += 1
                vehicle_location = vehicle.get_location()
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
                rot = transform.rotation

                # save vehicle position (one row per camera, each with its own car_visible)
                for cam_idx, cam_info in enumerate(camera_data):
                    x_min, y_min, x_max, y_max = cam_info['bbox']
                    car_visible = (
                        x_min <= vehicle_location.x <= x_max
                        and y_min <= vehicle_location.y <= y_max
                    )
                    per_camera_vehicle_position[cam_idx].loc[len(per_camera_vehicle_position[cam_idx])] = {
                        'x': vehicle_location.x,
                        'y': vehicle_location.y,
                        'z': vehicle_location.z,
                        'theta1': rot.pitch,
                        'theta2': rot.yaw,
                        'theta3': rot.roll,
                        'quantized_heading': int(round(((-rot.yaw) % 360.0) / 90.0) * 90) % 360,
                        'vx': velocity.x,
                        'vy': velocity.y,
                        'vz': velocity.z,
                        'speed': speed,
                        'car_visible': car_visible,
                    }
                # save camera frame
                for cam_info in camera_data:
                    try:
                        frame = cam_info['queue'].get(timeout=0.5)
                    except Empty:
                        raise RuntimeError(f"No frame received from camera {cam_info['id']}")
    
                    # Convert frame to numpy array (BGRA -> BGR)
                    arr = np.frombuffer(frame.raw_data, np.uint8).reshape(
                        (frame.height, frame.width, 4)
                    )[:, :, :3].copy()
                    
                    # Sanity: dimensions must match what we told ffmpeg
                    assert frame.width == util.WIDTH and frame.height == util.HEIGHT
    
                    # Write raw bytes to ffmpeg stdin
                    try:
                        cam_info['ffmpeg_proc'].stdin.write(arr.tobytes())
                    except BrokenPipeError:
                        print(f"Warning: ffmpeg process for camera {cam_info['id']} closed unexpectedly")
                
                dist = vehicle_location.distance(t.location)
                if tick_counter % print_interval == 0:
                    print(f"Distance to waypoint: {dist:.2f} meters")
                tick_counter += 1

                # If distance-to-waypoint stops changing for too long, bail out.
                DIST_STALL_ITERS = 500
                DIST_STALL_EPS = 0.25
                if prev_dist is not None and abs(dist - prev_dist) <= DIST_STALL_EPS:
                    dist_stall_count += 1
                else:
                    dist_stall_count = 0
                prev_dist = dist
                if dist_stall_count >= DIST_STALL_ITERS:
                    print("Breaking: distance to waypoint stalled.")
                    failed = True
                    break

                if dist < 20:  # distance tolerance to move to next waypoint
                    print("breaking due to distance heruistic")
                    break

            if failed:
                break

        print("Reached destination.")
        
    finally:
        print("Destroying vehicle...")
        # CARLA 0.9.10 LocalPlanner owns vehicle cleanup via __del__(),
        # so reset it before explicitly destroying the actor.
        agent._local_planner.reset_vehicle()
        if vehicle.is_alive:
            vehicle.destroy()

        # tick a few times to destroy the vehicle properly
        for _ in range(5):
            world.tick()

    return total_frames, failed

def init_cameras(client, world, camera_data):
    util.check_sync(world)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(util.WIDTH))
    cam_bp.set_attribute("image_size_y", str(util.HEIGHT))
    cam_bp.set_attribute("fov", str(util.FOV))
    cam_bp.set_attribute("sensor_tick", str(1.0 / util.FPS))

    # Spawn all cameras and set up queues and ffmpeg processes
    for config in CAMERA_CONFIGS:
        camera_id = config["id"]
        pos = config["pos"]
        rot = config["rot"]
        
        cam_loc = carla.Location(x=pos[0], y=pos[1], z=pos[2])
        cam_rot = carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
        cam_tf = carla.Transform(cam_loc, cam_rot)
        
        camera = world.try_spawn_actor(cam_bp, cam_tf)
        if camera is None:
            print(f"Warning: Failed to spawn camera {camera_id} (position occupied). Skipping.")
            continue

        # Create queue for this camera and set up listener
        q = Queue()
        camera.listen(q.put)

        # Create videos directory if it doesn't exist
        videos_dir = os.path.join(ROOT_DIR, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Set up ffmpeg process for this camera
        filename = os.path.join(videos_dir, f"camera_{camera_id}.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                    # overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",     # format we'll send from numpy
            "-s", f"{util.WIDTH}x{util.HEIGHT}",
            "-r", str(util.FPS),     # input frame rate (from util.py)
            "-i", "-",               # read video from stdin
            "-an",                   # no audio
            "-c:v", "libx264",
            "-g", str(GOP),          # fixed GOP: 1 I-frame + 10 P-frames
            "-keyint_min", str(GOP),
            "-sc_threshold", "0",    # do not insert scene-cut I-frames
            "-bf", "0",              # disable B-frames in ffmpeg
            "-x264-params", f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes=0",
            "-pix_fmt", "yuv420p",
            filename,
        ]

        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        bbox = util.camera_frustum_bbox_at_z(pos, rot, ground_z=0.0)
        camera_data.append({
            'camera': camera,
            'queue': q,
            'id': camera_id,
            'ffmpeg_proc': proc,
            'filename': filename,
            'config': config,
            'bbox': bbox,
        })
        
        print(f"Camera {camera_id} recording to {filename}")

    print(f"\nSpawned {len(camera_data)} cameras. Recording started.")

def stop_cameras(camera_data):
    print("\nShutting down cameras and ffmpeg processes...")

    for cam_info in camera_data:
        cam_info['camera'].stop()
        cam_info['camera'].destroy()
        
        if cam_info['ffmpeg_proc'].stdin:
            cam_info['ffmpeg_proc'].stdin.close()
        
        cam_info['ffmpeg_proc'].wait()
        print(f"Camera {cam_info['id']} saved to {cam_info['filename']}")

def main():
    os.makedirs(ROOT_DIR, exist_ok=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    load_town(client)
    world = client.get_world()

    # start cameras
    camera_data = []
    init_cameras(client, world, camera_data)

    # start worker function
    per_camera_vehicle_position = [
        pd.DataFrame(columns=['x', 'y', 'z', 'theta1', 'theta2', 'theta3', 'quantized_heading', 'vx', 'vy', 'vz', 'speed', 'car_visible'])
        for _ in range(len(camera_data))
    ]
    total_frames, failed = worker(client, world, camera_data, per_camera_vehicle_position)
    for i, df in enumerate(per_camera_vehicle_position):
        assert total_frames == len(df), f"total_frames ({total_frames}) != len(per_camera_vehicle_position[{i}]) ({len(df)})"

    # save truth dataframe to y/camera_X_truth.csv for each camera
    y_dir = os.path.join(ROOT_DIR, "y")
    os.makedirs(y_dir, exist_ok=True)
    for cam_info, vp in zip(camera_data, per_camera_vehicle_position):
        path = os.path.join(y_dir, f"camera_{cam_info['id']}_truth.csv")
        vp.to_csv(path, index=False)
        print(f"Saved truth to {path}")

    # save run parameters to params.json
    camera_params = {
        str(cfg["id"]): {"pos": list(cfg["pos"]), "rot": list(cfg["rot"])}
        for cfg in util.CAMERA_CONFIGS
    }
    params_path = os.path.join(ROOT_DIR, "params.json")
    with open(params_path, "w") as f:
        json.dump({
            "random_seed": SEED,
            "total_frames": total_frames,
            "camera_params": camera_params,
        }, f, indent=4)
    print(f"Saved params to {params_path}")

    # cleanup
    stop_cameras(camera_data)

    # reset to async mode so we don't freeze the simulator
    print("resetting to async...")
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    if failed:
        shutil.rmtree(ROOT_DIR, ignore_errors=True)
        print(f"FAILURE: distance-to-waypoint stalled; deleted saved folder {ROOT_DIR}")
    elif total_frames < MIN_FRAMES:
        shutil.rmtree(ROOT_DIR, ignore_errors=True)
        print(f"Skipping save: only {total_frames} frames (< {MIN_FRAMES}); deleted {ROOT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(0)
