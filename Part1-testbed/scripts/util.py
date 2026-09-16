import random
import carla
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOV = 90
FPS = 20

# Camera configurations
CAMERA_CONFIGS = [
    # Visible cameras
    # {"id": 4, "pos": (35.000, -210.000, 7.500), "rot": (-28.00, 86.00, 0.00)},
    # {"id": 5, "pos": (27.500, 212.500, 7.500), "rot": (-28.00, 268.00, 0.00)},

    # Encrypted cameras (active: 1, 2, 3, 6, 8, 9, 13, 14, 15, 16, 17)
    {"id": 1, "pos": (25.000, -165.000, 17.500), "rot": (-62.00, 44.00, 0.00)},
    {"id": 2, "pos": (25.000, -57.500, 15.000), "rot": (-54.00, 50.00, 0.00)},
    {"id": 3, "pos": (25.000, 30.000, 15.000), "rot": (-54.00, 50.00, 0.00)},
    {"id": 6, "pos": (22.500, 130.000, 12.500), "rot": (-60.00, 44.00, 0.00)},
    # {"id": 7, "pos": (62.500, -2.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 8, "pos": (50.000, -80.000, 12.500), "rot": (-50.00, 302.00, 0.00)},
    {"id": 9, "pos": (47.500, 82.500, 12.500), "rot": (-50.00, 50.00, 0.00)},
    # {"id": 10, "pos": (127.500, 0.000, 15.000), "rot": (-90.00, 270.00, 0.00)},
    # {"id": 11, "pos": (132.500, -132.500, 15.000), "rot": (-90.00, 302.00, 0.00)},
    # {"id": 12, "pos": (132.500, 127.500, 12.500), "rot": (-90.00, 56.00, 0.00)},
    {"id": 13, "pos": (7.500, -100.000, 17.500), "rot": (-54.00, 128.00, 0.00)},
    {"id": 14, "pos": (7.500, -10.000, 17.500), "rot": (-56.00, 124.00, 0.00)},
    {"id": 15, "pos": (7.500, 80.000, 17.500), "rot": (-56.00, 124.00, 0.00)},
    {"id": 16, "pos": (-57.500, -70.000, 12.500), "rot": (-44.00, 44.00, 0.00)},
    {"id": 17, "pos": (-57.500, 22.500, 12.500), "rot": (-44.00, 44.00, 0.00)},
    # {"id": 18, "pos": (-87.500, 0.000, 22.500), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 19, "pos": (-87.500, -92.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 20, "pos": (-87.500, 87.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 21, "pos": (-75.000, 145.000, 20.000), "rot": (-90.00, 72.00, 0.00)},
    # {"id": 22, "pos": (-75.000, -137.500, 20.000), "rot": (-90.00, 296.00, 0.00)},
    # {"id": 23, "pos": (-162.500, -92.500, 22.500), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 24, "pos": (-155.000, -5.000, 25.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 25, "pos": (-160.000, 87.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 26, "pos": (-125.000, 45.000, 20.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 27, "pos": (-125.000, -45.000, 20.000), "rot": (-90.00, 0.00, 0.00)},
    # {"id": 28, "pos": (-175.000, -137.500, 20.000), "rot": (-90.00, 54.00, 0.00)},
    # {"id": 29, "pos": (-175.000, 145.000, 20.000), "rot": (-90.00, 296.00, 0.00)},

    # Overhead spectator
    {"id": "overhead", "pos": (-50, 0, 260), "rot": (-90, 0, 0)}
]

# vehicle.toyota.prius collision OBB half-extents (meters). Length along local +X,
# width along +Y. Center offset is carla.BoundingBox.location.xy relative to the
# actor transform (normally ~0); z is unused for the flat ground footprint.
PRIUS_EXTENT_X = 2.2099347
PRIUS_EXTENT_Y = 0.996169
PRIUS_BBOX_OFFSET_X = 0.0
PRIUS_BBOX_OFFSET_Y = 0.0


def _camera_transform(camera_pos, camera_rot_deg):
    return carla.Transform(
        carla.Location(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2]),
        carla.Rotation(
            pitch=camera_rot_deg[0],
            yaw=camera_rot_deg[1],
            roll=camera_rot_deg[2],
        ),
    )


def camera_frustum_polygon_at_z(camera_pos, camera_rot_deg, ground_z=0.0):
    """
    Project the camera image rectangle onto z=ground_z via the CARLA pinhole
    model (horizontal FOV) and return the resulting world-XY quad (4x2).

    Uses the UE camera convention from CARLA docs:
      standard (X right, Y down, Z forward) <-> UE (x forward, y right, z up)
      via (X, Y, Z) = (y_ue, -z_ue, x_ue).
    """
    fov_rad = np.deg2rad(FOV)
    focal = WIDTH / (2.0 * np.tan(fov_rad / 2.0))
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    M = np.array(_camera_transform(camera_pos, camera_rot_deg).get_matrix())
    R = M[:3, :3]
    origin = M[:3, 3]

    pts = []
    for u, v in ((0.0, 0.0), (WIDTH, 0.0), (WIDTH, HEIGHT), (0.0, HEIGHT)):
        # Ray in standard camera coords, then convert to UE local.
        X = (u - cx) / focal
        Y = (v - cy) / focal
        dir_ue = np.array([1.0, X, -Y], dtype=float)  # (x_ue, y_ue, z_ue)
        dir_ue /= np.linalg.norm(dir_ue)
        dir_w = R @ dir_ue
        if abs(dir_w[2]) < 1e-12:
            raise RuntimeError(
                f"Camera ray parallel to ground plane at image ({u}, {v})"
            )
        s = (ground_z - origin[2]) / dir_w[2]
        if s <= 0:
            raise RuntimeError(
                f"Camera ray does not hit z={ground_z} (s={s}) at image ({u}, {v})"
            )
        hit = origin + s * dir_w
        pts.append((float(hit[0]), float(hit[1])))
    return np.asarray(pts, dtype=float)


def camera_frustum_bbox_at_z(camera_pos, camera_rot_deg, ground_z=0.0):
    """
    Axis-aligned bounding box of camera_frustum_polygon_at_z in world (x, y).

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    poly = camera_frustum_polygon_at_z(camera_pos, camera_rot_deg, ground_z)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def vehicle_footprint_polygon(
    x,
    y,
    yaw_deg,
    extent_x=PRIUS_EXTENT_X,
    extent_y=PRIUS_EXTENT_Y,
    offset_x=PRIUS_BBOX_OFFSET_X,
    offset_y=PRIUS_BBOX_OFFSET_Y,
):
    """
    Oriented rectangle of the vehicle as a flat 2D object in world XY (4x2).

    Matches CARLA's bounding-box convention: center at actor (x, y) plus
    yaw-rotated (offset_x, offset_y) (= BoundingBox.location.xy), with
    half-extents (extent_x, extent_y).
    """
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    cx = x + offset_x * c - offset_y * s
    cy = y + offset_x * s + offset_y * c
    corners = []
    for lx, ly in (
        (extent_x, extent_y),
        (extent_x, -extent_y),
        (-extent_x, -extent_y),
        (-extent_x, extent_y),
    ):
        corners.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return np.asarray(corners, dtype=float)


def _point_in_convex_polygon(point, poly):
    """True if point is inside a convex polygon (including boundary)."""
    x, y = point
    n = len(poly)
    sign = 0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) <= 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _segments_intersect(a, b, c, d, eps=1e-12):
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    def on_seg(o, p, q):
        return (
            min(o[0], p[0]) - eps <= q[0] <= max(o[0], p[0]) + eps
            and min(o[1], p[1]) - eps <= q[1] <= max(o[1], p[1]) + eps
        )

    d1, d2 = cross(a, b, c), cross(a, b, d)
    d3, d4 = cross(c, d, a), cross(c, d, b)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    if abs(d1) <= eps and on_seg(a, b, c):
        return True
    if abs(d2) <= eps and on_seg(a, b, d):
        return True
    if abs(d3) <= eps and on_seg(c, d, a):
        return True
    if abs(d4) <= eps and on_seg(c, d, b):
        return True
    return False


def convex_polygons_intersect(poly_a, poly_b):
    """
    True if two convex polygons share any area (including edge/point touch).
    """
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if any(_point_in_convex_polygon(p, b) for p in a):
        return True
    if any(_point_in_convex_polygon(p, a) for p in b):
        return True
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _segments_intersect(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    return False


def is_car_visible_in_fov(
    x, y, yaw_deg, fov_polygon,
    extent_x=PRIUS_EXTENT_X,
    extent_y=PRIUS_EXTENT_Y,
    fov_aabb=None,
    offset_x=PRIUS_BBOX_OFFSET_X,
    offset_y=PRIUS_BBOX_OFFSET_Y,
):
    """
    True if the flat vehicle OBB intersects the camera FOV on the ground plane.

    Both polygons are in world XY: the FOV is the image rectangle unprojected
    onto z=0, and the vehicle is the axis-aligned (in vehicle frame) 2D box
    from BoundingBox extent/location. Any shared area or boundary touch counts.
    """
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    cx = x + offset_x * c - offset_y * s
    cy = y + offset_x * s + offset_y * c
    # Coarse reject: circle around BB center vs FOV AABB.
    diag = (extent_x ** 2 + extent_y ** 2) ** 0.5
    if fov_aabb is None:
        xs, ys = fov_polygon[:, 0], fov_polygon[:, 1]
        fov_aabb = (xs.min(), ys.min(), xs.max(), ys.max())
    x_min, y_min, x_max, y_max = fov_aabb
    if (
        cx + diag < x_min
        or cx - diag > x_max
        or cy + diag < y_min
        or cy - diag > y_max
    ):
        return False
    car = vehicle_footprint_polygon(
        x, y, yaw_deg, extent_x, extent_y, offset_x=offset_x, offset_y=offset_y
    )
    return convex_polygons_intersect(fov_polygon, car)


def make_agent_ignore_traffic_lights(agent):
    """
    BasicAgent in some CARLA versions has no set_ignore_traffic_lights().
    Monkey-patch run_step so that _is_light_red is never treated as a hazard.
    """
    _original_run_step = agent.run_step
    _original_is_light_red = agent._is_light_red

    def _run_step_ignore_lights(debug=False):
        agent._is_light_red = lambda lights_list: (False, None)
        try:
            return _original_run_step(debug)
        finally:
            agent._is_light_red = _original_is_light_red

    agent.run_step = _run_step_ignore_lights


def common_init():
    random.seed(42)

def check_sync(world):
    # Detect sync mode & set up a safe tick loop
    settings = world.get_settings()

    print(f"Sync mode: {settings.synchronous_mode}")

    if settings.synchronous_mode == False:
        print("CARLA is in async mode! Setting to synchronous mode...")

        settings.synchronous_mode = True        # Enable synchronous mode
        settings.fixed_delta_seconds = 1 / FPS      # 20 FPS simulation step (adjust as needed)

        world.apply_settings(settings)

def create_camera(world):
    bp_lib = world.get_blueprint_library()

    # Camera blueprint
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WIDTH))
    cam_bp.set_attribute("image_size_y", str(HEIGHT))
    cam_bp.set_attribute("fov", str(FOV))
    cam_bp.set_attribute("sensor_tick", str(1.0 / FPS))

    # Pick a reasonable world-space location and aim it down the road
    # sp = random.choice(world.get_map().get_spawn_points())
    # cam_loc = sp.location + carla.Location(x=8.0, y=0.0, z=8.0)
    # cam_rot = carla.Rotation(pitch=-15.0, yaw=sp.rotation.yaw)  # look along lane

    # Hardcoded coordinates
    cam_loc = carla.Location(x=151.105438, y=-200.910126, z=8.275307)
    cam_rot = carla.Rotation(pitch=-15.000000, yaw=-178.560471, roll=0.000000)  # look along lane
    cam_tf = carla.Transform(cam_loc, cam_rot)

    return((cam_bp, cam_tf))

def get_closest_carla_vehicle(pos, vehicles):
    closest_act_pos = np.zeros(2)
    min_dist = float('inf')

    for vehicle in vehicles:
        act_pos_raw = vehicle.get_location()
        act_pos = np.asarray([act_pos_raw.x, act_pos_raw.y])
        dist = np.linalg.norm(act_pos - pos)

        if dist < min_dist:
            min_dist = dist
            closest_act_pos = act_pos

    return closest_act_pos, min_dist

def collect_vehicle_positions(world, vehicle_positions_dict, frame_number):
    """
    Iterate through all vehicles in the world and append their X, Y coordinates
    to arrays labeled with the car ID, along with the frame number.
    
    Args:
        world: CARLA world object
        vehicle_positions_dict: Dictionary to store positions, keyed by vehicle ID
                               Format: {vehicle_id: [(frame_num, x, y), (frame_num, x, y), ...]}
        frame_number: Current frame/tick number
    
    Returns:
        None (modifies vehicle_positions_dict in place)
    """
    vehicles = world.get_actors().filter('vehicle.*')
    
    for vehicle in vehicles:
        vehicle_id = vehicle.id
        location = vehicle.get_location()
        x, y = location.x, location.y
        
        # Initialize list for this vehicle if it doesn't exist
        if vehicle_id not in vehicle_positions_dict:
            vehicle_positions_dict[vehicle_id] = []
        
        # Append current position with frame number
        vehicle_positions_dict[vehicle_id].append((frame_number, x, y))