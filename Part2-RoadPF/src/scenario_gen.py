
from __future__ import annotations

import numpy as np

from .schema import Trajectory
from .sensors import Region, find_boundary_crossings


class RouteRejected(Exception):
    pass


def _simulate_along_route(graph, route_edges, base_speed, dt, max_steps,
                          speed_noise_frac, rng):
    xs = np.zeros(max_steps)
    ys = np.zeros(max_steps)
    sp = np.zeros(max_steps)
    hd = np.zeros(max_steps)
    ed = np.zeros(max_steps, dtype=int)

    cur_r = 0.0
    path_idx = 0
    cur_edge = int(route_edges[0])
    n_used = max_steps

    for t in range(max_steps):
        pos = graph.xy_from_re(np.array([cur_r]), np.array([cur_edge]))
        xs[t], ys[t] = float(pos[0, 0]), float(pos[1, 0])
        hd[t] = float(graph.edge_dir[cur_edge])
        ed[t] = cur_edge
        step_speed = max(0.5, base_speed + rng.normal(0, base_speed * speed_noise_frac))
        sp[t] = step_speed

        cur_r += step_speed * dt
        done = False
        for _ in range(50):
            elen = float(graph.edge_len[cur_edge])
            if cur_r <= elen:
                break
            cur_r -= elen
            path_idx += 1
            if path_idx >= len(route_edges):
                done = True
                break
            cur_edge = int(route_edges[path_idx])
        if done:
            n_used = t + 1
            break

    return (np.column_stack([xs, ys])[:n_used], sp[:n_used], hd[:n_used],
            ed[:n_used], n_used)


def _active_window(region: Region, xy):
    inside = region.contains(xy)
    if not inside.any():
        raise RouteRejected("trajectory never enters the region")
    idx = np.flatnonzero(inside)
    return int(idx[0]), int(idx[-1])


def build_transit_route(graph, region: Region, rng, min_inside_dist: float = 150.0,
                        max_attempts: int = 400):
    inbound, outbound = find_boundary_crossings(graph, region)
    if not inbound or not outbound:
        raise RuntimeError("region has no cordon crossings")

    for _ in range(max_attempts):
        i_edge, i_from, _i_to, _ = inbound[rng.randint(len(inbound))]
        o_edge, _o_from, o_to, _ = outbound[rng.randint(len(outbound))]
        if i_from == o_to:
            continue
        try:
            _nodes, edges = graph.shortest_path_edges(i_from, o_to)
        except Exception:
            continue
        if not edges:
            continue

        mids = np.array([0.5 * (graph.edge_p0[e] + graph.edge_p1[e]) for e in edges])
        inside_mask = region.contains(mids)
        inside_len = float(sum(graph.edge_len[e] for e, m in zip(edges, inside_mask) if m))
        if inside_len < min_inside_dist:
            continue
        return edges, int(i_from), int(o_to), inside_len

    raise RouteRejected(f"no transit route with >= {min_inside_dist} m inside "
                        f"the region after {max_attempts} attempts")


def generate_trajectory(graph, region: Region, seed: int, vehicle_id: int = 0,
                        dt: float = 0.25, max_steps: int = 1200,
                        speed_min: float = 6.0, speed_max: float = 12.0,
                        speed_noise_frac: float = 0.15,
                        min_inside_dist: float = 150.0,
                        exit_pad_steps: int = 8,
                        trajectory_id: str | None = None) -> Trajectory:
    rng = np.random.RandomState(seed)
    route_edges, src, dst, inside_len = build_transit_route(
        graph, region, rng, min_inside_dist=min_inside_dist)

    base_speed = rng.uniform(speed_min, speed_max)
    xy, speeds, headings, edge_ids, n_used = _simulate_along_route(
        graph, route_edges, base_speed, dt, max_steps, speed_noise_frac, rng)

    a_start, a_end = _active_window(region, xy)
    keep = min(len(xy), a_end + 1 + exit_pad_steps)
    xy, speeds, headings, edge_ids = xy[:keep], speeds[:keep], headings[:keep], edge_ids[:keep]
    timestamps = np.arange(len(xy), dtype=float) * dt

    stats = graph.route_topology_stats(route_edges)
    inside_edges = [e for e in route_edges
                    if region.contains_point(*(0.5 * (graph.edge_p0[e] + graph.edge_p1[e])))]
    inside_stats = graph.route_topology_stats(inside_edges)

    tid = trajectory_id if trajectory_id is not None else f"traj{seed:05d}"
    return Trajectory(
        trajectory_id=tid,
        vehicle_id=vehicle_id,
        timestamps=timestamps,
        xy=xy,
        speed=speeds,
        heading=headings,
        road_edge_id=edge_ids,
        active_start=a_start,
        active_end=min(a_end, len(xy) - 1),
        meta={
            "seed": int(seed),
            "dt": float(dt),
            "source_node": src,
            "dest_node": dst,
            "base_speed_mps": float(base_speed),
            "route_length_m": float(graph.route_length(route_edges)),
            "inside_length_m": float(inside_len),
            "n_route_edges": len(route_edges),
            "route_topology": stats,
            "inside_topology": inside_stats,
            "transit_duration_s": float((a_end - a_start) * dt),
        },
    )


def generate_multi_vehicle_scenario(graph, region: Region, seed: int, n_vehicles: int,
                                    overlap_target: float = 0.6,
                                    max_stagger_s: float = 12.0,
                                    **traj_kwargs) -> list[Trajectory]:
    rng = np.random.RandomState(seed + 77_000)
    dt = traj_kwargs.get("dt", 0.25)

    for attempt in range(60):
        trajs = []
        ok = True
        for v in range(n_vehicles):
            try:
                tr = generate_trajectory(
                    graph, region, seed=seed * 100 + v * 17 + attempt * 1013,
                    vehicle_id=v, trajectory_id=f"mv{seed:04d}_v{v}", **traj_kwargs)
            except RouteRejected:
                ok = False
                break
            trajs.append(tr)
        if not ok:
            continue


        offsets = [0.0] + [rng.uniform(0, max_stagger_s) for _ in range(n_vehicles - 1)]
        shifted = []
        for tr, off in zip(trajs, offsets):
            pad = int(round(off / dt))
            if pad > 0:
                xy = np.vstack([np.repeat(tr.xy[:1], pad, axis=0), tr.xy])
                sp = np.concatenate([np.zeros(pad), tr.speed])
                hd = np.concatenate([np.repeat(tr.heading[:1], pad), tr.heading])
                ed = np.concatenate([np.repeat(tr.road_edge_id[:1], pad), tr.road_edge_id])
                ts = np.arange(len(xy), dtype=float) * dt
                tr = Trajectory(
                    trajectory_id=tr.trajectory_id, vehicle_id=tr.vehicle_id,
                    timestamps=ts, xy=xy, speed=sp, heading=hd, road_edge_id=ed,
                    active_start=tr.active_start + pad, active_end=tr.active_end + pad,
                    meta={**tr.meta, "start_offset_s": float(off)})
            else:
                tr.meta["start_offset_s"] = 0.0
            shifted.append(tr)


        spans = [(t.active_start * dt, t.active_end * dt) for t in shifted]
        worst = 1.0
        for i in range(n_vehicles):
            for j in range(i + 1, n_vehicles):
                lo = max(spans[i][0], spans[j][0])
                hi = min(spans[i][1], spans[j][1])
                shortest = min(spans[i][1] - spans[i][0], spans[j][1] - spans[j][0])
                worst = min(worst, max(0.0, hi - lo) / max(shortest, 1e-6))
        if n_vehicles == 1 or worst >= overlap_target:
            for tr in shifted:
                tr.meta["min_pairwise_overlap"] = float(worst)
            return shifted

    raise RouteRejected(f"could not build a {n_vehicles}-vehicle scenario with "
                        f"overlap >= {overlap_target}")
