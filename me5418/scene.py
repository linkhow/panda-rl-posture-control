"""Panda scene and geometric collision queries; no tracking controller.

API references: official PyBullet Quickstart Guide (getJointInfo/getLinkState,
getClosestPoints/getContactPoints/setCollisionFilterPair) and URDF2Bullet.cpp.
See docs/stage02_api_sources.md for URLs and interpretation.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage02_scene.json"


class PandaScene:
    def __init__(self, config_path: Path = DEFAULT_CONFIG, gui: bool = False):
        self.config = copy.deepcopy(config_path) if isinstance(config_path, dict) else json.loads(Path(config_path).read_text())
        self.rng = np.random.default_rng(self.config["seed"])
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("PyBullet connection failed")
        self.closed = False
        self.reset_count = 0
        self.step_count = 0
        try:
            self._build()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            if p.isConnected(self.cid):
                p.disconnect(self.cid)
            self.closed = True

    def _build(self):
        c = self.config
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setRealTimeSimulation(0, physicsClientId=self.cid)
        p.setGravity(*c["gravity"], physicsClientId=self.cid)
        p.setTimeStep(c["time_step"], physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=c["solver_iterations"],
                                   deterministicOverlappingPairs=1, numSubSteps=0, physicsClientId=self.cid)
        # Include parents at loading, then explicitly filter only documented pairs.
        # Do not use EXCLUDE_ALL_PARENTS or MERGE_FIXED_LINKS.
        self.flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.urdf_path = Path(pybullet_data.getDataPath()) / c["model"]
        self.robot = p.loadURDF(str(self.urdf_path), basePosition=c["base_position"],
                               baseOrientation=c["base_orientation_xyzw"], useFixedBase=True,
                               flags=self.flags, globalScaling=1, physicsClientId=self.cid)
        self._map_model()
        self._configure_self_pairs()
        self.obstacle = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=c["obstacle"]["radius"], physicsClientId=self.cid),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=c["obstacle"]["radius"], rgbaColor=[0.85, 0.15, 0.12, 1], physicsClientId=self.cid),
            basePosition=c["obstacle"]["position"], physicsClientId=self.cid)
        # Visual-only floor and tool marker: no collision bodies, no support contacts.
        self.floor = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.005], rgbaColor=[0.84, 0.87, 0.91, 1], physicsClientId=self.cid),
            basePosition=[0, 0, -0.012], physicsClientId=self.cid)
        self.marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0.05, 0.8, 0.2, 1], physicsClientId=self.cid), physicsClientId=self.cid)
        self.reset()

    def _map_model(self):
        self.links = {-1: p.getBodyInfo(self.robot, physicsClientId=self.cid)[0].decode()}
        self.joints = []
        types = {p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic", p.JOINT_FIXED: "fixed"}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            a = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            self.links[i] = a[12].decode()
            self.joints.append({"joint_index": i, "joint_name": a[1].decode(), "joint_type": types.get(a[2], str(a[2])),
                "child_link_index": i, "child_link_name": self.links[i], "parent_link_index": a[16],
                "q_index": a[3], "u_index": a[4], "lower_limit": a[8], "upper_limit": a[9],
                "urdf_max_force": a[10], "urdf_max_velocity": a[11], "axis": list(a[13]),
                "parent_frame_position": list(a[14]), "parent_frame_orientation_xyzw": list(a[15])})
        self.joint_by_name = {j["joint_name"]: j for j in self.joints}
        self.link_by_name = {name: index for index, name in self.links.items()}
        self.arm = [self.joint_by_name[n]["joint_index"] for n in self.config["arm_joint_names"]]
        self.fingers = [self.joint_by_name[n]["joint_index"] for n in self.config["finger_joint_names"]]
        self.tool_link = self.link_by_name[self.config["tool"]["link_name"]]
        movable = sorted((j for j in self.joints if j["q_index"] >= 0), key=lambda j: j["q_index"])
        for j in self.joints:
            j["parent_link_name"] = self.links[j["parent_link_index"]]
            j["classification"] = "arm" if j["joint_index"] in self.arm else "finger" if j["joint_index"] in self.fingers else "fixed"
            j["movable_position_order"] = next((i for i, x in enumerate(movable) if x is j), None)
        self.shape_data = {i: p.getCollisionShapeData(self.robot, i, physicsClientId=self.cid) for i in self.links}
        self.geometry_links = [i for i in self.links if self.shape_data[i]]

    def _configure_self_pairs(self):
        # Fixed connections form one rigid assembly, even through shapeless links.
        component = {i: i for i in self.links}
        def root(i):
            while component[i] != i:
                i = component[i]
            return i
        for j in self.joints:
            if j["joint_type"] == "fixed":
                component[root(j["child_link_index"])] = root(j["parent_link_index"])
        parent_pairs = {tuple(sorted((j["child_link_index"], j["parent_link_index"]))) for j in self.joints}
        self.monitored_pairs = []
        self.pair_policy = []
        for a, b in itertools.combinations(sorted(self.links), 2):
            if a not in self.geometry_links or b not in self.geometry_links:
                reason = "no_collision_geometry"
            elif root(a) == root(b):
                reason = "same_fixed_rigid_assembly"
            elif (a, b) in parent_pairs:
                reason = "direct_joint_neighbors_attachment_overlap"
            else:
                reason = None
                self.monitored_pairs.append((a, b))
            enabled = reason is None
            self.pair_policy.append({"pair": [a, b], "link_names": [self.links[a], self.links[b]],
                                     "monitored": enabled, "excluded_reason": reason})
            p.setCollisionFilterPair(self.robot, self.robot, a, b, int(enabled), physicsClientId=self.cid)
        self.monitored_pair_set = set(self.monitored_pairs)

    def metadata(self):
        shapes = []
        for i in self.links:
            d = p.getDynamicsInfo(self.robot, i, physicsClientId=self.cid)
            shapes.append({"link_index": i, "link_name": self.links[i], "collision_shape_count": len(self.shape_data[i]),
                "mass": d[0], "local_inertial_position": list(d[3]), "local_inertial_orientation_xyzw": list(d[4]),
                "collision_margin": d[11] if len(d) > 11 else None,
                "collision_shapes": [{"geometry_type": x[2], "dimensions": list(x[3]), "mesh": x[4].decode(),
                    "local_shape_position": list(x[5]), "local_shape_orientation_xyzw": list(x[6])} for x in self.shape_data[i]]})
        return {"config": self.config, "urdf_path": str(self.urdf_path), "load_flags": self.flags,
                "load_flag_names": ["URDF_USE_SELF_COLLISION", "URDF_USE_SELF_COLLISION_INCLUDE_PARENT"],
                "joints": self.joints, "links": shapes, "self_pair_policy": self.pair_policy,
                "monitored_pair_count": len(self.monitored_pairs), "obstacle_monitored_link_indices": self.geometry_links,
                "base_included_in_obstacle_monitor": True,
                "floor": "visual only, no collision shape; excluded from task monitoring",
                "tool": {**self.config["tool"], "resolved_link_index": self.tool_link},
                "generalized_coordinate_note": "jointIndex selects a joint; linkIndex selects its child (-1 base); raw qIndex/uIndex are engine offsets. Movable order is independently sorted by qIndex and includes both fingers."}

    def reset(self, arm_positions=None):
        q = np.asarray(self.config["initial_arm_positions"] if arm_positions is None else arm_positions, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite arm angles")
        for index, value in zip(self.arm, q):
            j = self.joints[index]
            if not j["lower_limit"] <= value <= j["upper_limit"]:
                raise ValueError(f"Joint limit violation: {j['joint_name']}")
        for index, value in zip(self.arm + self.fingers, list(q) + self.config["finger_positions"]):
            p.resetJointState(self.robot, index, value, targetVelocity=0, physicsClientId=self.cid)
        self.reset_count += 1
        self.arm_targets = q.tolist()
        self._hold_motors()
        self.move_obstacle(self.config["obstacle"]["position"])
        self.refresh()

    def _hold_motors(self):
        for indices, targets, key in [(self.arm, self.arm_targets, "arm_motor"), (self.fingers, self.config["finger_positions"], "finger_motor")]:
            cfg = self.config[key]
            p.setJointMotorControlArray(self.robot, indices, p.POSITION_CONTROL, targetPositions=targets,
                targetVelocities=[0] * len(indices), forces=cfg["forces"], positionGains=[cfg["position_gain"]] * len(indices),
                velocityGains=[cfg["velocity_gain"]] * len(indices), physicsClientId=self.cid)

    def joint_positions(self):
        return np.array([x[0] for x in p.getJointStates(self.robot, self.arm + self.fingers, physicsClientId=self.cid)])

    def tool_pose(self):
        # [0:2] is world COM pose, [4:6] is world URDF-link-frame pose.
        s = p.getLinkState(self.robot, self.tool_link, computeForwardKinematics=True, physicsClientId=self.cid)
        cfg = self.config["tool"]
        xyz, quat = p.multiplyTransforms(s[4], s[5], cfg["local_position"], cfg["local_orientation_xyzw"])
        return {"position": list(xyz), "orientation_xyzw": list(quat), "frame": "world",
                "source_fields": "getLinkState[4],[5] + configured link-local tool transform"}

    def refresh(self):
        p.performCollisionDetection(physicsClientId=self.cid)
        if hasattr(self, "marker"):
            pose = self.tool_pose()
            p.resetBasePositionAndOrientation(self.marker, pose["position"], pose["orientation_xyzw"], physicsClientId=self.cid)

    def move_obstacle(self, position):
        p.resetBasePositionAndOrientation(self.obstacle, position, [0, 0, 0, 1], physicsClientId=self.cid)
        self.refresh()

    def _point(self, x):
        return {"body_a": x[1], "body_b": x[2], "link_a": x[3], "link_b": x[4],
                "link_name_a": self.links.get(x[3]) if x[1] == self.robot else "obstacle",
                "link_name_b": self.links.get(x[4]) if x[2] == self.robot else "obstacle",
                "position_on_a_world": list(x[5]), "position_on_b_world": list(x[6]),
                "normal_on_b_towards_a_world": list(x[7]), "signed_distance": x[8], "normal_force": x[9]}

    def _summarize(self, closest, contacts):
        cfg = self.config["collision"]
        points = sorted((self._point(x) for x in closest), key=lambda x: x["signed_distance"])
        d = points[0]["signed_distance"] if points else None
        tol = cfg["contact_tolerance"]
        contact_points = [self._point(x) for x in contacts]
        return {"distance_status": "within_query_range" if points else "beyond_query_range_censored",
                "query_range": cfg["query_range"], "minimum_signed_distance": d,
                "distance_lower_bound_if_censored": cfg["query_range"] if not points else None,
                "nearest": points[0] if points else None, "closest_points": points, "contact_points": contact_points,
                "contact_manifold_count": len(contact_points),
                "contact_or_penetration": any(x["signed_distance"] <= tol for x in points + contact_points),
                "penetration": any(x["signed_distance"] < -tol for x in points + contact_points),
                "within_safety_buffer": any(x["signed_distance"] < cfg["safety_buffer"] for x in points + contact_points)}

    def obstacle_query(self):
        closest = p.getClosestPoints(self.robot, self.obstacle, self.config["collision"]["query_range"], physicsClientId=self.cid)
        contacts = p.getContactPoints(self.robot, self.obstacle, physicsClientId=self.cid)
        return self._summarize(closest, contacts)

    def self_query(self):
        # Explicit distance queries can return filtered pairs: query only policy pairs.
        closest = []
        for a, b in self.monitored_pairs:
            closest.extend(p.getClosestPoints(self.robot, self.robot, self.config["collision"]["query_range"],
                linkIndexA=a, linkIndexB=b, physicsClientId=self.cid))
        contacts = [x for x in p.getContactPoints(self.robot, self.robot, physicsClientId=self.cid)
                    if tuple(sorted((x[3], x[4]))) in self.monitored_pair_set]
        data = self._summarize(closest, contacts)
        tol = self.config["collision"]["contact_tolerance"]
        # Canonical unordered pairs prevent double counting contact manifolds.
        hit_pairs = {tuple(sorted((x["link_a"], x["link_b"]))) for x in data["closest_points"] + data["contact_points"] if x["signed_distance"] <= tol}
        data["collision_pairs"] = [{"pair": list(x), "link_names": [self.links[i] for i in x]} for x in sorted(hit_pairs)]
        return data

    def detect(self):
        return {"obstacle": self.obstacle_query(), "self": self.self_query()}

    def step(self):
        # No resetJointState here. Positions evolve only through physics and motors.
        p.stepSimulation(physicsClientId=self.cid)
        self.step_count += 1
        self.refresh()
        return self.detect()

    def render(self, path: Path):
        from PIL import Image, ImageDraw
        c = self.config["camera"]
        self.refresh()
        view = p.computeViewMatrixFromYawPitchRoll(c["target"], c["distance"], c["yaw"], c["pitch"], 0, 2)
        projection = p.computeProjectionMatrixFOV(55, c["width"] / c["height"], 0.02, 5)
        image = p.getCameraImage(c["width"], c["height"], view, projection, renderer=p.ER_TINY_RENDERER,
            shadow=1, lightDirection=[-3, -4, 8], physicsClientId=self.cid)
        rgb = Image.fromarray(np.asarray(image[2], dtype=np.uint8).reshape(c["height"], c["width"], 4)).convert("RGB")
        draw = ImageDraw.Draw(rgb)
        draw.rectangle([10, 10, 530, 65], fill="white")
        draw.text((20, 18), "Panda scene | red: obstacle | green: tool point", fill="black")
        draw.text((20, 40), "TinyRenderer / collision-free markers / units: metre", fill="black")
        rgb.save(path)
        return {"renderer": "ER_TINY_RENDERER", "path": str(path), "width": c["width"], "height": c["height"],
                "tool_marker_position": self.tool_pose()["position"]}
