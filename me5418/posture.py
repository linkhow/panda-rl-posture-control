"""Seven-arm posture velocity providers and collision-surface point Jacobians.

See docs/stage04_api_sources.md. The output u is kinematic rad/s before
damped projection, never a physical force or a strict task-safety guarantee.
"""
from __future__ import annotations
import numpy as np
import pybullet as p
from .tracking import ToolJacobian


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


class LinkPointJacobian:
    def __init__(self, scene):
        self.scene = scene
        mapped = ToolJacobian(scene)
        self.indices, self.arm_columns = mapped.indices, mapped.arm_columns

    def link_frame(self, link):
        """Recover URDF link frame from double-precision COM/inertial data."""
        if link == -1:
            # Base is immobile. Its Jacobian is explicitly zero.
            position, quat = p.getBasePositionAndOrientation(self.scene.robot, physicsClientId=self.scene.cid)
            d = p.getDynamicsInfo(self.scene.robot, -1, physicsClientId=self.scene.cid)
            rc = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            ri = np.array(p.getMatrixFromQuaternion(d[4])).reshape(3, 3)
            rl = rc @ ri.T
            return np.array(position) - rl @ np.array(d[3]), rl
        s = p.getLinkState(self.scene.robot, link, computeForwardKinematics=1, physicsClientId=self.scene.cid)
        rc = np.array(p.getMatrixFromQuaternion(s[1])).reshape(3, 3)
        ri = np.array(p.getMatrixFromQuaternion(s[3])).reshape(3, 3)
        rl = rc @ ri.T
        return np.array(s[0]) - rl @ np.array(s[2]), rl

    def local_material_point(self, link, world):
        origin, rotation = self.link_frame(link)
        return rotation.T @ (np.asarray(world) - origin)

    def world_material_point(self, link, local):
        origin, rotation = self.link_frame(link)
        return origin + rotation @ np.asarray(local)

    def matrix(self, link, world):
        if link == -1:
            return np.zeros((3, 7))
        q = [s[0] for s in p.getJointStates(self.scene.robot, self.indices, physicsClientId=self.scene.cid)]
        linear, angular = p.calculateJacobian(self.scene.robot, link, [0, 0, 0],
                                              q, [0.] * 9, [0.] * 9, physicsClientId=self.scene.cid)
        # Installed Panda: localPosition=0 corresponds to URDF link origin,
        # verified independently even where the inertial translation is nonzero.
        # Shift its spatial velocity to the world surface point using rigid-body
        # kinematics; do not guess a nonzero API localPosition convention.
        origin, _ = self.link_frame(link)
        full = np.asarray(linear) - skew(np.asarray(world) - origin) @ np.asarray(angular)
        return full[:, self.arm_columns]


class ArtificialPotentialPosture:
    def __init__(self, scene, config):
        self.scene, self.config = scene, config
        self.points = LinkPointJacobian(scene)
        self.lower = np.array([scene.joints[i]["lower_limit"] for i in scene.arm])
        self.upper = np.array([scene.joints[i]["upper_limit"] for i in scene.arm])

    def joint_limit_velocity(self, q):
        c = self.config
        width = c["joint_influence_rad"]
        lower = np.clip((width - (q - self.lower)) / width, 0, 1)
        upper = np.clip((width - (self.upper - q)) / width, 0, 1)
        return c["joint_limit_max_rad_s"] * (lower**2 - upper**2)

    def compute(self, q, obstacle_detection):
        c = self.config
        nearest = {}
        # Exactly one minimum-distance point per robot collision link. Mesh
        # record count cannot multiply the repulsion strength accidentally.
        for point in obstacle_detection["closest_points"]:
            link = point["link_a"]
            if link not in nearest or point["signed_distance"] < nearest[link]["signed_distance"]:
                nearest[link] = point
        u_obstacle = np.zeros(7)
        active = []
        immobile_base_warning = False
        for link, point in sorted(nearest.items()):
            d = point["signed_distance"]
            if d >= c["influence_distance_m"]:
                continue
            if link == -1:
                immobile_base_warning = True
                continue
            normal = np.asarray(point["normal_on_b_towards_a_world"], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            # B=static sphere, A=robot: normal points from sphere toward robot.
            # Bounded, division-free quadratic avoids singularity at d=0.
            fraction = np.clip(1 - max(d, 0) / c["influence_distance_m"], 0, 1)
            strength = c["repulsion_max_rad2_per_m_s"] * fraction**2
            repulsion = strength * normal
            jac = self.points.matrix(link, point["position_on_a_world"])
            contribution = jac.T @ repulsion
            u_obstacle += contribution
            active.append({"link_index": link, "link_name": self.scene.links[link], "distance_m": d,
                           "robot_point_world_m": point["position_on_a_world"], "sphere_point_world_m": point["position_on_b_world"],
                           "normal_sphere_to_robot": normal.tolist(), "strength_rad2_per_m_s": float(strength),
                           "contribution_rad_s": contribution.tolist()})
        u_joint = self.joint_limit_velocity(np.asarray(q))
        return {"u": u_obstacle + u_joint, "u_obstacle": u_obstacle, "u_joint": u_joint,
                "active_points": active, "immobile_base_in_influence": immobile_base_warning}
