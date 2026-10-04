"""Versioned parameter-only scenes, seeded template groups and geometry screens."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pybullet as p
from .scene import PandaScene, ROOT
from .tracking import state, safety_reason, forbid_joint_resets, write_json, ToolJacobian
from .posture import LinkPointJacobian


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def parameter_key(scene):
    return {"q0":scene["initial_arm_q_rad"],"fingers":scene["finger_positions_m"],
            "start":scene["trajectory"]["start_world_m"],"goal":scene["trajectory"]["goal_world_m"],
            "T":scene["trajectory"]["duration_s"],"sphere":scene["sphere"],
            "execution":scene["execution"]}


def near_duplicate(a,b,tol):
    def close(x,y,bound): return np.max(np.abs(np.asarray(x)-np.asarray(y)))<=bound
    return (close(a["initial_arm_q_rad"],b["initial_arm_q_rad"],tol["q_rad"]) and
            close(a["trajectory"]["start_world_m"],b["trajectory"]["start_world_m"],tol["start_m"]) and
            close(a["trajectory"]["goal_world_m"],b["trajectory"]["goal_world_m"],tol["goal_m"]) and
            close(a["sphere"]["position_world_m"],b["sphere"]["position_world_m"],tol["sphere_m"]) and
            abs(a["sphere"]["radius_m"]-b["sphere"]["radius_m"])<=tol["radius_m"] and
            abs(a["trajectory"]["duration_s"]-b["trajectory"]["duration_s"])<=tol["duration_s"])


def execution_template(config):
    cfg=json.loads((ROOT/"configs/stage02_scene.json").read_text())
    tr=json.loads((ROOT/"configs/stage03_tracking.json").read_text())
    ap=json.loads((ROOT/"configs/stage04_posture_apf.json").read_text())
    return {"scene_config":cfg,"controller":tr["controller"],"validation":tr["validation"],
            "apf":ap["apf"],"duration_s":4.0,"settle_steps":240,
            "reference_start_rule":"240 position-hold steps, then tool_pose(); save measured start once and verify all methods within 1e-7 m",
            "reference_start_tolerance_m":1e-7,"physical_frequency_hz":240,
            "posture_frequency_hz":48,"posture_period_steps":5,"posture_cap_rad_s":.2}


def measured_start(execution):
    with PandaScene() as scene:
        scene.move_obstacle([2,2,2])
        with forbid_joint_resets():
            for _ in range(execution["settle_steps"]):
                c=scene.step()
                reason=safety_reason(scene,*state(scene),scene.tool_pose(),c,execution["validation"])
                if reason: raise RuntimeError(f"Template settling failed: {reason}")
        return scene.tool_pose()["position"],scene.metadata()


def make_scene(identifier,split,group,seed,index,execution,start,delta,sphere,category,version):
    ex=copy.deepcopy(execution)
    ex["scene_config"]["obstacle"]= {"position":list(sphere),"radius":.06}
    start=np.asarray(start); goal=start+np.asarray(delta)
    return {"id":identifier,"dataset_version":version,"schema_version":"stage05-scene-1",
            "split":split,"template_group":group,"generator_version":"surface-cone-2",
            "seed":seed,"candidate_index":index,
            "initial_arm_q_rad":ex["scene_config"]["initial_arm_positions"],
            "finger_positions_m":ex["scene_config"]["finger_positions"],
            "sphere":{"position_world_m":list(sphere),"radius_m":.06},
            "trajectory":{"start_world_m":start.tolist(),"goal_world_m":goal.tolist(),
                          "displacement_m":list(delta),"duration_s":4.0,"time_parameterization":"quintic 10u^3-15u^4+6u^5"},
            "execution":ex,"preassigned_geometric_category":category}


def validate_scene(s):
    """Check resolved parameter consistency, independently of discovery results."""
    for key in ["id","dataset_version","split","template_group","generator_version","seed","candidate_index","initial_arm_q_rad","finger_positions_m","sphere","trajectory","execution"]:
        if key not in s: raise ValueError(f"Missing scenario field {key}")
    ex=s["execution"];c=ex["scene_config"];tr=s["trajectory"]
    if len(s["initial_arm_q_rad"])!=7 or len(s["finger_positions_m"])!=2:
        raise ValueError("Seven arm plus two finger positions required")
    if not np.array_equal(s["initial_arm_q_rad"],c["initial_arm_positions"]) or not np.array_equal(s["finger_positions_m"],c["finger_positions"]):
        raise ValueError("Initial states disagree with actual scene config")
    if not np.array_equal(s["sphere"]["position_world_m"],c["obstacle"]["position"]) or s["sphere"]["radius_m"]!=c["obstacle"]["radius"]:
        raise ValueError("Sphere fields disagree with actual scene config")
    if not np.allclose(np.asarray(tr["start_world_m"])+tr["displacement_m"],tr["goal_world_m"],rtol=0,atol=1e-12):
        raise ValueError("Trajectory endpoints disagree")
    if tr["duration_s"]!=ex["duration_s"] or abs(c["time_step"]-1/240)>1e-15 or ex["posture_period_steps"]!=5:
        raise ValueError("Time/frequency configuration disagrees with frozen interface")
    if c["tool"]["link_name"]!="panda_grasptarget" or c["tool"]["local_position"]!=[0,0,0]:
        raise ValueError("Tool definition changed; revalidate the Jacobian before use")
    values=[*s["initial_arm_q_rad"],*s["finger_positions_m"],*s["sphere"]["position_world_m"],s["sphere"]["radius_m"],*tr["start_world_m"],*tr["goal_world_m"],tr["duration_s"]]
    if not np.isfinite(values).all(): raise ValueError("Non-finite scenario parameter")
    return True


def generate(config,mode,output):
    """All formal split assignments and geometry fixed BEFORE witness search."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/"candidates.json").exists(): raise FileExistsError(output)
    ex=execution_template(config); start,metadata=measured_start(ex)
    # Fully resolved rules and model limits, not merely a filename reference.
    ex["model_metadata"]={k:metadata[k] for k in ["load_flags","joints","self_pair_policy","monitored_pair_count","obstacle_monitored_link_indices","tool"]}
    seed=config[mode]["seed"]; rng=np.random.default_rng(seed)
    if any(config["sampling"]["q0_perturbation_rad"]):
        raise ValueError("This compact generator version fixes q0; perturbations need a new pilot/version")
    reserved=[]
    if mode=="formal":
        for source in config.get("reserved_development_candidates",[]):
            reserved.extend(x["parameters"] for x in json.loads((ROOT/source).read_text())["candidates"])
        dev=json.loads((ROOT/"configs/stage04_development_scenes.json").read_text())
        for i,c in enumerate(dev["candidates"]):
            reserved.append(make_scene("stage04-"+c["name"],"development","stage04",44,i,ex,c["reference_start_world_m"],c["displacement_m"],c["sphere_position_m"],c["category"],"stage04"))
    if mode=="pilot":
        groups=[("pilot",i) for i in range(config["pilot"]["candidate_count"])]
        variants=1; version=config["pilot"]["version"]
    else:
        group_splits=[]
        for split,count in config["formal"]["groups_per_split"].items(): group_splits += [split]*count
        # Fixed assignment, independent of controller outcomes.
        np.random.default_rng(config["split_seed"]).shuffle(group_splits)
        groups=[(split,i) for i,split in enumerate(group_splits)]
        variants=config["formal"]["variants_per_group"]; version=config["formal"]["version"]
    templates=[]
    with PandaScene() as scene:
        points=LinkPointJacobian(scene)
        for split,g in groups:
            direction=rng.normal(size=3); direction/=np.linalg.norm(direction)
            # Predetermined category cycles avoid outcome-defined difficulty.
            category=["far","near_link","tight_layout","near_link","tight_layout"][g%5]
            link=int(rng.choice([3,4])); axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
            templates.append({"group":f"{mode}-g{g:03d}","split":split,"direction":direction,"axis":axis,"category":category,"link":link})
        candidates=[]; previous=[]; hashes={}; index=0
        # Interleave groups so a total time budget does not favour one split.
        for variant in range(variants):
            for tpl in templates:
                d=tpl["direction"]+rng.normal(0,config["sampling"]["direction_cone_noise_sd"],3); d/=np.linalg.norm(d)
                length=rng.uniform(*config["sampling"]["length_range_m"]); delta=d*length
                if tpl["category"]=="far":
                    sphere=rng.uniform(1.5,2.5,3); construction={"type":"far cube"}
                else:
                    proxy_note=None
                    if tpl["category"]=="tight_layout" and config["sampling"].get("tight_surface_frame")=="linearized_goal_configuration":
                        # Purely geometric endpoint proxy, not a motion/witness.
                        scene.reset()
                        j=ToolJacobian(scene).matrix()
                        dq=j.T@np.linalg.solve(j@j.T+.02**2*np.eye(3),delta)
                        qproxy=np.asarray(ex["scene_config"]["initial_arm_positions"])+dq
                        scene.reset(qproxy)
                        proxy_note={"type":"one damped linearised position-Jacobian increment; static construction only, not a feasible motion", "proxy_q_rad":qproxy.tolist()}
                    axis=tpl["axis"]+rng.normal(0,config["sampling"]["surface_cone_noise_sd"],3); axis/=np.linalg.norm(axis)
                    origin,_=points.link_frame(tpl["link"])
                    scene.move_obstacle((origin+.4*axis).tolist())
                    raw=p.getClosestPoints(scene.robot,scene.obstacle,2,linkIndexA=tpl["link"],physicsClientId=scene.cid)
                    probe=min(raw,key=lambda x:x[8]); outward=-np.asarray(probe[7]); outward/=np.linalg.norm(outward)
                    gap=rng.uniform(*config["sampling"]["gap_ranges_m"][tpl["category"]])
                    sphere=np.asarray(probe[5])+(.06+gap)*outward
                    construction={"type":"initial real collision surface point plus outward normal*(radius+gap)","target_link":scene.links[tpl["link"]],"surface_point_m":list(probe[5]),"outward_normal":outward.tolist(),"sampled_gap_m":gap}
                    if proxy_note:
                        construction["type"]="linearised goal collision surface point plus outward normal*(radius+gap)"
                        construction["proxy"]=proxy_note
                    scene.reset()
                s=make_scene(f"{version}-{index:04d}",tpl["split"],tpl["group"],seed,index,ex,start,delta,sphere,tpl["category"],version)
                scene.move_obstacle(sphere.tolist()); c=scene.detect()
                reason=safety_reason(scene,*state(scene),scene.tool_pose(),c,ex["validation"])
                initial_gap=c["obstacle"]["minimum_signed_distance"]
                base=p.getClosestPoints(scene.robot,scene.obstacle,4,linkIndexA=-1,physicsClientId=scene.cid)
                base_gap=min(x[8] for x in base) if base else None
                segment=np.asarray(delta); v=np.clip(np.dot(np.asarray(sphere)-start,segment)/np.dot(segment,segment),0,1)
                tool_gap=float(np.linalg.norm(np.asarray(sphere)-(np.asarray(start)+v*segment))-.06)
                rejected=[]
                if reason: rejected.append("initialization_failure:"+reason)
                if initial_gap is not None and initial_gap<config["geometry"]["minimum_initial_gap_m"]: rejected.append("initial_gap_below_2mm")
                if base_gap is not None and base_gap<config["geometry"]["minimum_base_gap_m"]: rejected.append("immobile_base_gap_below_10mm")
                if tool_gap<config["geometry"]["minimum_tool_segment_gap_m"]: rejected.append("sphere_blocks_tool_segment_buffer")
                h=canonical_hash(parameter_key(s))
                if h in hashes: rejected.append("exact_duplicate:"+hashes[h])
                dup=next((a for a in previous if near_duplicate(s,a,config["deduplication"])),None)
                if dup: rejected.append("near_duplicate:"+dup["id"])
                dup=next((a for a in reserved if near_duplicate(s,a,config["deduplication"])),None)
                if dup: rejected.append("reserved_development_near_duplicate:"+dup["id"])
                # Compare against all previously generated candidates, including rejects.
                hashes[h]=s["id"]; previous.append(s)
                candidates.append({"parameters":s,"construction":construction,"parameter_sha256":h,
                                   "geometry":{"initial_min_gap_m":initial_gap,"base_gap_m":base_gap,"tool_segment_gap_m":tool_gap,"initial_self_gap_m":c["self"]["minimum_signed_distance"]},
                                   "status":"geometry_rejected" if rejected else "pending_search",
                                   "rejection_reasons":rejected,"feasibility_status":"not_checked",
                                   "witness_index":None,"attempts":[]})
                index+=1
    result={"version":version,"mode":mode,"protocol":config,"protocol_sha256":canonical_hash(config),
            "split_assignment":"seeded shuffle of preallocated template-group splits, before geometry and feasibility; near-duplicate checks are outcome-independent",
            "measured_reference_start_world_m":start,"candidates":candidates,
            "count":len(candidates),"group_assignments":[{"group":x["group"],"split":x["split"]} for x in templates]}
    write_json(output/"candidates.json",result)
    return result
