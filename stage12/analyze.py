"""Descriptive original/supplementary results; never simulates or selects models.

Run from a complete package with NumPy and Matplotlib. All input paths are CLI
arguments or repository-relative defaults. The analysis protocol is immutable
and checked before each run. Input CSVs are hashed; raw failures are retained.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = [
    "max_error_m", "rmse_m", "endpoint_error_m", "minimum_external_distance_m",
    "minimum_self_distance_m", "command_smoothness_rms_rad_s2",
    "max_actual_arm_velocity_rad_s", "speed_saturated_fraction",
    "soft_limited_fraction", "raw_policy_component_clip_fraction",
    "raw_policy_decisions_with_clip_fraction", "u_component_clip_fraction",
    "u_decisions_with_clip_fraction", "mean_u_abs_rad_s",
    "provider_mean_s", "core_mean_s", "outer_mean_s",
]
UNITS = {
    "max_error_m": "m", "rmse_m": "m", "endpoint_error_m": "m",
    "minimum_external_distance_m": "m", "minimum_self_distance_m": "m",
    "command_smoothness_rms_rad_s2": "rad/s^2",
    "max_actual_arm_velocity_rad_s": "rad/s", "mean_u_abs_rad_s": "rad/s",
    "provider_mean_s": "s/48Hz decision", "core_mean_s": "s/240Hz step",
    "outer_mean_s": "s/240Hz step including decision when due",
}
SEEDS = [550901, 551901, 552901]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row["success"] = row["success"].lower() == "true"
        for key in METRICS + ["completed_fraction", "first_failure_time_s"]:
            value = row.get(key)
            row[key] = float(value) if value not in ("", None, "None", "null") else None
    return rows


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "min": float(a.min()), "max": float(a.max())}


def value(row, key):
    v = row.get(key)
    return v is not None and np.isfinite(v)


def analyze_dataset(rows, dataset, primary, out):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    by = {m: {r["scene_id"]: r for r in rows if r["method"] == m} for m in methods}
    ids = list(by[methods[0]])
    assert all(set(by[m]) == set(ids) and len(by[m]) == sum(r["method"] == m for r in rows) for m in methods)
    assert all(by[m][i]["template_group"] == by[methods[0]][i]["template_group"] for m in methods for i in ids)
    groups = sorted({by[methods[0]][i]["template_group"] for i in ids})
    categories = ["far", "near_link", "tight_layout"]
    common = [i for i in ids if all(by[m][i]["success"] for m in primary)]
    group_ids = {g: [i for i in ids if by[methods[0]][i]["template_group"] == g] for g in groups}
    rng = np.random.default_rng(12042026)
    draws = rng.integers(0, len(groups), size=(5000, len(groups)))
    weights = np.array([np.bincount(x, minlength=len(groups)) for x in draws])
    group_sizes = np.array([len(group_ids[g]) for g in groups])
    draw_denominator = weights @ group_sizes
    method_table, strata, continuous, strata_continuous, failures = [], [], [], [], []
    for m in methods:
        records = [by[m][i] for i in ids]
        failure_counts = collections.Counter(r.get("failure_category") or "unspecified" for r in records if not r["success"])
        method_table.append({"dataset": dataset, "method": m, "n_scenes": len(ids), "successes": sum(r["success"] for r in records),
                             "success_rate": sum(r["success"] for r in records) / len(ids), "template_groups": len(groups),
                             "failures": len(ids) - sum(r["success"] for r in records),
                             "external_collision_band": failure_counts.get("external_collision_band", 0),
                             "self_collision_band": failure_counts.get("self_collision_band", 0),
                             "other_failures": sum(v for k, v in failure_counts.items() if k not in ("external_collision_band", "self_collision_band"))})
        for typ, values in (("category", categories), ("template_group", groups)):
            for v in values:
                selected = [r for r in records if r["geometry" if typ == "category" else "template_group"] == v]
                sf = collections.Counter(r.get("failure_category") or "unspecified" for r in selected if not r["success"])
                strata.append({"dataset": dataset, "method": m, "stratum_type": typ, "stratum": v, "n": len(selected),
                               "successes": sum(r["success"] for r in selected), "success_rate": sum(r["success"] for r in selected)/len(selected) if selected else None,
                               "external_collision_band": sf.get("external_collision_band", 0), "self_collision_band": sf.get("self_collision_band", 0),
                               "other_failures": sum(k for name, k in sf.items() if name not in ("external_collision_band", "self_collision_band"))})
        # Primary methods share a single comparable subset. Last methods get
        # separate best-vs-last paired summaries below, avoiding survivor mixing.
        if m in primary:
            for key in METRICS:
                s = summary([by[m][i][key] for i in common if value(by[m][i], key)])
                continuous.append({"dataset": dataset, "method": m, "metric": key, "unit": UNITS.get(key, "fraction"),
                                   "scope": "all_primary_common_complete_success", "n_common_complete": len(common), **s})
            for typ, values in (("category", categories), ("template_group", groups)):
                for v in values:
                    selected = [i for i in common if by[m][i]["geometry" if typ == "category" else "template_group"] == v]
                    for key in METRICS:
                        s = summary([by[m][i][key] for i in selected if value(by[m][i], key)])
                        strata_continuous.append({"dataset": dataset, "method": m, "stratum_type": typ, "stratum": v,
                                                  "metric": key, "unit": UNITS.get(key, "fraction"),
                                                  "scope": "all_primary_common_complete_success_within_stratum", "n_common_complete": len(selected), **s})
        for row in records:
            if not row["success"]:
                failures.append(row)
    fixed = "APF" if dataset == "original" else "APF_fixed"
    comparisons = [("tracking", f"PPO_best_{s}") for s in SEEDS] + [(fixed, f"PPO_best_{s}") for s in SEEDS]
    if dataset == "original":
        comparisons += [(f"PPO_best_{s}", f"PPO_last_{s}") for s in SEEDS]
    else:
        comparisons += [("tracking", fixed), ("tracking", "APF_tuned"), (fixed, "APF_tuned")]
        comparisons += [("APF_tuned", f"PPO_best_{s}") for s in SEEDS]
    pairs, paired_continuous, changes = [], [], []
    for baseline, candidate in comparisons:
        if baseline not in by or candidate not in by:
            continue
        a, b = by[baseline], by[candidate]
        gained = [i for i in ids if b[i]["success"] and not a[i]["success"]]
        lost = [i for i in ids if a[i]["success"] and not b[i]["success"]]
        joint = [i for i in ids if a[i]["success"] and b[i]["success"]]
        diff = {i: int(b[i]["success"]) - int(a[i]["success"]) for i in ids}
        group_sum = np.array([sum(diff[i] for i in group_ids[g]) for g in groups])
        resampled = (weights @ group_sum) / draw_denominator
        ci = np.quantile(resampled, [.025, .975])
        record = {"dataset": dataset, "baseline": baseline, "candidate": candidate, "n_scenes": len(ids), "n_template_groups": len(groups),
                  "gained": gained, "lost": lost, "joint_complete_success_IDs": joint,
                  "gained_n": len(gained), "lost_n": len(lost), "common_complete_n": len(joint),
                  "success_difference": sum(diff.values()) / len(ids),
                  "cluster_percentile95_low": float(ci[0]), "cluster_percentile95_high": float(ci[1]),
                  "equal_group_mean_difference": float(np.mean(group_sum/group_sizes)),
                  "bootstrap_replicates": 5000, "bootstrap_seed": 12042026, "descriptive_posthoc": True}
        pairs.append(record)
        for i in gained + lost:
            changes.append({"dataset": dataset, "baseline": baseline, "candidate": candidate, "scene_id": i,
                            "change": "gained" if i in gained else "lost", "geometry": a[i]["geometry"], "template_group": a[i]["template_group"],
                            "baseline_failure": a[i].get("failure_category"), "candidate_failure": b[i].get("failure_category")})
        for key in METRICS:
            usable = [i for i in joint if value(a[i], key) and value(b[i], key)]
            deltas = {i: b[i][key]-a[i][key] for i in usable}
            nums = np.array([sum(deltas[i] for i in group_ids[g] if i in deltas) for g in groups])
            counts = np.array([sum(i in deltas for i in group_ids[g]) for g in groups])
            den = weights @ counts
            sample = (weights @ nums)[den > 0] / den[den > 0]
            bounds = np.quantile(sample, [.025, .975]) if len(sample) else [None, None]
            paired_continuous.append({"dataset": dataset, "baseline": baseline, "candidate": candidate, "metric": key,
                                      "unit": UNITS.get(key, "fraction"), "n_common_complete": len(joint), "n_with_both_values": len(usable),
                                      "baseline_mean": float(np.mean([a[i][key] for i in usable])) if usable else None,
                                      "candidate_mean": float(np.mean([b[i][key] for i in usable])) if usable else None,
                                      "mean_paired_difference": float(np.mean(list(deltas.values()))) if usable else None,
                                      "cluster_percentile95_low": float(bounds[0]) if bounds[0] is not None else None,
                                      "cluster_percentile95_high": float(bounds[1]) if bounds[1] is not None else None,
                                      "bootstrap_defined_replicates": int(len(sample))})
    result = {"dataset": dataset, "n_scenes": len(ids), "template_groups": groups, "methods": method_table,
              "strata": strata, "primary_methods": primary, "all_primary_common_success_IDs": common,
              "common_success_metrics": continuous, "strata_common_success_metrics": strata_continuous, "paired": pairs, "paired_continuous": paired_continuous,
              "failure_counts": dict(collections.Counter(r.get("failure_category") or "unspecified" for r in failures)),
              "interval_interpretation": "Descriptive template-cluster resampling sensitivity; no claim of untouched task family, population coverage, significance, or independent seeds as scenarios"}
    write_json(out / f"{dataset}_analysis.json", result)
    for label, data in (("methods", method_table), ("strata", strata), ("continuous_common_primary", continuous),
                        ("strata_continuous", strata_continuous), ("paired_continuous", paired_continuous), ("paired_changes", changes), ("failures", failures)):
        write_csv(out / f"{dataset}_{label}.csv", data)
    write_csv(out / f"{dataset}_paired_success.csv", [{k:v for k,v in x.items() if not isinstance(v,list)} for x in pairs])
    return result


def plot_results(results, out):
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "me5418-stage12-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                               "axes.spines.top": False, "axes.spines.right": False,
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"tracking":"#697785", "APF":"#bd843f", "APF_fixed":"#bd843f", "APF_tuned":"#e2ad55",
              "PPO_best_550901":"#468cc0", "PPO_best_551901":"#69a77e", "PPO_best_552901":"#896ab1"}
    labels = lambda m: m.replace("PPO_best_", "PPO ").replace("PPO_last_", "last ").replace("APF_fixed", "fixed APF").replace("APF_tuned", "tuned APF")
    fig, axs = plt.subplots(1, len(results), figsize=(7.2, 2.45), squeeze=False)
    for ax, res in zip(axs[0], results):
        ms = res["primary_methods"]
        x = np.arange(len(ms));vals=[next(v for v in res["methods"] if v["method"]==m) for m in ms]
        ax.bar(x, [v["success_rate"]*100 for v in vals], color=[colors[m] for m in ms])
        for pos,v in zip(x,vals): ax.text(pos, v["success_rate"]*100+1, f'{v["successes"]}/{v["n_scenes"]}', ha="center", fontsize=7)
        ax.set_ylim(0, 109);ax.set_ylabel("Complete success (%)");ax.set_xticks(x, [labels(m) for m in ms], rotation=30, ha="right")
        ax.set_title(f'{res["dataset"].capitalize()}: {res["n_scenes"]} scenes, {len(res["template_groups"])} groups')
    fig.tight_layout();fig.savefig(out/"success_original_supplementary.png", dpi=220);fig.savefig(out/"success_original_supplementary.pdf");plt.close(fig)
    res=results[0]; ms=res["primary_methods"]
    groups=res["template_groups"]
    matrix=np.array([[next(x for x in res["strata"] if x["method"]==m and x["stratum"]==g and x["stratum_type"]=="template_group")["success_rate"] for m in ms] for g in groups])
    ns=[next(x for x in res["strata"] if x["method"]==ms[0] and x["stratum"]==g and x["stratum_type"]=="template_group")["n"] for g in groups]
    fig,ax=plt.subplots(figsize=(4.8,4.9));im=ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu",aspect="auto")
    ax.set_yticks(np.arange(len(groups)),[f'{g.replace("formal-", "")}, n={n}' for g,n in zip(groups,ns)])
    ax.set_xticks(np.arange(len(ms)),[labels(m) for m in ms],rotation=30,ha="right")
    for j,n in enumerate(ns):
        for k in range(len(ms)):ax.text(k,j,str(int(round(matrix[j,k]*n))),ha="center",va="center",color="white" if matrix[j,k]>.65 else "black",fontsize=7)
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label("Complete success fraction; cell = count")
    fig.tight_layout();fig.savefig(out/"original_template_success.png",dpi=220);fig.savefig(out/"original_template_success.pdf");plt.close(fig)
    for res in results:
        ms=res["primary_methods"]
        fig,axs=plt.subplots(1,2,figsize=(7.2,2.4))
        keys=["command_smoothness_rms_rad_s2","raw_policy_component_clip_fraction"]
        for ax,key in zip(axs,keys):
            selected=[m for m in ms if not key.startswith("raw_policy") or m.startswith("PPO")]
            values=[next(v for v in res["common_success_metrics"] if v["method"]==m and v["metric"]==key) for m in selected]
            ax.bar(range(len(selected)),[v["mean"]*(100 if key.startswith("raw_policy") else 1) for v in values], color=[colors[m] for m in selected])
            ax.set_xticks(range(len(selected)),[labels(m) for m in selected],rotation=30,ha="right")
            ax.set_ylabel("RMS 7D command derivative (rad/s²)" if not key.startswith("raw_policy") else "Raw action component clips (%)")
            ax.set_title(f'Common complete successes, n={len(res["all_primary_common_success_IDs"])}')
        fig.suptitle(res["dataset"].capitalize(),fontsize=9)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'{res["dataset"]}_smoothness_clipping.png',dpi=220);fig.savefig(out/f'{res["dataset"]}_smoothness_clipping.pdf');plt.close(fig)


def trace_analysis(rows, raw_root, out):
    methods = ["tracking", "APF"] + [f"PPO_best_{s}" for s in SEEDS]
    by={m:{r["scene_id"]:r for r in rows if r["method"]==m} for m in methods}
    common=[i for i in by["tracking"] if all(by[m][i]["success"] for m in methods)]
    records=[]
    for m in methods:
        for identifier in common:
            path=raw_root/m/identifier/"states.csv"
            if not path.exists():raise FileNotFoundError(path)
            with path.open() as f:
                reader=csv.reader(f);header=next(reader)
                idx={k:header.index(k) for k in ["command_applied","secondary_leakage_m_s","posture_updated","time_s"] + [f"command{i}" for i in range(7)]}
                leakage=[];commands=[];updated=[]
                for r in reader:
                    if r[idx["command_applied"]]!="1":continue
                    leakage.append(float(r[idx["secondary_leakage_m_s"]]))
                    commands.append([float(r[idx[f"command{j}"]]) for j in range(7)])
                    updated.append(r[idx["posture_updated"]]=="1")
            cmd=np.asarray(commands);jump=np.diff(np.vstack([np.zeros((1,7)),cmd]),axis=0);updated=np.asarray(updated)
            # Match the original definition: exclude initial zero-to-command.
            magnitudes=np.sqrt(np.sum((jump[1:]*240)**2,axis=1));updated=updated[1:]
            expected=by[m][identifier]["command_smoothness_rms_rad_s2"]
            # Episode summary excludes the initial transition from zero.
            measured=float(np.sqrt(np.mean(np.sum((np.diff(cmd,axis=0)*240)**2,axis=1))))
            assert np.isclose(measured,expected,atol=1e-12,rtol=1e-10),(m,identifier,measured,expected)
            records.append({"method":m,"scene_id":identifier,"template_group":by[m][identifier]["template_group"],"geometry":by[m][identifier]["geometry"],
                            "n_applied_physical_steps":len(cmd),"mean_secondary_leakage_m_s":float(np.mean(leakage)),"max_secondary_leakage_m_s":float(np.max(leakage)),
                            "command_derivative_rms_rad_s2_recomputed":measured,
                            "mean_derivative_magnitude_on_provider_updates_rad_s2":float(np.mean(magnitudes[updated])),
                            "mean_derivative_magnitude_on_held_steps_rad_s2":float(np.mean(magnitudes[~updated])),
                            "raw_policy_component_clip_fraction":by[m][identifier].get("raw_policy_component_clip_fraction"),
                            "speed_saturated_fraction":by[m][identifier]["speed_saturated_fraction"],"soft_limited_fraction":by[m][identifier]["soft_limited_fraction"],
                            "states_csv_SHA256":sha(path)})
    aggregates={}
    for m in methods:
        rr=[r for r in records if r["method"]==m]
        aggregates[m]={"n_common_complete":len(rr)}
        for key in ["mean_secondary_leakage_m_s","max_secondary_leakage_m_s","mean_derivative_magnitude_on_provider_updates_rad_s2","mean_derivative_magnitude_on_held_steps_rad_s2"]:
            aggregates[m][key]=summary([r[key] for r in rr])
        clip=np.array([r["raw_policy_component_clip_fraction"] for r in rr if r["raw_policy_component_clip_fraction"] is not None])
        smooth=np.array([r["command_derivative_rms_rad_s2_recomputed"] for r in rr if r["raw_policy_component_clip_fraction"] is not None])
        aggregates[m]["clip_smoothness_episode_Pearson_descriptive"]=float(np.corrcoef(clip,smooth)[0,1]) if len(clip)>1 and np.std(clip)>0 else None
    write_csv(out/"original_trace_diagnostics.csv",records)
    write_json(out/"original_trace_diagnostics.json",{"scope":"Same 104 jointly complete original primary successes; complete states read only, no simulation, no causal ablation", "methods":aggregates,
                                                       "all_smoothness_recomputations_match":True,"definition":"RMS Euclidean norm in seven joint dimensions of adjacent executed joint velocity command differences divided by 1/240 s; initial zero-to-first-command jump, mode switch, and terminal stop excluded to match original metric",
                                                       "correlation_warning":"Within-method scene-level Pearson association is descriptive and template-dependent; clipping is not proven to cause all roughness"})
    return aggregates


def pooled_original_metadata(original_csv, out):
    """Original published timing and raw action counts, including partial runs.

    These are compute diagnostics over observed decisions/steps and are never
    treated as performance means for failures that stopped early.
    """
    sources = {}; rows = []
    for filename in ("primary_results.json", "secondary_results.json"):
        path = original_csv.parent / filename
        if not path.exists():
            continue
        sources[filename + "_SHA256"] = sha(path)
        for method, result in json.loads(path.read_text()).items():
            n = result["actual_provider_decisions"]
            clip = result["raw_policy_component_clip_fraction_pooled"]
            anyclip = result["raw_policy_decisions_with_clip_fraction_pooled"]
            uclip = result["u_component_clip_fraction_pooled"]
            uanyclip = result["u_decisions_with_clip_fraction_pooled"]
            for scope, timing in result["timing_pooled_all_observed"].items():
                rows.append({"method": method, "scope": scope,
                             "n_timing_samples": timing["n"], "mean_s": timing["mean_s"],
                             "p95_s": timing["p95_s"], "p99_s": timing["p99_s"], "max_s": timing["max_s"],
                             "provider_decisions_all_observed": n, "raw_policy_components_denominator": 7*n if clip is not None else None,
                             "raw_policy_components_clipped": int(round(clip*7*n)) if clip is not None else None,
                             "raw_policy_component_clip_fraction": clip,
                             "raw_policy_decisions_with_clip": int(round(anyclip*n)) if anyclip is not None else None,
                             "raw_policy_decisions_with_clip_fraction": anyclip,
                             "posture_u_component_clip_fraction": uclip,
                             "posture_u_decisions_with_clip_fraction": uanyclip,
                             "scope_caveat": "All observed decisions/steps including failure prefixes; descriptive compute/clipping, not complete-episode motion comparison"})
    write_csv(out / "original_pooled_compute_clipping.csv", rows)
    write_json(out / "original_pooled_compute_clipping.json", {"sources": sources, "rows": rows})
    return sources


def tuning_analysis(csv_path, summary_path, out):
    rows=read_csv(csv_path);meta=json.loads(summary_path.read_text())
    assert meta["complete"]
    methods=list(dict.fromkeys(r["method"] for r in rows))
    by={m:{r["scene_id"]:r for r in rows if r["method"]==m} for m in methods}
    ids=list(by[methods[0]])
    assert all(set(by[m])==set(ids) and len(by[m])==sum(r["method"]==m for r in rows) for m in methods)
    common=[i for i in ids if all(by[m][i]["success"] for m in methods)]
    reports={r["id"]:r for r in meta["reports"]}
    table=[]
    for method in methods:
        rr=[by[method][i] for i in common];rep=reports[method];params=rep["parameters"]
        row={"candidate":method,"selected":method==meta["selected"]["id"],"n_validation":len(ids),"successes":sum(by[method][i]["success"] for i in ids),
             "d0_m":params["influence_distance_m"],"repulsion_gain_rad2_per_m_s":params["repulsion_max_rad2_per_m_s"],"n_common_complete":len(common)}
        assert row["successes"]==rep["successes"]
        for key in ["max_error_m","rmse_m","minimum_external_distance_m","minimum_self_distance_m","command_smoothness_rms_rad_s2","u_component_clip_fraction","u_decisions_with_clip_fraction","mean_u_abs_rad_s","provider_mean_s"]:
            st=summary([r[key] for r in rr if value(r,key)])
            row[key+"_n"]=st["n"];row[key+"_mean"]=st["mean"]
        successful=[by[method][i]["rmse_m"] for i in ids if by[method][i]["success"]]
        assert np.isclose(np.mean(successful),rep["mean_complete_success_rmse_m"],atol=1e-15,rtol=1e-12)
        table.append(row)
    ordered=sorted(table,key=lambda r:(-r["successes"],r["rmse_m_mean"],reports[r["candidate"]]["grid_index"]))
    assert ordered[0]["candidate"]==meta["selected"]["id"]
    result={"n_candidates":len(methods),"n_validation_scenarios":len(ids),"n_episodes":len(rows),"n_common_complete":len(common),
            "selected":meta["selected"]["id"],"selection_rule":meta["selection_rule"],"wall_s":meta["wall_s"],
            "failed_episode_count":sum(not r["success"] for r in rows),"all_candidates_primary_success_tied":len({r["successes"] for r in table})==1,
            "validation_success_saturation_caveat":"These validation scenarios do not distinguish candidates by success; the predeclared RMS tie-break selects weaker/shorter-range repulsion, without implying improved supplementary success",
            "table":table,"validation_tuning_csv_SHA256":sha(csv_path),"tuning_summary_SHA256":sha(summary_path)}
    write_json(out/"validation_grid_analysis.json",result);write_csv(out/"validation_grid_analysis.csv",table)
    return result


def report_numbers(results, out, tuning=None):
    bundle={"version":"stage12-bilingual-shared-numbers-v1", "datasets":{},
            "metric_scope":"continuous = joint complete successes of all primary methods in the named dataset; separate nonmissing counts",
            "rounding":"Display does not change raw values; percentages two decimals, mm four decimals, provider microseconds two decimals",
            "smoothness_definition":"sqrt(mean_t ||(command_t-command_t-1)/(1/240 s)||_2^2), rad/s^2; initial mode switch/terminal stop excluded",
            "no_seed_pooling_as_independent_scenes":True}
    for res in results:
        ds={"n_scenes":res["n_scenes"],"n_groups":len(res["template_groups"]),"n_common_complete":len(res["all_primary_common_success_IDs"]),"methods":[]}
        for method in res["methods"]:
            row=dict(method)
            row["success_percent"]=method["success_rate"]*100
            row["categories"]={x["stratum"]:{"n":x["n"],"successes":x["successes"],"external_failures":x["external_collision_band"],"self_failures":x["self_collision_band"]}
                               for x in res["strata"] if x["method"]==method["method"] and x["stratum_type"]=="category"}
            row["common_complete_metrics"]={}
            for met in res["common_success_metrics"]:
                if met["method"]!=method["method"]:continue
                key=met["metric"];v=dict(met)
                timing_key=key in ("provider_mean_s","core_mean_s","outer_mean_s")
                factor=1000 if key.endswith("_m") else 1e6 if timing_key else 100 if key.endswith("_fraction") else 1
                v["display_mean"]=met["mean"]*factor if met["mean"] is not None else None
                v["display_unit"]="mm" if key.endswith("_m") else "microseconds" if timing_key else "%" if key.endswith("_fraction") else met["unit"]
                row["common_complete_metrics"][key]=v
            ds["methods"].append(row)
        ds["paired"]=[{k:v for k,v in x.items() if not isinstance(v,list)} for x in res["paired"]]
        bundle["datasets"][res["dataset"]]=ds
    if tuning is not None:bundle["validation_grid"]=tuning
    write_json(out/"report_numbers.json",bundle)


def main():
    original_default=ROOT/"results/reference/episodes_all1048.csv"
    if not original_default.exists():original_default=ROOT/"results/final/episodes_all1048.csv"
    p=argparse.ArgumentParser();p.add_argument("--original",type=Path,default=original_default)
    p.add_argument("--supplementary",type=Path,default=HERE/"results/episodes_new600.csv")
    p.add_argument("--tuning",type=Path,default=HERE/"results/validation_tuning.csv");p.add_argument("--tuning-summary",type=Path,default=HERE/"results/tuning_summary.json")
    p.add_argument("--raw-original",type=Path);p.add_argument("--output",type=Path,default=HERE/"analysis");p.add_argument("--skip-figures",action="store_true")
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((HERE/"analysis/protocol.json").read_text());assert protocol["bootstrap"]["replicates"]==5000 and protocol["bootstrap"]["seed"]==12042026
    original_rows=read_csv(args.original);original_primary=["tracking","APF"]+[f"PPO_best_{s}" for s in SEEDS]
    results=[analyze_dataset(original_rows,"original",original_primary,out)]
    sources={"original_episodes_SHA256":sha(args.original),"analysis_protocol_SHA256":sha(HERE/"analysis/protocol.json"),"analysis_source_SHA256":sha(Path(__file__)),"numpy_version":np.__version__}
    sources.update(pooled_original_metadata(args.original,out))
    tuning=tuning_analysis(args.tuning,args.tuning_summary,out) if args.tuning.exists() and args.tuning_summary.exists() else None
    if tuning:
        sources["validation_tuning_csv_SHA256"]=sha(args.tuning);sources["tuning_summary_SHA256"]=sha(args.tuning_summary)
    if args.supplementary.exists():
        supplement=read_csv(args.supplementary);primary=["tracking","APF_fixed","APF_tuned"]+[f"PPO_best_{s}" for s in SEEDS]
        results.append(analyze_dataset(supplement,"supplementary",primary,out));sources["supplementary_episodes_SHA256"]=sha(args.supplementary)
    if args.raw_original:trace_analysis(original_rows,args.raw_original,out)
    if not args.skip_figures:plot_results(results,out)
    report_numbers(results,out,tuning)
    write_json(out/"analysis_provenance.json",sources)
    print(json.dumps({"datasets":[{"dataset":r["dataset"],"scenes":r["n_scenes"],"groups":len(r["template_groups"]),"common_primary_complete":len(r["all_primary_common_success_IDs"]),"successes":{m["method"]:m["successes"] for m in r["methods"]}} for r in results],"output":str(out)},indent=2))


if __name__=="__main__":main()
