<!-- page 1 -->
## Learning auxiliary posture control near obstacles

ME5418 · Group 44 · Final submission candidate · 4 October 2026

## Abstract

A fixed-base seven-joint Panda tracks a four-second Cartesian line while avoiding a static sphere. A shared damped least-squares tracker handles position; APF or PPO supplies bounded auxiliary joint velocity. On the original opened 131-scene test, tracking and fixed APF complete 106 and 123 scenes; three validation-selected PPO best models complete 129, 127 and 131. This supports an advantage over that fixed baseline, with seed-dependent motion costs. A new finite validation search freezes APF at d0=0.06 m, fmax=2 rad²/(m·s). On 100 newly generated, replay-witnessed scenes, successes are 93/99/100/98/98/99, in the order tracking/fixed APF/tuned APF/PPO550901/551901/552901. The new experiment is supplemental generalization in a development-informed task family. Complete failures, paired changes, template-group uncertainty and independent environment checks are retained. No new scientific PPO training is added.

## 1. Motivation and contribution

A tool can clear an obstacle while the elbow collides. Redundancy offers configuration choices, but reactive repulsion and a learned policy make different local trade-offs. The useful question is therefore whether learned auxiliary motion improves completion under matched permissions, and what tracking, clearance, smoothness and computation it costs. The contribution is an auditable control comparison and reproducible pipeline, rather than a new RL algorithm.

The completed project adds a tuned conventional baseline, a separately frozen supplemental protocol, a full portable course package and a more cautious explanation of success-versus-cost trade-offs. Text is rewritten from executed evidence; the proposal and peer project are contextual references, not result sources.

<!-- page 2 -->
## 2. Task and shared control contract

Panda has seven actuated arm joints, position-held fingers (0.02 m each) and a fixed base. One static sphere of radius 0.06 m is placed near a 0.03-0.08 m straight tool-position path. All scenes share the original initial arm configuration. The reference uses quintic timing over T=4 s. Orientation, grasping, dynamic obstacles, perception and hardware are outside this task.

$x_r(t)=x_0+(10s^3-15s^4+6s^5)(x_g-x_0),\quad s=t/T$
Equation 1. Fixed-time reference (m, s).

$J_\lambda^{+}=J^T(JJ^T+\lambda^2I)^{-1},\quad N_\lambda=I-J_\lambda^{+}J$
Equation 2. Damped inverse and approximate null-space map.

$\dot q_c=\mathrm{limit}\{J_\lambda^{+}(\dot x_r+K_p(x_r-x))+N_\lambda u\}$
Equation 3. Common tracker plus auxiliary posture velocity.

Kp=4 s⁻¹, lambda=0.02 m/rad. Auxiliary u is clipped per joint to ±0.2 rad/s at 48 Hz and held across five 240 Hz motor/safety steps. The combined arm command is uniformly scaled to 1 rad/s, then projected at a 0.05 rad soft limit margin. All methods use the same URDF effort limits and velocity motors. N_lambda is damped, so JN_lambda is generally nonzero; auxiliary control can leak into position.

Complete success requires the full 4 s, tracking error at every sample ≤0.01 m, no obstacle/self contact band (signed distance ≤0.0001 m), no hard-position or URDF-speed violation and finite state. Forty-five configured self-link pairs are monitored. A 0.1 mm contact band is distinct from negative penetration and the wider self buffer. First failure latches and stops motors; endpoints of incomplete runs remain null.

<!-- page 3 -->
## 3. APF and reinforcement learning definition

$u_{APF}=\sum_l J_l^T n_l f_{max}[\mathrm{clip}(1-\max(d_l,0)/d_0,0,1)]^2+u_{joint}$
Equation 4. Bounded APF; d in m, u in rad/s.

Each collision link contributes one minimum signed surface distance and world normal, preventing mesh record count from multiplying force. The joint term starts within 0.3 rad of a limit and has a 0.1 rad/s cap. Fixed APF uses d0=0.10 m and fmax=4 rad²/(m·s). This is a bounded APF-inspired implementation [1], not the singular inverse-distance potential in all APF formulations. It has no self-repulsion term.

At 48 Hz, the RL transition observes 91 manually scaled float 32 features: 41 base features (q/v, position error, reference velocity, time fraction, future relative references at 0.2/0.5/1 s, sphere geometry, previous command), plus 10 links × 5 (distance, 3-normal, validity). No ID, split, template or feasibility witness enters features. Distances beyond the 0.3 m query are censored with a validity flag. There is no VecNormalize.

$u=0.2\,\mathrm{clip}(a_{raw},-1,1),\quad a_{raw}\in\mathbb{R}^7$
Equation 5. Gaussian policy output to bounded physical action.

$L^{CLIP}=\mathbb{E}_t[\min(\rho_t\hat A_t,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]$
Equation 6. PPO clipped surrogate; rho is the policy probability ratio.

PPO [2,4] uses separate 64×64 Tanh MLP policy/value networks; Gaussian initial std=1, entropy coefficient 0, value coefficient 0.5 and gradient norm cap 0.5. On-policy rollout advantages feed a clipped policy surrogate and value loss. Learning rate 3 e-4, gamma .995, GAE .95, clip .2, batch 256 and 10 epochs. Four CPU workers each collect 512 policy interactions per rollout (2048 total). Discounting uses policy time, not each physics step. Deterministic evaluation uses the Gaussian mean, clipped in float 32 exactly as the library predict path.

Dense reward integrates per physics step: progress +1/s; normalized squared tracking −1/s; bounded external proximity −0.5/s within 0.05 m; mean normalized action² −0.02/s; clipped command-jump² −0.02/s (scale 0.2 rad/s). Success/failure adds ±20. Self-clearance has no dense reward; self collision provides a terminal failure signal. Task completion/failure and the intrinsic 4 s horizon terminate; an external cap truncates without a success bonus [5].

<!-- page 4 -->
## 4. Separate frozen experimental protocols

Original: 340 accepted scenarios, train151/validation58/test131, with 20 test template groups and category counts far36/near62/tight33. Feasibility used actual motor witnesses plus replay. Three formal seeds550901/551901/552901 each trained 1,024,000 interactions; total 3,072,000. Best was selected from 10 checkpoints spaced 102,400 steps by validation complete success, discounted return, then earlier time. Selected steps409600/512000/102400; all last models retained as secondary. The opened test was run once for 8 methods (1048 episodes), not used for subsequent APF selection.

New Stage 12: before physics, save 9 candidates d0∈{.06,.10,.14} m × fmax∈{2,4,8}, all shared joint terms and control permissions. Run all 58 validation IDs for each candidate. Rank complete-success count, then mean successful RMSE, then fixed grid index; retain every failed prefix. Freeze the selected parameters and evidence hashes before generating new scenes. Each phase has a predeclared 1800 s cap: search 522 episodes; generation ≤864 attempts+288 replays; comparison 600 episodes.

After APF freeze, seed 2026100412 creates 288 candidates across 24 groups in variant-major order. Geometry gates and exact/near-duplicate checks include 755 earlier candidate geometries (including accepted 340). The first 100 with successful motor execution and replay are accepted. Witness providers are predeclared fixed/gentle/early APF, excluding tuned APF and PPO. Their solver family still biases acceptance; failed finite search does not prove infeasibility. Witnesses remain outside policy inputs.

Primary measures are complete success and same-ID gains/losses per seed. Continuous comparisons use common complete successes; null clearance stays censored. A paired whole-template bootstrap uses 5000 draws and RNG 12042026, keeping variants and all methods together. Report episode-weighted differences and equal-group sensitivity. The old 20 and new 19 groups, plus construction dependence, make intervals descriptive rather than distribution-free significance guarantees [6].

<!-- page 5 -->
## 5. Original experiment: preserved results

| Method / 方法 | Success | Rate % | External | Self | F/N/T |
| --- | --- | --- | --- | --- | --- |
| tracking | 106/131 | 80.92 | 25 | 0 | 36/52/18 |
| APF | 123/131 | 93.89 | 8 | 0 | 36/57/30 |
| best 550901 | 129/131 | 98.47 | 2 | 0 | 36/62/31 |
| best 551901 | 127/131 | 96.95 | 3 | 1 | 36/60/31 |
| best 552901 | 131/131 | 100.00 | 0 | 0 | 36/62/33 |
| last 550901 | 125/131 | 95.42 | 6 | 0 | 36/61/28 |
| last 551901 | 129/131 | 98.47 | 2 | 0 | 36/62/31 |
| last 552901 | 125/131 | 95.42 | 2 | 4 | 36/62/27 |
Table 1. Same 131 tasks; external/self are first-failure counts. F/N/T successes out of36/62/33.

| Best seed | Gain | Loss | Δ pp | Group95% pp |
| --- | --- | --- | --- | --- |
| best 550901 | 7 | 1 | 4.58 | [-1.74, 14.29] |
| best 551901 | 6 | 2 | 3.05 | [-2.31, 10.07] |
| best 552901 | 8 | 0 | 6.11 | [0.00, 16.00] |
Table 2. Against original fixed APF; 20-group paired bootstrap.

All best seeds exceed fixed APF in counts, but none is a universal advantage:550901 loses 0032;551901 loses0032/0067 while gaining other IDs. Best→last gains/losses are0/4,3/1,0/6, so 551901 last improves by 2 yet remains secondary. Model roles never change after test. Equal-group best-minus-APF differences are 0.125,−0.708,6.375 percentage points, showing that weighting and tiny groups affect interpretation.

![](docs/reports/stage12/assets/template_heatmap.png)
Figure 1. Per-template success; variants are correlated, not independent seeds.

<!-- page 6 -->
## 6. New validation-tuned APF and 100-scene comparison

| d0 m | fmax | Success | RMSE mm |
| --- | --- | --- | --- |
| 0.06 | 2 | 58/58 | 0.04230 |
| 0.06 | 4 | 58/58 | 0.04394 |
| 0.06 | 8 | 58/58 | 0.04437 |
| 0.10 | 2 | 58/58 | 0.04542 |
| 0.10 | 4 | 58/58 | 0.04751 |
| 0.10 | 8 | 58/58 | 0.04947 |
| 0.14 | 2 | 58/58 | 0.04900 |
| 0.14 | 4 | 58/58 | 0.05165 |
| 0.14 | 8 | 58/58 | 0.05390 |
Table 3. Complete finite validation grid; successful-only tie-break metric.

All 9 candidates succeed58/58; validation success is saturated. Selected by RMSE: d0=0.06 m, fmax=2 rad²/(m·s). 522 validation episodes: 424.69 s; generation: 101 motor attempts + 100 replays, 204.07 s; six-method comparison: 600 episodes, 512.63 s. Accepted categories: far 41, near 49, tight 10. Witnesses are 99 fixed APF and 1 gentle APF, a strong solver-family filter. All 100 replay witnesses pass; maximum state/command difference 0. Audit confirms zero exact/near duplicates and zero overlap with the reserved 755; failures are retained.

| Method / 方法 | Success | Rate % | External | Self | F/N/T |
| --- | --- | --- | --- | --- | --- |
| tracking | 93/100 | 93.00 | 7 | 0 | 41/44/8 |
| fixed APF | 99/100 | 99.00 | 1 | 0 | 41/49/9 |
| tuned APF | 100/100 | 100.00 | 0 | 0 | 41/49/10 |
| best 550901 | 98/100 | 98.00 | 2 | 0 | 41/47/10 |
| best 551901 | 98/100 | 98.00 | 2 | 0 | 41/47/10 |
| best 552901 | 99/100 | 99.00 | 1 | 0 | 41/48/10 |
Table 4. New 100 shared scenes; distinct from original test 131. F/N/T successes out of41/49/10.

| Best seed | Gain | Loss | Δ pp | Group95% pp |
| --- | --- | --- | --- | --- |
| best 550901 | 0 | 2 | -2.00 | [-5.63, 0.00] |
| best 551901 | 0 | 2 | -2.00 | [-7.32, 0.00] |
| best 552901 | 0 | 1 | -1.00 | [-3.66, 0.00] |
Table 5. Against frozen tuned APF; paired 19-group bootstrap; all differences are sample-conditioned.

On this screened supplement the three PPO best models complete98/98/99, below tuned APF100/100. The original advantage over fixed APF is not evidence of superiority over a tuned baseline. It does not justify retuning on these 100, changing success rules, replacing seeds or retraining to restore a preferred ranking. APF-family witness screening limits generalization to this accepted population.

<!-- page 7 -->
## 7. Motion costs, clipping and timing

| Method | RMSE mm | Gap mm | Smooth rad/s² | Clip % | Outer ms |
| --- | --- | --- | --- | --- | --- |
| tracking | 0.0350 | 19.95 | 0.034 | NA | 0.628 |
| APF | 0.0446 | 25.76 | 0.127 | NA | 0.648 |
| best 550901 | 0.0503 | 26.60 | 0.370 | 35.53 | 0.706 |
| best 551901 | 0.0459 | 26.55 | 0.466 | 48.70 | 0.712 |
| best 552901 | 0.0421 | 26.38 | 0.116 | 8.15 | 0.702 |
Table 6. Original all-primary common complete n=104; clearance non-null n=68.

| Method | RMSE mm | Gap mm | Smooth rad/s² | Clip % | Outer ms |
| --- | --- | --- | --- | --- | --- |
| tracking | 0.0366 | 18.80 | 0.034 | NA | 0.612 |
| fixed APF | 0.0467 | 23.47 | 0.133 | NA | 0.631 |
| tuned APF | 0.0401 | 23.12 | 0.085 | NA | 0.625 |
| best 550901 | 0.0549 | 23.47 | 0.413 | 44.08 | 0.680 |
| best 551901 | 0.0385 | 23.45 | 0.341 | 55.11 | 0.681 |
| best 552901 | 0.0438 | 22.43 | 0.130 | 10.73 | 0.672 |
Table 7. New all-six common complete n=91; clearance non-null n=50.

$S=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K-1}\left\|\frac{\dot q_{c,k}-\dot q_{c,k-1}}{\Delta t}\right\|_2^2}$
Equation 7. Command-vector derivative RMS (rad/s²); not measured acceleration; excludes initial zero-to-command and final stop jumps.

Tables report per-episode mean summaries, not failed prefixes. Clip% is raw Gaussian components outside[-1,1], denominator 7×policy decisions; APF has no Gaussian and is NA. On the same 104 common successes, original per-episode mean any-clip decision percentages are88.61/99.81/54.58 for best550/551/552; these use a different denominator. Speed/soft-limit saturation is 0 on original 104 shared successes, not a guarantee on arbitrary scenes.

Provider means are 48 Hz decisions; outer means are 240 Hz loops including observation, provider when due, tracker, recording, motors, simulation and safety. Setup/settle/render/export are excluded; wall budgets include execution overhead. Timing is CPU and host/load dependent, with no deadline proof. Pairwise common-success costs and all category/template detail remain in the analysis CSV/JSON.

<!-- page 8 -->
## 8. Failure and mechanism analysis

The original 53 failures comprise 48 external-band and 5 self-band first failures. External negative clearance occurs 0 times; one last552901/0415 self gap is−0.010058 mm, still within the 0.1 mm band rather than penetration beyond−0.1 mm. Hard-position, actual-speed,1 cm tracking and nonfinite violations are 0. A displayed wider self-buffer flag alone is not failure. The 13 retained new failure episodes are all external-contact-band (tracking 7, fixed 1, tuned 0, PPO2/2/1), and are in Table 4 and supplementary_failures.csv, including times, links, categories and partial completion.

Direct observation: on the original 104 common successes,550/551 have higher command derivative RMS, raw clipping and mean auxiliary effort than 552. Commands change more on 48 Hz provider refreshes than held physics steps; logged JN_lambda u is nonzero. Mean leakage is0.08596/0.14499/0.15641/0.11153 mm/s for APF/550/551/552. No speed/soft gate saturation appears in that subset.

Clipping coexists with roughness across these models; it is not an identified cause. Within-seed episode clip-versus-smoothness Pearson values are−.432/−.518/−.288 (dependent, descriptive rows). The roughest 551 scene 0477 has only 14.29% component clipping, yet a 0.17805 rad/s auxiliary-vector jump and 0.15018 rad/s command jump at 1.0625 s. Claims that clipping alone caused oscillation are unsupported.

Plausible but untested mechanisms include the small bounded smoothness penalty relative to terminal success, deterministic mean magnitude, exploration history and absence of explicit self-clearance features/reward (q still indirectly encodes configuration). Best/last differences can reflect continued optimization and checkpoint-selection noise, but test differences cannot diagnose training causality. No reward/input/projection/self-repulsion ablation was performed, so performance is attributed to the whole controller.

<!-- page 9 -->
## 9. Limitations, development reflection and future work

Both experiments use one fixed q0, one sphere and short lines. Feasibility witnesses filter the population through a finite provider family; accepted scenarios do not represent arbitrary obstacles, initial states or unreachable tasks. Newfar share is 41%, versus36/131 originally, and tight share falls from33/131 to10/100. The new dataset is generated after parameter freeze but its family and screening were developed with prior knowledge. Training seeds share the same tasks. Template groups are few and may retain shared construction dependence; bootstrap intervals do not establish population-wide superiority.

Engineering reflection is grounded in this delivery, not invented personal experience. A fixed baseline left the proposal’s fair-tuning commitment incomplete; the finite validation-only search closes that gap while showing why a preferred outcome cannot drive more tuning. Earlier demos lacked complete learning entry points; explicit train/validate/load/evaluate/generate commands and a fresh installation make scope inspectable. Original logging labels and action-clipping counters required precise timing boundaries and raw-action recording.

Another lesson is that successful counts conceal trade-offs and late-model deterioration. Shared-complete metrics avoid rewarding early failures; per-ID losses reveal local regressions. Damped projection and censored distances demand faithful definitions. Frozen sources and historical hashes remain executable rather than being rewritten for new code. These are reproducibility practices, not a claim of independent student authorship.

Future studies should predeclare one causal ablation (dense self-clearance, action/command smoothing or projection choice), finite development budget and separate evaluation before retraining. Vary q0, obstacle geometry and path family with witnesses from more diverse providers; report selection bias and solver failures. Hardware validation needs new actuation, sensing and safety work. No additional training is justified merely by a peer’s lower PPO result.

<!-- page 10 -->
## 10. Reproducibility, disclosure and references

The full Stage 12 course package contains all runtime code, train/validation/load/evaluation/generation commands, locked dependencies, parameter datasets, six saved models (three primary), reports and result indices. A separate Release asset retains all new raw tuning, candidate/witness/replay and 600 comparison episodes. Historical 1048 raw evidence is separate. Stage 11 remains a lightweight demonstration. Hashes are checked before trusted model loading.

A new independent directory and new Python 3.11.16 venv installed PyBullet 3.2.7, Gymnasium 1.3.0, SB3 2.9.0, Torch 2.13.0+cpu, NumPy 2.4.6, Matplotlib 3.11.2 and TensorBoard 2.21.0. Six models load; five representatives (3 success,2 expected failure) match all compared states with maximum difference 0. Nine risk checks include action scale/hold, truncation, latched failure, collision/limit gates, data isolation and model loading. A 2048-step smoke updates parameters and reloads actions/optimizer with difference 0; it is an engineering check, excluded from scientific training totals. Exact commands, versions and hashes are in delivery/stage12/validation.

Codex assisted with implementation, execution, auditing, analysis, bilingual drafting and packaging. No member duties, personal mastery or originality percentage is inferred. No course template was supplied; both reports use≤10 pages including references. Course-specific AI disclosure/originality rules and final acceptance still require course confirmation. Cross-machine, full GUI, strict real-time and real-robot validation remain unverified.

## References (same in both versions; accessed 4 October 2026)

[1] O. Khatib. Real-Time Obstacle Avoidance for Manipulators and Mobile Robots. IJRR 5(1), 90-98, 1986. https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf

[2] J. Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[3] E. Coumans and Y. Bai. PyBullet Quickstart Guide; Bullet 3 Panda URDF. https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet

[4] A. Raffin et al. Stable-Baselines 3: Reliable Reinforcement Learning Implementations. JMLR 22(268), 1-8, 2021. https://jmlr.org/papers/v22/20-1364.html

[5] Farama Foundation. Gymnasium: Handling Time Limits (v 1.3.0). https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

[6] A. C. Cameron and D. L. Miller. A Practitioner's Guide to Cluster-Robust Inference. JHR 50(2), 317-372, 2015. https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf
