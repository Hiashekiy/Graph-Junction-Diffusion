# 新模型评测指南（Report Guide）

一个模型训完之后，**哪些结论必须回答、按什么顺序、用什么命令、看哪几个数、哪些数不能跨配置比**。

每节格式统一：**要回答什么 → 为什么 → 命令 → 看哪个数 → 健康标准 → 陷阱**。
命令一律写成**单行**（cmd / PowerShell 都能直接粘），路径按仓库根。

> **先说一条最重要的**：任何数字都必须**连着口径一起写**。
> 同一个 checkpoint、同一个样本，strict 2/3 和 历史 2/64 可以给出**完全相反**的结论
> （实测 `test_1000` 第 0 条：标尺下 6 条路线全部到达，历史口径下 6 条全部 broken）。
> 脱离口径的"GoalHit = 0.8"没有意义。见 §1。

---

## 0. 评测顺序总览

| 层 | 要回答的问题 | 成本 | 命令入口 |
|---|---|---|---|
| **L0 能不能用** | 代码/数据/权重自洽吗 | 1 分钟 | `pytest`、`verify_didi_pipeline.py` |
| **L1 学到了没有** | 训练曲线是否在改善、有没有塌缩 | 1 分钟 | `tools/collect_run.py`、`tools/compare_curves.py` |
| **L2 主实验** | 在固定测试集上的主指标是多少 | 分钟级 | `scripts/evaluate.py --data .../test_1000.pkl` |
| **L3 解码口径** | strict 与历史、top_k/beam 的影响 | 分钟级 | `tools/ablate_didi_multi_decode.py` |
| **L4 鲁棒性 / 成本** | 换 OD 还行吗、推理多贵 | 分钟级 | `shuffled_od_1000.pkl`、`benchmark_didi_inference.py` |
| **L5 显著性** | 差异是真的还是噪声 | 分钟级 | `tools/paired_compare.py`、`tools/multiseed_eval.py` |
| **L6 消融** | 每个组件各贡献多少 | 需重训 | 见 §8 的表 |
| **L7 复现性 / 泄漏** | 同种子逐位一致吗、有泄漏吗 | 分钟级 | `--deterministic`、`tools/check_leakage.py` |
| **L8 人工检查** | 路径形状对不对 | 分钟级 | 面板 / `visualize_didi_paths.py` |

**只做 L0+L1+L2 就能出一份能交的报告**；L3–L5 决定这份报告有没有说服力；L6 决定能不能写"因为我们加了 X"。

---

## 1. 三把尺子（读数字之前先确认用的是哪把）

| 用在哪 | 解码器 | top_k / beam | 谁在用 |
|---|---|---|---|
| **训练 miner** | 历史（`strict=false`） | 2 / 8 | `mine_success_and_failure`，**故意放宽**以便看得见失败轨迹 |
| **验证 / 选 best.pt / 最终评测** | strict 三池 | 2 / 3 | `Trainer.validate()` 与 `scripts/evaluate.py` 读同一组 `evaluation.*` |
| **可视化面板** | 两者可选，**默认标尺** | 跟 config | 面板「解码口径」选择器（README §22.6） |

- 验证与最终评测**已经是同一把尺子**（`evaluation.decode / strict_decode / top_k / beam_width /
  stochastic_sampling` 由 config 驱动，`Trainer.validate()` 和 `evaluate.py` 都读它）。
  这一条是踩过坑之后才统一的：曾经验证走 `single`、报表走 strict，于是 `best.pt` 是照一把尺子挑的、
  报表却拿另一把尺子的数字。
- 面板默认也已切到标尺；面板上解出来的路径**应该和评测表对得上**，对不上先查口径。

**写报告时每条数字后面都要标口径**，例如：`GoalHit 0.87（strict 2/3, beam=3, deterministic, test_1000）`。

---

## 2. 指标可信度分级

| 指标 | 含义 | 能跨配置比吗 | 备注 |
|---|---|---|---|
| `goal_hit_rate` | 解出的路径到达 goal 的比例 | ✅ | 主指标 |
| `optimal_path_rate` | 到达**且**等于最短路（跳数口径） | ✅ | |
| `weighted_optimal_coverage_rate` | 加权图上按真实 cost 判定的最优覆盖率 | ⚠️ | 只在 multi 口径下，且受 beam 影响 |
| `success_cost_ratio` | 只在成功样本上 `L_pred / L_optimal` | ✅ | 成功率为 0 时**无定义** |
| `loop_rate` / `broken_rate` | 成环 / 中止（NULL、dead-end、超步数） | ✅ | 必须标 strict 还是历史 |
| `path_similarity_score` | nLCS 选择分（未到达记 0） | ✅ | DiDi 的 `selection_metric` |
| `nLCS(success)` / `edge_f1` | 成功样本的路径相似度 / 边 F1 | ✅ | 只统计成功样本，与成功率一起看 |
| `pred_cost_ratio` / `pred_over_gt_cost_ratio` | 预测 cost / GT cost | ✅ | GT 不是最短路（实测中位 1.12） |
| `dtw_km` | km-based DTW（真实距离） | ✅ | 需要 `data.coords_file`；缺坐标 → NaN |
| `coverage_rate` | 路径表里至少一条到终点 | ❌ | **top-k 逃逸口**：`top_k=1` 864/1000 → `top_k=2` 963/1000 |
| `optimal_coverage_rate` | 表里至少一条最短路 | ❌ | **beam 预算 best-of-N**：beam=1 95/1000 → beam=64 447/1000 |
| `multi_best_goal_cost` | 表里 cost 最小的到达路径 | ❌ | 同上的 beam 预算效应 |
| `KLEV` | 分布散度 | ❌ | 基本等于 `log(1/eps)`（eps 1e-12→0.1 时 23.25→0.29） |
| `JSEV` | 有界分布散度 | ⚠️ | 比 KLEV 稳定，作附录 |
| `soft_goal_reachability` | 可微软可达性 | ❌ | 训练用的代理指标，**不能**替代硬指标 |
| `decision accuracy`（`x0_acc`） | 逐 decision 准确率 | ❌ | NULL 占 93.6%，"全押 NULL"就能很漂亮 |

**一句话**：带 ❌ 的只能在同一份 config 下做前后对比，**不要**放进跨模型的主表。

---

## 3. L0 · 能不能用（1 分钟）

**要回答**：代码、数据、权重三者自洽吗？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
E:/CondaEnvData/envs/GGMPC/python.exe tools/verify_didi_pipeline.py --data data/didi/graph/chengdu
```

期望：`538 passed` + `all 27 unit checks passed`。

**再看三样东西**：

1. **run 目录自证**：`outputs/runs/<run>/run_config.json` 里的 `paths.run_name`、
   `loss.*`、`evaluation.*` 是否就是你这次跑的。
   ⚠️ `scripts/train.py` **不记录** `--data` / `--val-data`（已知问题），所以"这个 run 用的哪份数据"
   只能靠 `paths.data_dir` + 命名区分。
2. **checkpoint ↔ config 一致**：`tests/test_checkpoint_mismatch.py` 覆盖了这条；
   手动确认 `run_config.json` 里的 `model.flow_steps` / `d_model` / `use_edge_cost` 和权重匹配。
3. **目标版本**：`loss.type` + `loss.null_loss_type` + `loss.trajectory.enabled`。
   不同目标版本的 checkpoint **不可比、也不能 `--resume` 混训**
   （`run_config.json` 每次启动都被无条件重写，事后看不出来）。

---

## 4. L1 · 学到了没有（训练曲线，1 分钟）

**要回答**：曲线是在改善，还是塌到了平凡解？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe tools/summarize_run.py outputs/runs/didi_chengdu_loss_improved --skip-eval
E:/CondaEnvData/envs/GGMPC/python.exe tools/collect_run.py outputs/runs/didi_chengdu
```

**看这几个数（第四版目标）**：

| 指标 | 健康方向 | 说明 |
|---|---|---|
| `train_active_branch_acc` | ↑ | **最重要**：真实 branch 选对的比例 |
| `train_mean_gt_branch_prob` | ↑ | 给 GT branch 的平均概率 |
| `train_path_nll` | ↓ | 主监督项 |
| `train_pred_active_rate` | 靠近监督比例（≈0.34） | 远高于它 = 过度激活 → loop 高 |
| `train_trajectory_loss` | ↓ | 多轨迹集合损失（已含 λ_T） |
| `train_traj_success_mass` | ↑ | 落在"到达 goal"上的集合质量 |
| `train_traj_failure_mass` | ↓ | 与上一条互补，和为 1 |
| `train_traj_num_candidates` | > 1 | 长期贴 1 = miner 什么都挖不到，这一项等于没开 |
| `train_null_saturation_rate` | 到 1.0 就停 | 到 1.0 说明该调 `rho_null`，再加 λ 无效 |
| `val_x0_acc` / `train_x0_acc` | **别看** | 已改名 `*_all_decision_acc` 并移出主日志 |

**红旗**：

| 现象 | 最可能的原因 |
|---|---|
| `train_x0_acc ≈ NULL 占比`、`goal_hit` 恒 0 | NULL 塌缩（旧 CE 的平凡最优解） |
| `traj_success_mass` 不动、`traj_num_success = 1` | miner 只挖到 GT，beam 预算不够 |
| `traj_raw_success` / `raw_*` 一直贴着 `max_success`/`max_failure` | 同上，先加 `beam_width` |
| `traj_failure_mass` ↑ 而 `trajectory_loss` ↓ | 只是失败代价被压掉了，模型未必变好 |
| `null_saturation_rate = 0` 且 `pred_active_rate` 低 | NULL 侧还没压住 |
| `val` 曲线抖动、`best.pt` 早停 | 验证集太小（DiDi val 512 条），配合 L5 看显著性 |

**实例**（2026-09-16，`didi_chengdu_loss_improved` 第 39 轮，第四版目标）—— 这组读数是"健康但预算吃紧"的样子：

```text
train_active_branch_acc   0.786      <- 第三版同期只有 0.536，说明 L_traj 起作用了
train_traj_success_mass   0.871      <- 集合质量集中在"到达 goal"上
train_traj_failure_mass   0.129
train_null_saturation_rate 0.166     <- 远没到 1，饱和式 NULL 还有梯度（正常，还早）
train_traj_num_candidates 8.856      <- 池子上限 = 1(GT)+4+4 = 9，已经顶格
train_traj_raw_success   13.300  >  max_success = 4     <- miner 挖到 13 条只留下 4 条
train_traj_raw_null      45.659 / raw_loop 32.641       <- 失败轨迹也远多于 max_failure = 4
train_traj_raw_no_decision 0.000                        <- 没有"零 decision"的空 trace 被剔除
```

`raw_*` 全部远超 `max_success`/`max_failure` 且 `num_candidates` 顶格 —— 这说明
**进入 loss 的候选是被 cap 砍掉的，不是 miner 挖不到**。想继续压 `trajectory_loss`，
下一步应该调 `max_success`/`max_failure` 或 `beam_width`，而不是加 `weight`。

---

## 5. L2 · 主实验：固定 test_1000（分钟级）

**要回答**：主指标是多少，和基线比怎么样。

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/test_1000.pkl --deterministic --baselines --out outputs/runs/didi_chengdu_loss_improved/eval_test_1000.json
```

- **口径不用手写**：`--decode / --top-k / --beam-width / --strict-decode` 的 CLI 默认是 `None`，
  会从 `evaluation.*` 读（当前 = **strict multi 2/3**）。
- **必须带 `--deterministic`**：config 里 `evaluation.stochastic_sampling: false`，
  与验证期保持一致，否则同一 checkpoint 会出两个数。
- **必须用 `test_1000.pkl`**（GDP 风格固定子集）：全量 `test.pkl` 是 1307 条，
  和论文/README §21 的表不是一个口径，两个都报时要写清楚。

**先确认 `best.pt` 是哪一轮**。`selection_metric` 是 `path_similarity_score`（**strict 口径下**算的），
`best.pt` 可能明显早于 `last.pt`：

```text
实例（didi_chengdu_loss_improved）       checkpoint 里存着 epoch 号，直接读得到
  best.pt -> epoch 15        last.pt -> epoch 39        ← 差了 24 轮

验证曲线（epoch, PathSim, GoalHit）
    5  0.3971  0.992        20  0.3878  0.973        30  0.3961  1.000
   10  0.4024  0.986        25  0.3943  0.996        35  0.3816  0.996
   15  0.4145  0.998   <- best     最高与最低只差 0.033
```

两个后果，报告里必须处理：

1. **别只报 `best.pt`**：它比 `last.pt` 早 24 轮，趋势完全不同。两个都测，写清用的是哪个。
2. **验证噪声和模型差异同量级**（PathSim 在 0.3816～0.4145 之间抖，跨度 0.033）——
   任何小于这个幅度的"提升"都**必须**走 §8 的配对检验才能写进结论。

**报告里至少放这 6 个数**（连同口径）：

```
GoalHit / OptimalPath / Broken / Loop        ← metrics
PathSim / EdgeF1 / DTW(km)                    ← real_metrics
pred_over_gt_cost_ratio                       ← 成本
mean_elapsed                                   ← 成本（推理时间）
```

**基线**：`--baselines` 会给 Dijkstra 参照；真实数据上还要记住
**GT 本身不是最短路**（实测 non-optimal：train 93.4% / val 92.4% / test 94.4%；
`gt_cost_ratio` 中位数 1.12、test p95 1.53），
所以"OptimalPathRate 低"不一定说明模型差 —— 要同时看 `PathSim` 和 `pred_over_gt_cost_ratio`。

**已知的 strict 口径基线读数**（第三版目标 checkpoint，`test_1000`）：
PathSim 0.4438 / EdgeF1 0.3828 / DTW 0.2276 km / broken 0。
第四版换了目标函数，**这组数要重测**，不能直接沿用。

---

## 6. L3 · 解码口径：strict 与历史到底差多少

**要回答**：结论对解码口径有多敏感？换个 top_k/beam 会不会翻盘？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe tools/ablate_didi_multi_decode.py --run outputs/runs/didi_chengdu_loss_improved --data data/didi/graph/chengdu/test_1000.pkl
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_multipath.py --run outputs/runs/didi_chengdu_loss_improved --data data/didi/graph/chengdu/test_1000.pkl --top-k 2 --beam-width 3
```

**必看**：同一份 checkpoint 在 `strict=True` 与 `strict=False` 下的
`broken_rate` / `PathSim` / `DTW` 三列并排 —— 这组的差异通常**远大于**模型之间的差异。

**陷阱**：

- strict 会在 top-k **之前**把 NULL / loop / dead-end 全部 mask 掉，所以 `broken_rate` 会掉到 0，
  这是**定义使然**，不是模型变好了；
- strict 下 `null_policy` 失效、`filter_dead_branches` 恒为真；
- 报告里凡是出现 `coverage_rate` / `optimal_coverage_rate`，必须一起写 `top_k` 和 `beam_width`
  （见 §2 的 ❌）。

---

## 7. L4 · 鲁棒性与成本（分钟级）

**要回答**：换一批 OD 还行吗？"走廊"这个设计本身成立吗？推理多贵？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/shuffled_od_1000.pkl --deterministic --out outputs/runs/didi_chengdu_loss_improved/eval_shuffled.json
E:/CondaEnvData/envs/GGMPC/python.exe tools/breakdown_eval.py --data data/didi/graph/chengdu/test_1000.pkl --out outputs/runs/didi_chengdu_loss_improved/breakdown.json outputs/runs/didi_chengdu_loss_improved/eval_test_1000.json
E:/CondaEnvData/envs/GGMPC/python.exe tools/benchmark_didi_inference.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/test_1000.pkl --count 200
```

- **shuffled OD 没有真实 GT**：相似度指标会跳过，只能看 `goal_hit` / `broken` / `coverage` 这类集合语义。
  它回答的是"模型是不是只会背训练过的 OD"。
- **corridor retention 不是模型指标**：`rho=1.5` 时 train GT containment 0.91，
  剩下 9% 记 `corridor_miss` 并**单独报告**，不要混进模型指标里。
- **分桶**：`breakdown_eval` 按难度 / 结构模式拆；DiDi 上按 GT 长度分桶更有意义
  （长 OD 是难点，均值会把它抹平）。
- **成本**：`mean_elapsed` 与 `benchmark_didi_inference.py` 的 p50/p95 一起报；
  注意 `flow_steps` 直接乘推理时间，**训练与推理必须同轮数**。

---

## 8. L5 · 统计显著性：差异是真的吗

**要回答**：A 比 B 高 2 个点，是真提升还是噪声？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe tools/paired_compare.py --a outputs/runs/didi_chengdu_loss_improved/eval_test_1000.json --b outputs/runs/didi_chengdu/eval_test_1000.json --metric goal_hit --out outputs/reports/paired_goal_hit.json
E:/CondaEnvData/envs/GGMPC/python.exe tools/multiseed_eval.py --a outputs/runs/didi_chengdu_loss_improved --b outputs/runs/didi_chengdu --data data/didi/graph/chengdu/test_1000.pkl --seeds 0 1 2 --metric goal_hit
```

- `paired_compare` 做**同一批 query 的配对 + McNemar 精确检验**（`src/evaluation/paired.py`），
  这是唯一能说"差异显著"的方式；直接比两个均值不行。
- **什么时候必须做**：差异 < 3 个百分点时；或者要写"我们的方法更好"时。
- 多种子：报告里给 `mean ± std`，单种子结果只能当趋势。

---

## 9. L6 · 消融

**要回答**：每个组件各贡献多少？

| 变量 | 怎么关 | 备注 |
|---|---|---|
| Edge cost 进不进网络 | `--set model.use_edge_cost=false` | 权重不匹配，**必须重训**；对照 run 在 2026-09-16 的清理里已删除，要用就重训一份 |
| 图信息交流轮数 | `--set model.flow_steps=1/2/3` | 训练与推理轮数必须一致 |
| 多轨迹集合损失 | `--set loss.trajectory.enabled=false` | 退回第三版目标 |
| 饱和式 NULL | `--set loss.null_loss_type=nll` | 退回无界 NLL（λ 要同时调回 0.3） |
| 三个子项 | `--set loss.trajectory.success_weight=0` 等 | 一次只关一个 |

计划中的 loss 消融表（README §24.14，**尚未执行**）：

| 组 | PathNLL | 局部 NULL | L_succ | L_sim | L_fail |
|---|---|---|---|---|---|
| A Baseline | ✓ | × | × | × | × |
| B Low-NULL | ✓ | 0.10 普通 NLL | × | × | × |
| C Saturating-NULL | ✓ | 0.10 饱和式 | × | × | × |
| D + Success | ✓ | 0.10 饱和式 | ✓ | × | × |
| E + Success+Sim | ✓ | 0.10 饱和式 | ✓ | ✓ | × |
| F Full | ✓ | 0.10 饱和式 | ✓ | ✓ | ✓ |

A→C 回答"饱和式本身有没有用"，C→F 回答"多轨迹项逐块加有没有用"。
**一次只动一个变量**，且每组都要走完 L2+L5。

---

## 10. L7 · 复现性与泄漏

**要回答**：同样的命令能跑出同样的数吗？测试集干净吗？

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --all data/unweighted/unweighted_train.pkl data/unweighted/unweighted_val.pkl data/unweighted/unweighted_test.pkl
```

- **逐位复现**：同一 checkpoint + `--deterministic` + 同一份数据 → 两次评测的
  `metrics` 必须完全一致。不一致先查 `evaluation.stochastic_sampling` 与 `seed`。
- **零破坏回归**：改公共代码后，旧 checkpoint 的评测必须逐位不变
  （`outputs/runs/controlled_unweighted/regression_after_weighted_extension.json` 是这个套路）。
- **泄漏**：`verify_didi_pipeline.py` 已断言 shuffled OD 与 test OD 不相交（991 对）；
  `check_leakage.py` 查数据集之间的图/边重叠。
  **corridor 由 train split 选定后冻结**，val/test 不得再按 GT 调 `rho`。

---

## 11. L8 · 人工检查（别只信数字）

```cmd
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py --device cpu
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_didi_paths.py --run outputs/runs/didi_chengdu_loss_improved --num 6 --cols 2 --select mixed --goal-fraction 0.6 --mode both --multi-k 2 --deterministic
```

看四件事：

1. **GT（蓝虚线）和预测（橙线）的形状**：分歧出现在第几个路口？那里有没有明显更好的选择？
2. **`--select mixed`**（默认 6:4 混成功/失败）：只看成功会高估模型，只看失败看不出它对在哪。
3. **面板的「扩散过程」**：模型是在哪一步"想歪"的？NULL 是不是集中出现在某类路口。
4. **面板的口径**：状态行会印 `口径 ...`，确认你看的和报表是同一把尺子。

---

## 12. 报告模板（直接抄）

```markdown
# <run 名> 评测报告

## 0. 一句话结论
<用主指标说清楚：比什么、好多少、显著吗>

## 1. 训练配置（自证）
| 项 | 值 |
|---|---|
| config | configs/didi_chengdu.yaml |
| 训练数据 | data/didi/graph/chengdu/train.pkl（4678 条） |
| 验证数据 | data/didi/graph/chengdu/val.pkl（512 条） |
| 目标函数 | path_nll + 0.10·NULL-sat(ρ=0.6) + 0.50·L_traj |
| T / flow_steps / batch | 50 / 1 / 8 |
| epochs / 用时 | 100 / <挂钟时间> |
| best.pt | epoch <N>（按 path_similarity_score 选） |

## 2. 训练曲线
<active_branch_acc ↑ / trajectory_loss ↓ / success_mass ↑ / null_saturation_rate>
<红旗检查：有没有塌缩、miner 有没有挖到东西>

## 3. 主实验（口径：strict 2/3, beam=3, deterministic, test_1000, n=1000）
| 指标 | 本模型 | 基线/Dijkstra | 上一版 |
|---|---|---|---|
| GoalHit | | | |
| OptimalPath | | | |
| Broken / Loop | | | |
| PathSim | | | |
| EdgeF1 | | | |
| DTW(km) | | | |
| pred/GT cost | | | |
| mean_elapsed | | | |

## 4. 解码口径敏感性（strict vs 历史，top_k/beam 扫描）
<表：口径 × (broken, PathSim, DTW)>

## 5. 鲁棒性
<shuffled OD；GT 长度分桶；corridor retention>

## 6. 显著性
<配对 McNemar p 值；多种子 mean±std>

## 7. 消融（若做了）
<一次一个变量>

## 8. 复现性
<同种子逐位一致；零破坏回归>

## 9. 失败案例
<可视化挑 3 条，说清错在哪一步>

## 10. 已知限制
<KLEV 不能比 / coverage 是 top-k 逃逸口 / GT 不是最短路 / 样本量 …>
```

---

## 13. 红旗清单（出现就停下来查）

| 现象 | 先查什么 |
|---|---|
| 面板/报告/README 的数字对不上 | **口径**是否同一把（§1） |
| `goal_hit` 高但 `PathSim` 低 | GT 本身不是最短路（DiDi），或路径绕远 |
| `broken_rate = 0` | 是不是 strict 口径（定义使然） |
| `coverage_rate` 接近 1 | top_k/beam 是不是被放大了（§2） |
| `dtw_km` 是 NaN | `data.coords_file` 缺失，或样本有节点没坐标 |
| 两次评测数字不同 | `stochastic_sampling` / `--deterministic` / seed |
| 换了目标版本但指标没变 | `run_config.json` 被重写了，实际加载的是旧权重 |
| `train_traj_num_candidates` 长期 = 1 | miner 没挖到候选，`L_traj` 恒为 0 |
| val 曲线抖得厉害 | 验证集太小，看 L5 显著性 |

---

## 14. 一页速查

```cmd
REM L0
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
E:/CondaEnvData/envs/GGMPC/python.exe tools/verify_didi_pipeline.py --data data/didi/graph/chengdu
REM L1
E:/CondaEnvData/envs/GGMPC/python.exe tools/summarize_run.py outputs/runs/didi_chengdu_loss_improved --skip-eval
REM L2
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/test_1000.pkl --deterministic --baselines --out outputs/runs/didi_chengdu_loss_improved/eval_test_1000.json
REM L3
E:/CondaEnvData/envs/GGMPC/python.exe tools/ablate_didi_multi_decode.py --run outputs/runs/didi_chengdu_loss_improved --data data/didi/graph/chengdu/test_1000.pkl
REM L4
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/shuffled_od_1000.pkl --deterministic --out outputs/runs/didi_chengdu_loss_improved/eval_shuffled.json
E:/CondaEnvData/envs/GGMPC/python.exe tools/benchmark_didi_inference.py --config configs/didi_chengdu.yaml --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt --data data/didi/graph/chengdu/test_1000.pkl --count 200
REM L5
E:/CondaEnvData/envs/GGMPC/python.exe tools/paired_compare.py --a outputs/runs/didi_chengdu_loss_improved/eval_test_1000.json --b outputs/runs/didi_chengdu/eval_test_1000.json --metric goal_hit --out outputs/reports/paired_goal_hit.json
REM L7
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --all data/unweighted/unweighted_train.pkl data/unweighted/unweighted_val.pkl data/unweighted/unweighted_test.pkl
REM L8
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_didi_paths.py --run outputs/runs/didi_chengdu_loss_improved --num 6 --cols 2 --select mixed --mode both --multi-k 2 --deterministic
```

相关文档：主 README §7（指标定义）、§21（多分支口径）、§22（面板与解码口径）、
§23（single readout）、§24（DiDi 真实数据与当前训练目标）。
