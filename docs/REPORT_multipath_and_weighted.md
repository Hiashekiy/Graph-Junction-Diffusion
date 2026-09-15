# 多分支解码 / 加权模型 评测报告

表格由脚本从原始评测 JSON 直接生成（不是手抄）；数据源见第 8 节。

## 1. 配置

| 项 | 取值 |
|---|---|
| 测试集（加权） | `data/weighted_controlled/weighted_controlled_test.pkl`，300 条，w ~ U(1,10) |
| 测试集（无权） | `data/controlled/controlled_test.pkl`，300 条，边没有 `weight` 属性 |
| single 解码 | **最终 candidate probability 的组内 argmax**（reverse chain 仍按 posterior 采样） |
| single_sampled | 直接解码采样出来的 z0 —— 旧默认行为，现只作诊断对照 |
| 多分支 | `top_k=2`、`beam_width=3`、`filter_dead_branches=True`；stop / skip 各一遍 |
| 设备 / seed | **CPU**（逐位可复现）/ seed 0 |

## 2. 指标定义

| 指标 | 定义 | 分母 | 备注 |
|---|---|---|---|
| `goal_hit_rate` | 解码出的路径最终到达 goal 的比例 | 全部 query | 主指标 |
| `optimal_path_rate` | 路径真实 cost（加权 = 边权和；无权 = 跳数）**恰好等于**最优 cost 的比例 | 全部 query | 没到 goal 记不最优，恒有 optimal ≤ goal_hit |
| `success_cost_ratio` | `pred_cost / optimal_cost` 均值 | **只有到达 goal 的 query** | 平均口径，会互相抵消 |
| `loop_rate` / `broken_rate` | 成环 / 中止（NULL、dead-end、超步数）的比例 | 全部 query | |
| `coverage_rate` | 存活表里**至少一条**到达 goal 的比例（集合语义） | 全部 query | 与挑了哪条无关 |
| `weighted_optimal_coverage_rate` | 存活表里**至少一条真实最优**路径的比例 | 全部 query | 加权数据集上才有这个名字 |
| `mean_goal_paths` / `mean_finished_paths` | 表里平均有多少条到终点 / 总共多少条走完的路径 | — | 表规模 |
| `mean_filtered_dead_branches` / `mean_pruned` | 平均被必死 branch 预筛选剔掉 / 被 beam 丢掉的数量 | — | 搜索统计 |

## 3. 解码器与 readout 定义

| 名字 | 用什么状态 | 每个路口 | 随机性 |
|---|---|---|---|
| `single`（默认，本次改动） | 最后一个 reverse step 的 candidate probability 的**组内 argmax** | 1 个候选 | 链仍采样，只有最终 readout 确定 |
| `single_sampled`（诊断） | reverse chain **采样**出来的 z0 | 1 个候选 | 有（抽 50 步） |
| deterministic rollout（`stochastic=False`） | **每一步** posterior 取 argmax 走到底 | 1 个候选 | 无 |
| `multi_best` | 存活表里累计 log 概率最高的一条（可以是 broken） | top-k 分支 | 概率来自同一条链 |
| `best_goal` | 到达 goal 的路径里概率最高 | top-k 分支 | 同上 |
| `best_goal_cost` | 到达 goal 的路径里真实 cost 最低 | top-k 分支 | 同上 |

## 4. 加权测试集结果（300 条）

| model | decoder | null | goal_hit | optimal | cost_ratio | loop | broken |
|---|---|---|---|---|---|---|---|
| Ours weighted（带 Edge Cost Encoder） | single（新默认：最终 argmax readout） | — | 0.8633 | 0.6900 | 1.0070 | 0.0000 | 0.1367 |
| Ours weighted（带 Edge Cost Encoder） | single_sampled（诊断口径，旧默认） | — | 0.8400 | 0.6800 | 1.0074 | 0.0067 | 0.1533 |
| Ours weighted（带 Edge Cost Encoder） | multi_best | stop | 0.8667 | 0.6933 | 1.0070 | 0.0000 | 0.1333 |
| Ours weighted（带 Edge Cost Encoder） | best_goal | stop | 1.0000 | 0.7833 | 1.0081 | 0.0000 | 0.0000 |
| Ours weighted（带 Edge Cost Encoder） | best_goal_cost | stop | 1.0000 | 0.9100 | 1.0028 | 0.0000 | 0.0000 |
| Ours weighted（带 Edge Cost Encoder） | multi_best | skip | 1.0000 | 0.7800 | 1.0081 | 0.0000 | 0.0000 |
| Ours weighted（带 Edge Cost Encoder） | best_goal | skip | 1.0000 | 0.7800 | 1.0081 | 0.0000 | 0.0000 |
| Ours weighted（带 Edge Cost Encoder） | best_goal_cost | skip | 1.0000 | 0.9800 | 1.0003 | 0.0000 | 0.0000 |
| Ours weighted + cost 消融（看不到边权） | single（新默认：最终 argmax readout） | — | 0.8767 | 0.3667 | 1.0557 | 0.0000 | 0.1233 |
| Ours weighted + cost 消融（看不到边权） | single_sampled（诊断口径，旧默认） | — | 0.7600 | 0.2867 | 1.0666 | 0.0033 | 0.2367 |
| Ours weighted + cost 消融（看不到边权） | multi_best | stop | 0.8400 | 0.3633 | 1.0556 | 0.0033 | 0.1567 |
| Ours weighted + cost 消融（看不到边权） | best_goal | stop | 1.0000 | 0.4333 | 1.0559 | 0.0000 | 0.0000 |
| Ours weighted + cost 消融（看不到边权） | best_goal_cost | stop | 1.0000 | 0.7533 | 1.0168 | 0.0000 | 0.0000 |
| Ours weighted + cost 消融（看不到边权） | multi_best | skip | 0.9967 | 0.4333 | 1.0560 | 0.0033 | 0.0000 |
| Ours weighted + cost 消融（看不到边权） | best_goal | skip | 1.0000 | 0.4333 | 1.0559 | 0.0000 | 0.0000 |
| Ours weighted + cost 消融（看不到边权） | best_goal_cost | skip | 1.0000 | 0.8767 | 1.0058 | 0.0000 | 0.0000 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | single（新默认：最终 argmax readout） | — | 0.9433 | 0.5067 | 1.0344 | 0.0000 | 0.0567 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | single_sampled（诊断口径，旧默认） | — | 0.9367 | 0.4867 | 1.0375 | 0.0000 | 0.0633 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | multi_best | stop | 0.9433 | 0.5067 | 1.0340 | 0.0000 | 0.0567 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | best_goal | stop | 1.0000 | 0.5267 | 1.0350 | 0.0000 | 0.0000 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | best_goal_cost | stop | 1.0000 | 0.7633 | 1.0129 | 0.0000 | 0.0000 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | multi_best | skip | 1.0000 | 0.5267 | 1.0350 | 0.0000 | 0.0000 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | best_goal | skip | 1.0000 | 0.5267 | 1.0350 | 0.0000 | 0.0000 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | best_goal_cost | skip | 1.0000 | 0.8733 | 1.0055 | 0.0000 | 0.0000 |

集合语义与表规模：

| model | null | coverage | w_opt_cov | mean_goal_paths | mean_finished | mean_filtered | mean_pruned |
|---|---|---|---|---|---|---|---|
| Ours weighted（带 Edge Cost Encoder） | stop | 1.0000 | 0.9100 | 2.86 | 8.78 | 11.16 | 0.33 |
| Ours weighted（带 Edge Cost Encoder） | skip | 1.0000 | 0.9800 | 4.48 | 9.60 | 13.24 | 0.91 |
| Ours weighted + cost 消融（看不到边权） | stop | 1.0000 | 0.7533 | 3.13 | 8.78 | 11.35 | 0.29 |
| Ours weighted + cost 消融（看不到边权） | skip | 1.0000 | 0.8767 | 5.24 | 9.48 | 13.35 | 1.07 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | stop | 1.0000 | 0.7633 | 2.26 | 9.16 | 11.67 | 0.29 |
| 无权模型 rev2_mixed（跨任务，看不到边权） | skip | 1.0000 | 0.8733 | 3.86 | 9.74 | 13.52 | 0.90 |

参考基线：Dijkstra oracle `goal_hit=1.0000 / cost_ratio=1.0000`；Greedy-BFS `goal_hit=1.0000 / cost_ratio=1.0350`。
数据集难度：`weighted_conflict_rate` 0.4597、`bfs_cost_ratio` 1.0344、`gt_path_is_weighted_optimal_fraction` 1.0。

## 5. 无权测试集结果（300 条）

| model | decoder | null | goal_hit | optimal | cost_ratio | loop | broken |
|---|---|---|---|---|---|---|---|
| rev2_mixed（最新无权模型） | single（新默认：最终 argmax readout） | — | 0.9233 | 0.9000 | 1.0014 | 0.0000 | 0.0767 |
| rev2_mixed（最新无权模型） | single_sampled（诊断口径，旧默认） | — | 0.9300 | 0.8833 | 1.0030 | 0.0000 | 0.0700 |
| rev2_mixed（最新无权模型） | multi_best | stop | 0.9233 | 0.9000 | 1.0014 | 0.0000 | 0.0767 |
| rev2_mixed（最新无权模型） | best_goal | stop | 1.0000 | 0.9700 | 1.0021 | 0.0000 | 0.0000 |
| rev2_mixed（最新无权模型） | best_goal_cost | stop | 1.0000 | 0.9967 | 1.0004 | 0.0000 | 0.0000 |
| rev2_mixed（最新无权模型） | multi_best | skip | 1.0000 | 0.9733 | 1.0015 | 0.0000 | 0.0000 |
| rev2_mixed（最新无权模型） | best_goal | skip | 1.0000 | 0.9733 | 1.0015 | 0.0000 | 0.0000 |
| rev2_mixed（最新无权模型） | best_goal_cost | skip | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| rev2_longmix（长链偏重无权模型） | single（新默认：最终 argmax readout） | — | 0.9567 | 0.9500 | 1.0004 | 0.0000 | 0.0433 |
| rev2_longmix（长链偏重无权模型） | single_sampled（诊断口径，旧默认） | — | 0.9300 | 0.9033 | 1.0022 | 0.0000 | 0.0700 |
| rev2_longmix（长链偏重无权模型） | multi_best | stop | 0.9533 | 0.9467 | 1.0004 | 0.0000 | 0.0467 |
| rev2_longmix（长链偏重无权模型） | best_goal | stop | 1.0000 | 0.9933 | 1.0004 | 0.0000 | 0.0000 |
| rev2_longmix（长链偏重无权模型） | best_goal_cost | stop | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| rev2_longmix（长链偏重无权模型） | multi_best | skip | 1.0000 | 0.9933 | 1.0004 | 0.0000 | 0.0000 |
| rev2_longmix（长链偏重无权模型） | best_goal | skip | 1.0000 | 0.9933 | 1.0004 | 0.0000 | 0.0000 |
| rev2_longmix（长链偏重无权模型） | best_goal_cost | skip | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| controlled_flow3（第一版无权，3 轮交流） | single（新默认：最终 argmax readout） | — | 0.9733 | 0.8567 | 1.0090 | 0.0000 | 0.0267 |
| controlled_flow3（第一版无权，3 轮交流） | single_sampled（诊断口径，旧默认） | — | 0.9167 | 0.7233 | 1.0183 | 0.0000 | 0.0833 |
| controlled_flow3（第一版无权，3 轮交流） | multi_best | stop | 0.9767 | 0.8667 | 1.0084 | 0.0000 | 0.0233 |
| controlled_flow3（第一版无权，3 轮交流） | best_goal | stop | 1.0000 | 0.8767 | 1.0096 | 0.0000 | 0.0000 |
| controlled_flow3（第一版无权，3 轮交流） | best_goal_cost | stop | 1.0000 | 0.9867 | 1.0008 | 0.0000 | 0.0000 |
| controlled_flow3（第一版无权，3 轮交流） | multi_best | skip | 1.0000 | 0.8767 | 1.0096 | 0.0000 | 0.0000 |
| controlled_flow3（第一版无权，3 轮交流） | best_goal | skip | 1.0000 | 0.8767 | 1.0096 | 0.0000 | 0.0000 |
| controlled_flow3（第一版无权，3 轮交流） | best_goal_cost | skip | 1.0000 | 0.9933 | 1.0005 | 0.0000 | 0.0000 |
| controlled_flow1（第一版无权，1 轮交流） | single（新默认：最终 argmax readout） | — | 0.9967 | 0.8767 | 1.0096 | 0.0000 | 0.0033 |
| controlled_flow1（第一版无权，1 轮交流） | single_sampled（诊断口径，旧默认） | — | 0.8933 | 0.7333 | 1.0158 | 0.0033 | 0.1033 |
| controlled_flow1（第一版无权，1 轮交流） | multi_best | stop | 0.9967 | 0.8867 | 1.0084 | 0.0000 | 0.0033 |
| controlled_flow1（第一版无权，1 轮交流） | best_goal | stop | 1.0000 | 0.8867 | 1.0085 | 0.0000 | 0.0000 |
| controlled_flow1（第一版无权，1 轮交流） | best_goal_cost | stop | 1.0000 | 0.9933 | 1.0007 | 0.0000 | 0.0000 |
| controlled_flow1（第一版无权，1 轮交流） | multi_best | skip | 1.0000 | 0.8867 | 1.0085 | 0.0000 | 0.0000 |
| controlled_flow1（第一版无权，1 轮交流） | best_goal | skip | 1.0000 | 0.8867 | 1.0085 | 0.0000 | 0.0000 |
| controlled_flow1（第一版无权，1 轮交流） | best_goal_cost | skip | 1.0000 | 0.9967 | 1.0005 | 0.0000 | 0.0000 |
| Ours weighted（跨到无权数据，边权全为 1） | single（新默认：最终 argmax readout） | — | 0.8100 | 0.7833 | 1.0024 | 0.0000 | 0.1900 |
| Ours weighted（跨到无权数据，边权全为 1） | single_sampled（诊断口径，旧默认） | — | 0.8233 | 0.7667 | 1.0061 | 0.0000 | 0.1767 |
| Ours weighted（跨到无权数据，边权全为 1） | multi_best | stop | 0.8100 | 0.7833 | 1.0024 | 0.0000 | 0.1900 |
| Ours weighted（跨到无权数据，边权全为 1） | best_goal | stop | 1.0000 | 0.9500 | 1.0039 | 0.0000 | 0.0000 |
| Ours weighted（跨到无权数据，边权全为 1） | best_goal_cost | stop | 1.0000 | 0.9800 | 1.0013 | 0.0000 | 0.0000 |
| Ours weighted（跨到无权数据，边权全为 1） | multi_best | skip | 1.0000 | 0.9500 | 1.0039 | 0.0000 | 0.0000 |
| Ours weighted（跨到无权数据，边权全为 1） | best_goal | skip | 1.0000 | 0.9500 | 1.0039 | 0.0000 | 0.0000 |
| Ours weighted（跨到无权数据，边权全为 1） | best_goal_cost | skip | 1.0000 | 0.9967 | 1.0002 | 0.0000 | 0.0000 |

| model | null | coverage | w_opt_cov | mean_goal_paths | mean_finished | mean_filtered | mean_pruned |
|---|---|---|---|---|---|---|---|
| rev2_mixed（最新无权模型） | stop | 1.0000 | 0.9967 | 2.18 | 9.00 | 12.36 | 0.34 |
| rev2_mixed（最新无权模型） | skip | 1.0000 | 1.0000 | 3.68 | 9.63 | 14.38 | 0.99 |
| rev2_longmix（长链偏重无权模型） | stop | 1.0000 | 1.0000 | 2.13 | 8.46 | 11.44 | 0.17 |
| rev2_longmix（长链偏重无权模型） | skip | 1.0000 | 1.0000 | 3.16 | 9.45 | 13.70 | 0.63 |
| controlled_flow3（第一版无权，3 轮交流） | stop | 1.0000 | 0.9867 | 2.66 | 9.39 | 12.50 | 0.21 |
| controlled_flow3（第一版无权，3 轮交流） | skip | 1.0000 | 0.9933 | 4.91 | 10.05 | 15.40 | 1.43 |
| controlled_flow1（第一版无权，1 轮交流） | stop | 1.0000 | 0.9933 | 2.56 | 9.37 | 12.65 | 0.33 |
| controlled_flow1（第一版无权，1 轮交流） | skip | 1.0000 | 0.9967 | 3.81 | 9.99 | 15.12 | 1.25 |
| Ours weighted（跨到无权数据，边权全为 1） | stop | 1.0000 | 0.9800 | 2.77 | 9.10 | 12.66 | 0.45 |
| Ours weighted（跨到无权数据，边权全为 1） | skip | 1.0000 | 0.9967 | 4.34 | 9.61 | 14.59 | 1.14 |

无权图上所有边权为 1，`best_goal_cost` 等价于「到达 goal 的路径里跳数最少的那条」。

