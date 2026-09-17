# 论文评测协议（Evaluation Protocol）

> 本文是**论文定稿用**的评测协议：规定评什么、用哪个模型、哪个数据集、哪把尺子、指标怎么算。
> 所有结论数字都必须能追溯到 `outputs/eval/` 下的一个 JSON，命名可反查（见 §7）。
>
> 通用评测方法论（L0–L8 阶梯、红旗清单、报告模板）另见 `docs/EVAL_GUIDE.md`；
> 本文只规定**最终这一轮**要跑的东西。
>
> 执行入口：`python scripts/run_eval_protocol.py`（见 §7）。

---

## 1. 目的

回答三个问题，对应论文的三张表：

| 论文结论 | 靠哪几组实验 |
|---|---|
| **C1** 大图微调是否真的提升了规模泛化能力 | 主表 D2 行） + 搜索预算曲线（§5 第 3、4 组） |
| **C2** 微调有没有造成灾难性遗忘 | 主表 D1、D3 行 |
| **C3** 提升来自模型本身而不是解码器搜索更暴力 | beam 3 / 8 / 16 的差距收敛（§5 第 2、3、4 组） |

补充：D4–D6 的 shuffled-OD 用来排除"模型只是记住了真实 OD 对"这一质疑。

---

## 2. 被评模型

两个模型**必须成对评测**，任何一张表都不允许只出现其中一个。

| 代号 | checkpoint | 来源 | 选择依据 |
|---|---|---|---|
| **M0** base | `outputs/runs/didi_chengdu_loss_improved/best.pt` | 成都普通图训练，lr 1e-4 | val PathSim 最优 = epoch 15 |
| **M1** finetuned | `outputs/runs/didi_chengdu_large_finetune/best.pt` | 由 M0 **warm start**，lr 2e-5、batch 1、20 epoch | 混合 val PathSim 最优 = epoch 13 |

两者**结构完全相同**（`d_model=128 / ffn_hidden=256 / flow_steps=1 / use_edge_cost=true`），
所以可以直接互相对照，差异只来自训练方式与数据分布。

训练自证（可复核）：

```
outputs/runs/<run>/run_config.json    解析后的完整配置
outputs/runs/<run>/history.json       逐 epoch 曲线
```

⚠️ **不要**拿 `history.json` 里的 val PathSim 去横向比两个模型：
M0 的 val 是普通成都 val，M1 的 val 是 `1/3 normal + 2/3 long` 的混合分布，
绝对值不可比。可比的是本文 §5 的 test 数字。

---

## 3. 数据集

全部是**固定 1000 条**的子集（GDP 风格固定子集，保证跨模型、跨口径逐条对齐）。

| 代号 | 文件 | n | GT 来源 | 图规模（median） | 用途 |
|---|---|---|---|---|---|
| **D1** | `data/didi/graph/chengdu/test_1000.pkl` | 1000 | `observed`（真实司机轨迹） | ~353 节点 / 22 跳 | 普通图能力（遗忘检查） |
| **D2** | `data/didi/graph/chengdu_long/test_1000.pkl` | 1000 | `observed` | ~1800 节点 / 76 跳 | **大图能力（核心）** |
| **D3** | `data/didi/graph/xian/test_1000.pkl` | 1000 | `observed` | 西安路网 | 跨城市泛化（遗忘检查） |
| **D4** | `data/didi/graph/chengdu/shuffled_od_1000.pkl` | 1000 | `dijkstra_placeholder` | 同 D1 | 随机 OD 泛化 |
| **D5** | `data/didi/graph/chengdu_long/shuffled_od_1000.pkl` | 1000 | `dijkstra_placeholder` | 同 D2 | 大图随机 OD |
| **D6** | `data/didi/graph/xian/shuffled_od_1000.pkl` | 1000 | `dijkstra_placeholder` | 西安 | 西安随机 OD |

**关于 `dijkstra_placeholder`**：D4–D6 的"GT"是**合成的最短路占位符**，不是真实轨迹。
因此这三组上 **`path_similarity_score` / `normalized_lcs` 无意义**（评测器会跳过，
见 `real_path_metrics.num_skipped_placeholder_gt`），只能看 **GoalHit / Broken / cost 类**指标。
表格里这三组不要填 PathSim。

**关于 D2**：`gt_cost_ratio ≈ 1.38`，即真实司机路径平均比最短路长 38%。
这不是数据问题，是任务定义 —— GT 是**观测行为**，不是最优解。所以
`optimal_path_rate` 在真实数据上只作 secondary 指标（见 §6）。

---

## 4. 解码口径

三把尺子，**只有 beam 宽度不同**，其余全同：

| 代号 | 参数 |
|---|---|
| **R3** | `--decode multi --strict-decode --top-k 2 --beam-width 3 --deterministic` |
| **R8** | 同上，`--beam-width 8` |
| **R16** | 同上，`--beam-width 16` |

共同点（由 config 的 `evaluation.*` 段保证，勿在命令行覆盖）：

```
decode                = multi     存活路径表解码
strict_decode         = true      三池语义：NULL / loop / dead-end 在 top-k 之前 mask，
                                  失败路径直接淘汰，最终候选池只有走到 Goal 的完整路径
top_k                 = 2         每个 decision 取概率最高的 2 条合法 branch 分叉
null_policy           = stop
filter_dead_branches  = false
stochastic_sampling   = false     = --deterministic，posterior argmax，无随机性
```

**R3 是主口径**，论文所有主表数字用 R3。R8 / R16 只用于搜索预算曲线（C3）。

⚠️ 不要用 R16 选 best.pt 或报主表 —— R16 能把旧模型的大图 GoalHit 从 0.781 抬到 0.977，
那是 decoder 搜得更暴力，看不出"模型本身的决策场强不强"。这正是 C3 要分辨的事。

---

## 5. 测试矩阵（16 次评测）

| # | 数据集 | 口径 | M0 | M1 | baselines | 对应结论 |
|---|---|---|---|---|---|---|
| 1 | D1 chengdu | R3 | ✅ | ✅ | ✅ | C2 |
| 2 | D2 chengdu_long | R3 | ✅ | ✅ | ✅ | **C1** |
| 3 | D2 chengdu_long | R8 | ✅ | ✅ | — | C3 |
| 4 | D2 chengdu_long | R16 | ✅ | ✅ | — | C3 |
| 5 | D3 xian | R3 | ✅ | ✅ | ✅ | C2 |
| 6 | D4 chengdu shuffled | R3 | ✅ | ✅ | — | 防质疑 |
| 7 | D5 chengdu_long shuffled | R3 | ✅ | ✅ | — | 防质疑 |
| 8 | D6 xian shuffled | R3 | ✅ | ✅ | — | 防质疑 |

`baselines` = 额外跑 shortest_path / greedy_bfs / random_greedy 三个传统基线，
并做**配对**显著性检验（`paired_vs_baselines` 段），用于论文"模型 vs 传统方法"那张表。

---

## 6. 指标定义

设第 $i$ 条 query 的预测路径为 $P_i$、真实观测路径为 $G_i$，
$C(\cdot)$ 为路径的道路长度之和（米），$C^*_i$ 为同一 OD 的 Dijkstra 最优 cost，$N$ 为 query 总数。

### 6.1 到达类（一级指标，最可信）

| 指标 | 定义 |
|---|---|
| `goal_hit_rate` | $\frac{1}{N}\sum_i \mathbb{1}[P_i \text{ 完整走到 goal}]$。strict 口径下 $P_i$ 还必须每条边都存在于图中（否则记 `broken`） |
| `broken_rate` | $P_i$ 含**图中不存在的边**的占比（结构非法，最严重的失败） |
| `loop_rate` | 解码因成环终止的占比 |
| `coverage_rate` | 候选池里**至少有一条**路径到达 goal 的 query 占比。strict 下与 `goal_hit_rate` 同值（候选池只剩 success），非 strict 下会更宽 |
| `optimal_path_rate` | $C(P_i) = C^*_i$ 的占比（相对+绝对容差均 $10^{-6}$）。**真实数据上只作 secondary** —— GT 本身不是最短路 |
| `optimal_coverage_rate` | 候选池里至少有一条达到 $C^*_i$ 的占比 |

### 6.2 路径相似度类

| 指标 | 定义 |
|---|---|
| `normalized_lcs` (nLCS) | $\frac{1}{N}\sum_i \frac{\mathrm{LCS}(P_i,\,G_i)}{\lvert G_i\rvert}$，最长公共子序列长度按 GT 长度归一化 |
| **`path_similarity_score`** (PathSim) | $\frac{1}{N}\sum_i \mathrm{nLCS}_i \cdot \mathbb{1}[\text{goal\_hit}_i]$ —— **没到终点记 0**。这是选 best.pt 用的指标 |
| `normalized_lcs_success` | 只在**成功到达**的 query 上平均 nLCS（"到了的里面像不像"） |
| `edge_precision` / `edge_recall` / `edge_f1` | 把两条路径看成**无向边集合**，算 P / R / F1。对"多走了几条街"比 nLCS 更敏感 |
| `dtw_km` | 预测与 GT 的 **km 级 DTW**（用真实经纬度算球面距离，序列对齐后取平均点距，单位 km）。全样本均值 |
| `dtw_km_success` | 同上，只在成功样本上（失败路径的 DTW 没有可比性） |

> `path_similarity_score` 与 `normalized_lcs` 的区别是关键：
> 前者惩罚"没到终点"，后者不惩罚。历史 best readout 出过 bug 就是因为混用了这两个。

### 6.3 代价类

| 指标 | 定义 |
|---|---|
| `pred_over_gt_cost_ratio` | $\frac{1}{N}\sum_i \frac{C(P_i)}{C(G_i)}$ —— **预测路径相对真实路径的长度比**。>1 表示绕远 |
| `gt_cost_ratio` | $\frac{1}{N}\sum_i \frac{C(G_i)}{C^*_i}$ —— 数据本身的属性（司机比最短路长多少），**与模型无关**，用于说明任务难度 |
| `success_cost_ratio` | 只在成功样本上 $\frac{C(P_i)}{C^*_i}$ |

### 6.4 分布类（数据集级，非单样本）

| 指标 | 定义 |
|---|---|
| `jsev` | $\mathrm{JS}(p_{gt} \,\|\, q_{pred})$，边访问频率分布的 Jensen–Shannon 散度。越小越好 |
| `klev` | $\mathrm{KL}(p_{gt} \,\|\, q_{pred})$。JSEV 更稳，KLEV 在 support 不一致时会爆 |

两者都是**数据集级**统计，样本少于几百条时会抖，不要在小 split 上引用。

### 6.5 成本类

| 指标 | 定义 |
|---|---|
| `mean_elapsed` | 单条 query 平均解码耗时（秒） |
| `wall_time` | 整次评测墙钟时间（秒）。**不进论文表格**，只做 sanity check |

### 6.6 指标可信度分级

- **一级（可直接下结论）**：`goal_hit_rate`、`broken_rate`、`loop_rate`、`pred_over_gt_cost_ratio`、`normalized_lcs`、`edge_f1`
- **二级（需配合语境）**：`optimal_path_rate` / `optimal_coverage_rate`（真实 GT 非最短路）、`dtw_km`（受坐标补齐影响，111/2891 个节点坐标由邻居插值）
- **弱（只做辅助）**：`jsev` / `klev`

---

## 7. 执行

一条命令跑完整套 16 次评测：

```bash
python scripts/run_eval_protocol.py                    # 跑缺失的，已存在的跳过
python scripts/run_eval_protocol.py --force            # 全部重跑
python scripts/run_eval_protocol.py --only M1 --datasets D2 --beams 3
python scripts/run_eval_protocol.py --dry-run          # 只打印将执行什么
```

产物统一落在 **`outputs/eval/`**，文件名可反查模型 × 数据集 × 口径：

```
outputs/eval/<dataset>__beam<N>__<model>.json
例如：outputs/eval/chengdu_long_test1000__beam3__finetuned.json
      outputs/eval/chengdu_test1000__beam16__base.json
outputs/eval/protocol_index.json      本次协议的总索引（每次运行覆盖）
```

> 评测结果**不再**写进 `outputs/runs/<run>/`。原因是 run 目录只应保存
> "训练产出的东西"（权重 / 曲线 / 配置），评测是**跨模型**的对比，
> 放在一起会让"同一个模型多套口径"和"同一次口径多个模型"混在一个目录里。

单次评测的底层命令（协议脚本内部就是调它）：

```bash
python scripts/evaluate.py \
  --config configs/didi_chengdu.yaml \          # 西安数据集换成 configs/didi_xian.yaml
  --checkpoint outputs/runs/<run>/best.pt \
  --data data/didi/graph/<dataset>.pkl \
  --out outputs/eval/<dataset>__beam<N>__<model>.json \
  --beam-width <N> --device cuda --no-progress
```

`--config` 决定**坐标文件**（DTW 用）与**图文件**（`paths.data_dir/graph_global.pkl`），
所以成都数据集配 `didi_chengdu.yaml`、西安配 `didi_xian.yaml`，不能混。
解码口径（strict / top_k / null_policy / deterministic）全部由 config 的 `evaluation.*` 提供，
命令行只覆盖 `--beam-width`，避免手滑改坏尺子。

---

## 8. 结果汇总

```bash
python scripts/report_eval_protocol.py          # 生成 outputs/eval/PROTOCOL_REPORT.md
```

报告按 §5 的矩阵逐格列出两个模型的指标与差值，并自动跑 §9 的验收判定。

---

## 9. 验收标准

对照 `LARGE_GRAPH_FINETUNE_PLAN.md` 第 22–23 节，**六条必须同时成立**才算微调成功：

| # | 判据 | 含义 |
|---|---|---|
| 1 | D2 / R3 的 `goal_hit_rate` **明显高于** 0.782 | 大图能力真提升 |
| 2 | D2 / R3 的 `broken_rate` **明显低于** 0.218 | 结构性失败减少 |
| 3 | D2 / R3 的 `path_similarity_score` **明显高于** 0.116 | 路径形态更像真实轨迹 |
| 4 | D2 / R3 的 `pred_over_gt_cost_ratio` **明显低于** 1.35 | 不再绕远 |
| 5 | D1 / R3 的 `goal_hit_rate` **不下降**（≥0.99） | 无灾难性遗忘 |
| 6 | D3 / R3 的 `goal_hit_rate` **不下降**（≥0.99） | 跨城市泛化保留 |

（上表右列的 0.782 / 0.218 / 0.116 / 1.35 是 M0 的基线值，写死为验收阈值。）

**C3 的附加判据**：记 $g(b)$ 为 D2 在 beam $b$ 下的 GoalHit，
要求微调后 $g(16) - g(3)$ **显著变小** —— 即小 beam 已经接近上限，
说明提升来自模型的概率场而不是搜索预算。

---

## 10. 注意事项

1. **两个模型必须用同一份代码跑**。历史上出现过同一配置两次运行差 1 条 query 的情况
   （`repeat_passes` 参数加入前后），定稿前务必整套重跑，不要混用不同时期的 JSON。
2. **D4–D6 不要报 PathSim**，GT 是占位符（见 §3）。
3. **D2 的 `mean_optimal_cost ≈ 8455 m`**，而 `gt_cost_ratio ≈ 1.38`；
   引用 `optimal_path_rate` 时必须同时给出 `gt_cost_ratio`，否则会被误读成"模型很差"。
4. **坐标是补齐过的**（2780/2891 实测 + 111 邻居插值），DTW 的绝对精度受此限制；
   跨模型比较没问题（同一套坐标），但不要与其他论文的 DTW 绝对值直接比。
5. **`wall_time` 不是性能指标**，GPU 型号/负载一变就不可比。
6. 所有数字必须来自 `outputs/eval/` 下的 JSON；**手抄到论文里的每个数字都要能反查到文件名**。
