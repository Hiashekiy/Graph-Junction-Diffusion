# Graph-Junction-Diffusion V2

**Edge-State Conditioned Recurrent Graph Flow Denoiser**

V2 是一个完整的重写：底层任务仍然是 *Junction Categorical Diffusion*（在每个
decision node 上对 Branch Segment 做离散扩散），但去噪网络换成了全新的信息流设计：

```
Node = Information                    H_t 承载并累积图上的信息
Edge State = Information-Flow Condition   z_t 展开成 selected / unselected 边状态
Reverse Diffusion = Recurrent Graph Information Propagation
```

核心闭环：

```
z_t  ->  E_t = Psi(z_t)  ->  H_{t-1} = F_theta(H_t, E_t, tau_t)  ->  p_theta(z_0)  ->  z_{t-1}
```

递推状态是 **(H_t, z_t)** —— 节点信息跨 reverse step 完整保留，边状态持续去噪。

设计文档：

- `docs/Graph-Junction-Diffusion_Edge-State-Recurrent-Graph-Flow_网络设计报告_V2.1.md`
- `docs/Graph-Junction-Diffusion_V2_代码实施指南.md`

---

## 0. 速查（数据集 / 模型 / 指标 / 全部指令）

> 下面所有命令都是**单行**、不带 shell 变量、不带续行符，可以直接粘贴到 Git Bash 里执行。
> 解释器统一写成 `E:/CondaEnvData/envs/GGMPC/python.exe`（PATH 里的 `python` 没装 torch）。

### 0.1 数据集（`data/` 下按来源分 6 组，被 .gitignore 忽略，不进版本库）

文件名一律保持不变——历史 `eval*.json` / `mp_*.json` 是按**文件名**记录数据集的，改名会让
它们对不上号——只是按来源挪进子目录，逐份索引见 `data/README.md`。

编号空间与语义见第 3 节；这里只说"每份文件是什么、多大规模、拿来干什么"。

**A. 当前主力数据集 —— `data/controlled/`（`controlled_junction` 生成器，骨架 + 干扰分支）**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `controlled_train.pkl` | 主训练集（3000 样本按图 8:1:1 切出来的 train） | 2400 条 / 每 query 6.53 决策 / 34.7 候选 | 旧版 run 的训练集 |
| `controlled_val.pkl` | 主验证集 | 300 条 / 6.55 决策 | 旧版 run 的验证集 |
| `controlled_test.pkl` | **标准测试集** | 300 条 / 6.49 决策 | 所有 run 都在这上面报 test 指标 |
| `controlled_summary.json` | 上面三份的生成统计（决策数/候选数/长度分布） | — | 查数据分布 |

**B. 长链（决策数 ≥ 9）—— `data/long/`**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `controlled_long.pkl` | **长链专测集**（全 hard，9–11 决策） | 400 条 / 9.20 决策 / 26.0 跳 | 检验"多轮交流"和长链能力 |
| `controlled_longpool.pkl` | 长链候选池（生成时只留 ≥9 决策） | 1050 条 / 9.22 决策 | 给训练集补长链样本 |
| `controlled_longmix_train.pkl` | `controlled_train` + 池子里 900 条 | 3300 条 / 7.26 决策 / ≥9 决策占 30.9% | `v2_rev2_longmix` 的训练集 |
| `controlled_longmix_val.pkl` | `controlled_val` + 池子里 150 条 | 450 条 / 7.44 决策 | 对应的验证集 |

**C. 旧 V1 数据 —— `data/oldv1/`（随机图 er/ba/ws/geometric，20–80 节点，候选=下一跳边；
原始 `.pt` 在 `data/processed/v1/`）**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `data/processed/v1/train.pt` | V1 原始 train（torch dict：`graphs` / `queries` / `meta`） | 5000 图 / 25000 query | V1 格式原始数据 |
| `data/processed/v1/val.pt` | V1 原始 val | 500 图 / 2500 query | 同上 |
| `data/processed/v1/test.pt` | V1 原始 test | 500 图 / 2500 query | 同上 |
| `data/processed/v1/ood_size.pt` | V1 规模外推集（**100–200 节点**） | 300 图 / 1500 query | 只做规模外推测试，**不进训练** |
| `data/processed/v1/manifest.json`、`*_meta.json` | 上面四份的统计 | — | 查数据分布 |
| `oldv1_train_sub.pkl` | V1 train 随机抽 2000 条转成 V2 格式（成功 1970） | 1970 条 / 37.9 决策 / 275 候选 | 混入训练集 |
| `oldv1_val_sub.pkl` | V1 val 随机抽 400 条转 V2（成功 390） | 390 条 / 36.5 决策 | 混入验证集 |
| `oldv1_test.pkl` | V1 test 转 V2（成功 2444 / 2500） | 2444 条 / 38.2 决策 / 268 候选 | **跨分布测试集**（旧数据上到底行不行） |

> 本机工作副本里没有保留 `data/processed/v1/*.pt`（V1 时代的归档件），需要时先从归档恢复，
> 再跑第 18 节的转换命令。

> V1 与 V2 的差别：V1 的候选是"下一跳的边"，V2 是"走到下一个 structural endpoint 的
> branch segment"。所以 V1 的 `.pt` 必须先转换（`tools/convert_v1_dataset.py`）才能喂给
> 现在的模型；约 2.2% 的 query 因为"两个 degree=2 节点构成的三角"无法用 V2 语义表示而被跳过。

**D. 混合训练集 —— `data/mixed/`（当前最新模型用的）**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `mixed_oldv1_train.pkl` | `controlled_longmix_train` + `oldv1_train_sub` | **5270 条** / 18.7 决策 / 127 候选 | `v2_rev2_mixed` 的训练集 |
| `mixed_oldv1_val.pkl` | `controlled_longmix_val` + `oldv1_val_sub` | **840 条** / 20.9 决策 | 对应的验证集（模型选择用它） |
| `mixed_oldv1_{train,val}_summary.json` | 合并统计（组成、决策数直方图、长链占比） | — | 查合并结果 |

**E. 冒烟小数据集 —— `data/smoke/`**：`smoke_{train,val,test}.pkl`（26 / 3 / 3 条），只用来快速跑通流程。

**F. 公开图数据 —— `data/public_graphs/`**：由 `scripts/download_graph_datasets.py` 下载的
DIMACS9/10 路网（`--profile recommended` 约 14 MB：rome99、Luxembourg OSM、NY / BAY / COL），
另有 `--profile snap`（roadNet-CA/PA/TX）与 `--profile all`（再加 CLRS30）可选。

DIMACS9 的 `.gr` 只有弧长、不带坐标，加 `--with-coords` 会连 `*.co.gz` / `*.xyz.bz2` 坐标
伴随文件一起下；有了坐标 `tools/visualize_public_graphs.py` 才能把它们画成地图（没有坐标的
rome99 退化成力导向布局）。文件清单见 `data/README.md`。

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --list
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --with-coords
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py
```

**G. Weighted 数据集 —— `data/weighted_controlled/`（第 20 节的带权扩展）**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `weighted_controlled_train.pkl` | 加权主训练集（3000 条按图 8:1:1 切出来的 train） | 2400 条 | `v2_weighted_controlled` 的训练集 |
| `weighted_controlled_val.pkl` | 加权验证集 | 300 条 | 模型选择 |
| `weighted_controlled_test.pkl` | 加权标准测试集 | 300 条 | 带权能力的主指标 |
| `weighted_controlled_summary.json` | 生成统计 + **weighted sanity check**（conflict rate / bfs cost ratio / 权重分布） | — | 判断这份数据到底难不难 |

每条边 w ~ U(1, 10)（可用 `data.edge_weight.distribution` 换 loguniform），GT 是 Dijkstra 最小 cost 路径。
实测：conflict rate 0.460、bfs cost ratio 1.034、`gt_path_is_weighted_optimal_fraction` 1.0。

**H. DiDi 成都真实道路数据集 —— `data/didi_chengdu_gjd/`（第 24 节，**GT 是真实车辆历史路径**）**

| 文件 | 是什么 | 规模 | 用途 |
|---|---|---|---|
| `train.pkl` / `val.pkl` | 真实路网 OD 训练/验证集 | 4687 / 533 条 | `didi_chengdu_flow1_weighted` |
| `test.pkl` | 完整测试集 | 1275 条 | 全量 test 指标 |
| `test_1000.pkl` | GDP 风格固定子集 | 1000 条 | 论文主表 |
| `shuffled_od_1000.pkl` | OD 打乱重配，**无真实 GT** | 741 条 | 只测 GoalHit/Loop/Broken/CostRatio/时间 |
| `graph_global.pkl` | 全局无向有权成都路网 | 2891 节点 / 4403 边 | 建图复现与可视化 |
| `split_manifest.csv` / `metadata.json` / `stats.json` | 逐样本清单 / 建图与 rho 元数据 / 清洗漏斗与各 split 摘要 | — | 可复现性与数据质量 |

GT 是 CSV 里真实车辆走过的路径（93.7% 都不是最短路，cost ratio 中位数 1.12），
corridor 用 rho=1.5 的 OD 椭球。生成命令与全部实测结论见**第 24 节**。

### 0.2 训练好的模型（`outputs/runs/`，checkpoint 不进版本库）

| run 目录 | 网络/推理 | 训练集 | 训练规模 | best epoch | val goal_hit | `controlled_test`(300) goal / optimal | `controlled_long`(400) goal / optimal | `oldv1_test`(2444) goal / optimal |
|---|---|---|---|---|---|---|---|---|
| `outputs/runs/v2_controlled_100ep` | 旧 attention（K 只吃 edge），flow_steps=1 | `controlled_train` | 80 epoch | 无 epoch 标注 | 0.9067 | 0.8933 / 0.7333 | 0.601 / 0.453（5 种子） | 0.605 / 0.380 |
| `outputs/runs/v2_controlled_100ep_flow3` | 旧 attention，flow_steps=3 | `controlled_train` | 100 epoch | 80 | 0.8900 | 0.9167 / 0.7233 | 0.6685 / 0.5335（5 种子） | 0.604 / 0.367 |
| `outputs/runs/v2_rev2_longmix/v2_rev2_longmix` | 新 attention（node+edge K）+ Soft Goal，flow_steps=3 | `controlled_longmix_train` | 100 epoch（52 s/ep） | 75 | 0.9444 | 0.9300 / 0.9033 | 0.9125 / 0.8850 | 0.2275 / 0.1911 |
| **`outputs/runs/v2_rev2_mixed`**（最新） | 新 attention + Soft Goal + horizon cap=24，flow_steps=3 | `mixed_oldv1_train` | 100 epoch（183 s/ep） | **90** | **0.9548** | 0.9300 / 0.8833 | 0.8950 / 0.7900 | **0.9763 / 0.9677** |
| **`outputs/runs/v2_weighted_controlled`**（第 20 节，带权） | 新 attention + Soft Goal + **Edge Cost Encoder**，flow_steps=3 | `weighted_controlled_train`（w~U(1,10)） | 100 epoch（~120 s/ep） | 95 | 0.8233 | — | — | — |
| `outputs/runs/v2_weighted_controlled_cost_ablated`（第 20 节的 cost 消融） | 新 attention + Soft Goal，**看不到 edge cost** | 同一份带权训练集 | 100 epoch（~120 s/ep） | 75 | 0.8033 | — | — | — |

> 两个 weighted run 的评测集不是上面三个（那些是无权图），而是 `data/weighted_controlled/weighted_controlled_test.pkl`：
> 见第 20.8 节 —— Ours `goal 0.8633 / optimal 0.6900 / cost_ratio 1.0070`，cost 消融 `0.8767 / 0.3667 / 1.0557`，
> Greedy-BFS `1.0000 / — / 1.0350`，Dijkstra oracle `1.0000 / 1.0000 / 1.0000`（逐条配对 p=7e-23）。

> 上表是**单次评测（seed 0）** 的数字；括号里注明的行是 5 种子均值。多种子配对检验的
> 结论（`multiseed_*.json`）：
>
> * flow3 vs flow1：test 打平（−0.008，p=0.44），long 显著更好（+0.068，p=2.6e-7）；
> * longmix vs flow3：test +0.042（p=3.8e-4）、long +0.235（p=6e-50）；
> * mixed vs longmix：test +0.001（p=0.90，打平）、long goal −0.010（p=0.35，打平）
>   但 **long 的 optimal −0.067（p=1.3e-7，显著下降）**、**oldv1 goal +0.746（p≈0，3 种子）**。
>
> 也就是说：把旧数据混进来，**旧数据上从"灾难性退化"变成"几乎全对"，代价是长链上
> "恰好走最短路"的比例掉了 6.7 个点**。
>
> 推理侧还有一招**多分支（存活路径表）解码**（第 19 节，增强版见第 21 节）：`v2_rev2_mixed` 在
> controlled_test 上 goal_hit 0.930 → **1.000**、optimal 0.883 → **0.973**（k=2 + NULL 不停），
> controlled 两个测试集的 coverage 都是 **1.0000**（路径表里必有一条能到终点）。

### 0.3 指标说明

| 指标 | 含义 |
|---|---|
| `goal_hit_rate` | 解码出的路径最终到达 goal 的比例（**模型选择主指标**） |
| `optimal_path_rate` | 到达 goal **且跳数等于最短路**的比例（另一种等长最短路也算） |
| `success_cost_ratio` | 只在成功的 query 上平均 `预测跳数 / 最短路跳数`，1.0 = 全最优 |
| `loop_rate` | 解码时重复访问 structural node（判 loop）的比例 |
| `broken_rate` | 选了 NULL / branch 撞 dead-end / 超步数上限 的比例 |
| `mean_elapsed` | 每条 query 的平均推理秒数 |
| `soft_goal_reachability` | 软可达性代理指标（可微，训练用它；**不能替代**上面的硬指标） |
| `coverage_rate` | 仅 `--decode multi`：存活路径表里**至少有一条**路径到终点（集合语义上界） |
| `optimal_coverage_rate` | 仅 `--decode multi`：路径表里**至少有一条**最短路 |

失败只有三种情况：`broken:NULL`（在路口选了 NULL 提前停）、`broken:dead-end`
（branch 通往 degree-1 死胡同）、`loop`（重复节点）。旧数据上还有一种特殊形态是
"出门两步折返回 start"（`v2_rev2_longmix` 在 `oldv1_test` 上占 50%，混入旧数据后归零）。

### 0.4 全部指令（单行，可直接粘贴）

**训练**

```bash
# tiny overfit（先验证链路能过拟合；小数据 + 高 lr）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --tiny --name tiny --set diffusion.T=20 --set training.epochs=200 --set training.lr=3.0e-3

# 正式训练：长链偏重（v2_rev2_longmix 用的这条）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_longmix --data data/long/controlled_longmix_train.pkl --val-data data/long/controlled_longmix_val.pkl

# 正式训练：旧数据融合（v2_rev2_mixed，最新，100 epoch 约 5 小时）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_mixed --data data/mixed/mixed_oldv1_train.pkl --val-data data/mixed/mixed_oldv1_val.pkl

# 对照：降低 NULL 权重（混合集 NULL 占 68%，模型容易"该走却选 NULL"）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_mixed_nw03 --data data/mixed/mixed_oldv1_train.pkl --val-data data/mixed/mixed_oldv1_val.pkl --set loss.null_weight=0.3

# Weighted 扩展：主实验（weighted GT + Edge Cost Encoder；命令细节见第 20 节）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow_weighted.yaml --name v2_weighted_controlled --data data/weighted_controlled/weighted_controlled_train.pkl --val-data data/weighted_controlled/weighted_controlled_val.pkl

# Weighted 扩展：关键消融（同一份 weighted GT，但不给模型 edge cost）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow_weighted.yaml --name v2_weighted_controlled_cost_ablated --data data/weighted_controlled/weighted_controlled_train.pkl --val-data data/weighted_controlled/weighted_controlled_val.pkl --set model.use_edge_cost=false

# 续训（从 last.pt 再跑 30 轮；--extra-epochs 是"在已跑轮数之上再加多少"）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_mixed --data data/mixed/mixed_oldv1_train.pkl --val-data data/mixed/mixed_oldv1_val.pkl --resume outputs/runs/v2_rev2_mixed/last.pt --extra-epochs 30
```

**评测**

```bash
# 标准测试集
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/controlled/controlled_test.pkl --device cuda --no-progress --baselines --out outputs/runs/v2_rev2_mixed/eval_test.json

# 长链专测集
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/long/controlled_long.pkl --device cuda --no-progress --out outputs/runs/v2_rev2_mixed/eval_long.json

# 旧 V1 数据（跨分布）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/oldv1/oldv1_test.pkl --device cuda --no-progress --out outputs/runs/v2_rev2_mixed/eval_oldv1_test.json

# 多分支（存活路径表）解码：主指标取累计概率最高的路径，额外报 coverage
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/long/controlled_long.pkl --device cuda --no-progress --decode multi --top-k 2 --null-policy skip --out outputs/runs/v2_rev2_mixed/eval_long_multi.json

# 多分支增强（第 21 节）：必死 branch 预筛选 + 三条口径（best / best_goal / best_goal_cost）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_weighted_controlled/run_config.json --checkpoint outputs/runs/v2_weighted_controlled/best.pt --data data/weighted_controlled/weighted_controlled_test.pkl --device cuda --no-progress --decode multi --top-k 2 --null-policy skip --filter-dead-branches --out outputs/runs/v2_weighted_controlled/eval_test_multi.json

# 三套口径一起出（单路径 / 多分支 best / coverage）
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_multipath.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --top-k 2 --beam-width 64 --null-policy skip --out outputs/runs/v2_rev2_mixed/mp_long_k2_skip.json

# 多种子配对检验（A=基线 run，B=新 run；--decode multi 时两个 run 都用同一模式）
E:/CondaEnvData/envs/GGMPC/python.exe tools/multiseed_eval.py --a outputs/runs/v2_rev2_longmix/v2_rev2_longmix --b outputs/runs/v2_rev2_mixed --data data/oldv1/oldv1_test.pkl --seeds 0,1,2 --metric goal_hit --out outputs/runs/v2_rev2_mixed/multiseed_oldv1_vs_prev.json

# 按难度 / 结构模式 / 决策数 / source 类型拆桶
E:/CondaEnvData/envs/GGMPC/python.exe tools/breakdown_eval.py outputs/runs/v2_rev2_mixed/eval_oldv1_test.json --data data/oldv1/oldv1_test.pkl

# 训练曲线按真实 epoch 对齐比较
E:/CondaEnvData/envs/GGMPC/python.exe tools/compare_curves.py --a outputs/runs/v2_rev2_longmix/v2_rev2_longmix --b outputs/runs/v2_rev2_mixed --label-a longmix --label-b mixed --every 5
```

**可视化 / 取路径**

```bash
# 单路径：随机抽 16 张画图（不加 --select-seed 每次抽的不一样，实际种子会打印）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --num 16 --cols 2 --labels --select random --select-seed 0 --out outputs/figures/paths_mixed_long.png

# 多分支：细线画出整张存活路径表，粗线是累计概率最高的那条
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --num 8 --cols 2 --labels --select random --select-seed 0 --multi-k 2 --null-policy skip --out outputs/figures/paths_mixed_long_multi.png

# 只看失败样本（broken / loop / optimal / goal 任选）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --num 8 --cols 2 --labels --select broken --out outputs/figures/paths_mixed_long_broken.png

# 指定下标（最稳的复现方式）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_rev2_mixed --data data/controlled/controlled_test.pkl --num 4 --cols 2 --labels --select indices --indices 3,7,42,101 --out outputs/figures/paths_mixed_indices.png

# 只要路径文本 / JSON，不画图
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_rev2_mixed --data data/oldv1/oldv1_test.pkl --index 1242
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --index 111 --json
```

**交互看板（加权模型 + 报告页签）**

```bash
# 启动（默认 http://127.0.0.1:8765/，自动打开浏览器；--no-browser 则不打开）
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py
```

看板有三个页签：路径可视化 / 测试指标可视化 / **实验报告**。指标页含 `multi`、
`multi · best_goal`、`multi · best_goal_cost` 与 `multi k=N / stop|skip` 等口径，并对每个模型标出
**带权 / 带权·无cost / 无权**；报告页直接读 `docs/REPORT_multipath_and_weighted.md` 与
`outputs/*summary*.json`。细节见第 22 节与 `dashboard/README.md`。

**数据（生成 / 转换 / 合并 / 查泄漏）**

```bash
# 重新生成主数据集（当前数据都在，一般不需要重跑）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml

# 看数据语义（branch / z0 / batch 形状）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/inspect_dataset.py --data data/controlled/controlled_train.pkl --limit 3

# V1 -> V2 转换（--sample 随机抽样；不能用 --limit，V1 的 query 按图类型分块存）
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/test.pt --out data/oldv1/oldv1_test.pkl
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/train.pt --out data/oldv1/oldv1_train_sub.pkl --sample 2000 --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/val.pt --out data/oldv1/oldv1_val_sub.pkl --sample 400 --seed 0

# 合并数据集（--sample N:INDEX 表示第 INDEX 份输入随机抽 N 条）
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_train.pkl --input data/long/controlled_longmix_train.pkl --input data/oldv1/oldv1_train_sub.pkl --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_val.pkl --input data/long/controlled_longmix_val.pkl --input data/oldv1/oldv1_val_sub.pkl --seed 0

# 图级泄漏检查（必须 0 重叠才能训；有任何一对重叠就退出码 1）
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --pair data/mixed/mixed_oldv1_train.pkl data/mixed/mixed_oldv1_val.pkl --pair data/mixed/mixed_oldv1_train.pkl data/oldv1/oldv1_test.pkl --pair data/mixed/mixed_oldv1_train.pkl data/controlled/controlled_test.pkl
```

**DiDi 成都真实道路数据（第 24 节）**

```bash
# 阶段 0：扫描原始文件，确认道路长度列名 / 转换率 / GT cost ratio（不建数据）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --scan-only

# 阶段 1：在 train split 上比较 rho 候选，选 corridor 参数并冻结进配置
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --scan-corridor

# 阶段 2：生成 train/val/test/test_1000/shuffled_od_1000（约 4 分钟，产出 ~1GB）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --build

# 自检（不需要 pytest / torch，20 条断言覆盖转换 / corridor / 指标 / 已生成数据集）
E:/CondaEnvData/envs/GGMPC/python.exe tools/verify_didi_pipeline.py --data data/didi_chengdu_gjd
```

**测试与静态检查**

```bash
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
E:/CondaEnvData/envs/GGMPC/python.exe tools/semantic_check.py --strict
```

---

## 1. 安装

```bash
pip install -r requirements.txt
```

依赖：`torch`、`networkx`、`numpy`、`PyYAML`、`pytest`。

## 2. 目录结构

```
Graph-Junction-Diffusion/
├── configs/graph_flow.yaml          第一版配置
├── src/
│   ├── data/                        graph_generators / branch_segments / decision_field
│   │                                dataset_builder / dataset / collate
│   ├── diffusion/                   schedule / posterior / categorical / sampler
│   ├── models/                      node_embedding / edge_state / time_encoder
│   │                                graph_flow / branch_scorer / denoiser
│   ├── training/                    setup / losses / trainer / checkpoint
│   ├── evaluation/                  path_decoder / metrics / evaluator / baselines
│   └── utils/                       config / nn / seed / segment_ops
├── scripts/                         generate_dataset / inspect_dataset / train / evaluate
├── tools/                           smoke_tiny_overfit（tiny 过拟合验收）
│                                    semantic_check（AST 静态检查，不依赖 torch）
└── tests/                           见第 6 节
```

## 3. 数据语义（最重要的一层）

节点只有四类：`0=Ordinary 1=Junction 2=Start 3=Goal`（Start/Goal 优先级高于度数）。

Decision set：`D = (J ∪ {s | deg(s) > 1}) \ {g}`。

**候选不再是"下一个邻居"，而是一整条 Branch Segment：**

```
J1 ─ a ─ b ─ J2        branch.nodes = [J1, a, b, J2]
                       branch.physical_edges = [(J1,a), (a,b), (b,J2)]
```

- 普通 Junction 的类别集合是 `{NULL, B1..BK}`；
- Source 没有 NULL（`{B1..BK}`）；
- Goal 永远不是 decision node；
- 物理边（无向，唯一 ID）与 message edge（两个方向）严格分开，
  两个方向永远共享同一个 edge state。

`z_0`：GT path 上经过的 decision node 指向与 path 后续完全一致的那条 branch，
没经过的 junction 一律 NULL。这些语义在构造数据集时被强制校验
（`validate_decision_field`），不合法就丢弃该 OD 对。

## 4. 网络结构（与设计报告 V2.1 逐条对应）

| 设计报告 | 实现 |
|---|---|
| `H_T = TypeEmbedding(V)` | `models/node_embedding.py`，只初始化一次 |
| `z_t -> E_t` 五种展开步骤 | `models/edge_state.py::expand_to_edge_state` |
| `tau_t` 正弦编码 + MLP | `models/time_encoder.py::TimeEncoder` |
| AdaLN：`h_hat = (1+gamma) LN(h) + beta` | `models/graph_flow.py` + `TimeConditioner` |
| `Q = W_Q h_v`（接收节点） | `graph_flow.py::q_proj(H_hat[dst])` |
| `K = W_{K,n} h_u + W_{K,e} e_uv`（发送节点 + 边状态，第二轮修订 A） | `graph_flow.py::k_node_proj(H_hat[src]) + k_edge_proj(edge_feat)`，再乘 `1/sqrt(2)` |
| `V = W_V h_u`（发送节点） | `graph_flow.py::v_proj(H_hat[src])` |
| 每条入边 softmax | `segment_softmax(score, dst, num_nodes)` |
| Start/Goal 只出不进 | `valid_msg = ~fixed_mask[dst]`，并从聚合里剔除 |
| `H~ = LN(H + W_O m)`，再 `LN(H~ + FFN(H~))` | `graph_flow.py` residual 更新 |
| Start/Goal 状态 clamp | `torch.where(fixed_mask, H_t, H_new)` |
| 一步 reverse step 内部的图信息交流轮数 | `denoiser.py::step` 调 `graph_flow.forward_multi`，轮数 = `model.flow_steps` |
| 所有 timestep 共享 `F_theta` | 只有一个 module 实例，参数与 T 无关 |
| Branch Mean Pool（排除 owner） | `branch_scorer.py::branch_mean_pool` |
| `[h_i, h_bar_ik, tau_t] -> 1 logit` | `BranchScorer.branch_mlp`（3d -> 2d -> 1） |
| `[h_i, tau_t] -> 1 logit` | `BranchScorer.null_mlp`（2d -> d -> 1） |
| grouped softmax | `utils/segment_ops.py::grouped_log_softmax` |
| categorical posterior 数学不变 | `diffusion/posterior.py`（原样保留） |
| sampler 状态是 `(H_t, z_t)` | `diffusion/sampler.py` |
| recurrent reverse-chain 训练 | `training/losses.py::recurrent_reverse_loss` |

**第一版刻意不包含**：LapPE、RWSE、DegreeEmbedding、Node-ID Embedding、
Global Pool、Static Graph Encoder、旧 Routing-State Encoder。

> **与设计报告 V2.1 的唯一一处有意偏离**：报告第 10 节写的是
> "1 个 Graph Flow Block / Reverse Step"（并把"完整 reverse chain 提供 50 轮信息
> 传播"当作足够），现在改成 `model.flow_steps` 轮 / step（默认 3）。报告第 11 节的
> "所有 timestep 共享同一个 `F_theta`" 仍然成立 —— 轮次之间也共享同一个 Cell，
> 各轮靠 `SlotEmbedding(k)` 注入 AdaLN 条件区分（详见第 14 节）。若把
> `model.flow_steps` 设为 1，行为与报告 V2.1 的原始设计逐位一致。

## 5. 使用

> **命令速查见第 0.4 节**（数据集、模型、指标、全部指令都整理在那里，且都是单行、
> 可直接粘贴到 Git Bash）。这里只留一条最小闭环，其余用速查表里的命令。

```bash
# 看数据语义 -> 训练 -> 评测（以当前最新的 run 名 v2_rev2_mixed 为例）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/inspect_dataset.py --data data/mixed/mixed_oldv1_train.pkl --limit 3
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_mixed --data data/mixed/mixed_oldv1_train.pkl --val-data data/mixed/mixed_oldv1_val.pkl
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/controlled/controlled_test.pkl --device cuda --no-progress --baselines --out outputs/runs/v2_rev2_mixed/eval_test.json
```

旧的示例（`data/er_256_train.pkl`、`outputs/runs/graph_flow`）是更早一版的数据与 run 名（这些文件本机已不存在），
现在都不存在了；完整的历史命令留在第 11–16 节的记录里，仅作过程留档。

## 6. 测试

```bash
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
```

覆盖实施指南第 25 节的全部条目：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_branch_segments.py` | `J1-a-b-J2` 得到完整 segment；decision set；物理边/message 边分离 |
| `test_decision_field.py` | NULL / source 无 NULL / off-path 为 NULL；validator 六条检查 |
| `test_edge_state.py` | selected 覆盖整条 branch；NULL 不写 selected；两方向同状态 |
| `test_node_embedding.py` | 同类型节点初始向量相同；四种类型互不相同 |
| `test_graph_flow.py` | Q/K/V 角色、grouped attention、Start/Goal clamp、residual、AdaLN |
| `test_branch_scorer.py` | mean pool 排除 owner、NULL/source 类别、grouped softmax 和 = 1 |
| `test_persistent_state.py` | `H_T->H_{T-1}->H_{T-2}` 传递、参数共享、梯度穿链 |
| `test_reverse_chain.py` | 完整 T 步链、forward trajectory（alpha_t）、decoder、训练能降 loss |
| `test_diffusion.py` | categorical 前向/后验数学（原 V1 测试，全部保留） |
| `test_grouped_softmax.py` | ragged grouped softmax / segment reduction |
| `test_config.py` / `test_config_effects.py` | 配置项真的生效（`d_time` / `flow_steps` / `amp` / schedule） |
| `test_controlled_graph.py` | 生成器难度契约、GT 重算、干扰分支类型 |
| `test_split_and_relabel.py` | 按 graph 划分（无 topology leakage）、节点重编号 |
| `test_source_forced.py` | 单出口 source 的被迫段永久 selected |
| `test_buckets.py` | 评测分桶口径（cost 均值只用成功样本、空桶 NaN、分桶恰好划分全量） |
| `test_checkpoint_mismatch.py` | checkpoint 与模型结构不匹配时的可读报错（flow_steps 必须一致）、legacy `k_proj` 的无损映射 |
| `test_soft_goal.py` | 软可达性（0.8x0.7=0.56、NULL/dead-end 贡献 0、可微、不串图、horizon cap、`goal_reach_weight=0` 等于纯 CE） |
| `test_multi_path_decoder.py` | 存活路径表解码（k=1 等价贪心、k=2 救回丢掉的路、NULL stop/skip、beam 截断、排序与可复现） |

## 7. 评测指标

主指标（模型选择用）：

- **Goal Hit Rate** = 到达 goal 的 query 数 / 总 query 数
- **Optimal Path Rate** = 成功且路径 cost 等于 GT 最短路的比例
- **Success Cost Ratio** = `L_pred / L_optimal`，**只在成功样本上统计**
- **Loop Rate / Broken Rate / Inference Time**

另外两个只在新版本里出现的指标：

- **Soft Goal Reachability**：可微的软可达性代理指标（训练损失用它，也写进 `history.json`），
  **不能替代**上面的硬指标；
- **Coverage Rate / Optimal Coverage Rate**：只有 `--decode multi`（存活路径表）才有，
  表示"路径表里至少有一条到终点 / 至少有一条最短路"，是集合语义的上界。

`decision accuracy` 只是 debug 指标，**不能用来选模型**（它和端到端表现严重脱节）。
注意它还分两类：NULL 决策占了 80% 以上，所以整体 accuracy 会看起来很漂亮 ——
真正决定路径的是**非 NULL（active）决策**的准确率。

完整指标清单（含每个指标的确切含义）见 **第 0.3 节**。

## 8. 已验证的落地状态

在一台有 torch 2.9 / CUDA 的机器上实跑过（数据规模很小，仅用于验证链路）：

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q` | **262 passed**（含 soft goal / 多分支解码 / legacy checkpoint 映射 / 分桶口径） |
| 静态检查 | `E:/CondaEnvData/envs/GGMPC/python.exe tools/semantic_check.py --strict` | checked 63 modules, **0 problems** |
| 数据生成 | `E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py ...` | 通过，含逐样本语义校验；同 seed 重跑 attempts/accepted 逐位一致 |
| 泄漏检查 | `E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py ...` | 混合训练集 ↔ val / 各测试集：**0 图级重叠** |
| tiny overfit | `E:/CondaEnvData/envs/GGMPC/python.exe tools/smoke_tiny_overfit.py 200 4` | 4/4 goal、optimal **1.0000**、broken 0 |
| 完整链路 | `E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --checkpoint .../best.pt --data ...` | 见第 0.2 节的四个 run |

> 本节是**历史记录**，数字留的是当时的实测量；最新一次全量检查见第 0.2 / 0.4 节。

训练语料与两个 run 的对照见第 13、14 节；`outputs/runs/*/run_config.json` 记录了
每个 run 真正用的配置（`model.flow_steps`、`T`、`batch_size`）。

tiny overfit（`flow_steps=3`，4 个固定 query，本次实测）：

```text
epoch  30: train_loss=0.589  train_x0_acc=0.803  full-chain goal_hit=0.000（4 条全部 broken）
epoch 200: train_loss=0.068  train_x0_acc=1.000  full-chain goal_hit=1.000（optimal 1.000、broken 0）
```

也就是说：**loss 下降 → teacher-forced 单步学会 → 完整 reverse chain 的 Goal Hit 上升**
这条链路是通的。注意两点：默认 `training.lr=1e-4` 在 tiny 集上偏小，验链路时可以先调大；
`flow_steps=3` 比单轮更难早期过拟合（30 epoch 时整链还是全 broken，200 epoch 才 4/4），
与正式训练里"前 20 个 epoch 学得比单轮慢"是同一个现象。

历史参考（`flow_steps=1`、16 个 query、T=10、d_model=32、lr=3e-3、300 epochs）：
`epoch 100 goal_hit=0.250 → epoch 200 = 0.875 → epoch 300 = 1.000（loop=0, broken=0）`。

## 9. 实现过程中踩到的坑（改代码前先读）

这些都是实跑暴露、并且已经修掉 + 加了回归测试的问题：

1. **一维张量上的 `argmax(dim=-1)` 返回 0-dim 张量**。
   `candidate_log_prob` 是 flat `[C]`，`logits.argmax(dim=-1)` 得到的是标量而不是
   逐元素索引，于是 accuracy 会恒定错成"目标恰好是 0 号候选"的比例
   （实测 0.005 vs 真实 0.98）。现在统一用 `losses.grouped_argmax`
   （补齐到等宽候选组 + `topk(1)`，再把组内位置换算回 flat index）。
2. **loss 不能按候选组大小归一化**。`-log p / C_i` 会让均匀分布下的 CE 变成
   `log(C_i)/C_i` 而不是 `log(C_i)`，梯度尺度随候选数漂移。现在是
   `L_t = (1/M) Σ_i w_i·(-log p)`（实施指南第 21 节的原式）。
3. **物理边 ID 必须按图平移**。`GraphSegments` 里的物理边是每图局部编号，
   batch 里必须加 per-graph offset，否则第二张图起会写进/读到别的图的边状态
   （静默串图，不报错）。
4. **`Config.section()` 必须返回 `Config` 而不是 dict**。返回 dict 时
   `.get(key, default)` 走 `Mapping` 的默认实现（默认值 None），会静默吞掉
   CLI 覆盖；同时 `--set` 必须用 `action="append"`，`nargs="*"` 会让重复的
   `--set` 只剩最后一个。
5. **schedule 在 CPU、batch 在 CUDA** 时，`alpha/beta` 常量要显式 `.to(device)`。
6. **decoder 要先走"被迫段"**：度为 1 的 source 不是 decision node，它的第一步
   是唯一通路，必须先从 s 走到第一个 decision/goal，再开始按 z 选 branch；
   批量解码还要同时用 `decision_offset` 与 `candidate_offset`。
7. **`set_rng_state` 要求 CPU ByteTensor**；`torch.load(map_location='cuda')`
   会把 RNG state 搬到 GPU，需要显式搬回 CPU。
8. **`scatter_reduce(..., reduce="amin", include_self=True)` 的初值要给够大的哨兵**。
   用 `-1` 当初值的话 `amin` 永远取到 `-1`，`first` 对每个 decision 都变成 -1
   （source decision 会拿到非法候选）。见 `heuristic_prior_deterministic`。
9. **小数据集的按图划分会出现空 split**。4 张图按 0.8/0.1/0.1 取整时 val/test
   都会是 0 张图，于是 `validate()` 什么都不返回、best-checkpoint 静默失效。
   现在 splitter 会强制每个 split 非空，`build_datasets` 也会对空 split 直接报错。
10. **AMP 下 `masked_scatter` 要求 self 与 source 同 dtype**。fp16 的 `H.new_zeros`
    配上 fp32 的 Linear 输出会报 "expected self and source to have same dtypes"。
11. **`history.json` 丢了前 N 个 epoch 时，`index + 1` 会把所有 epoch 号报小**。
    `v2_controlled_100ep` 的 history 只剩 epoch 21-100 的 80 条，于是
    "best validation: epoch 60" 其实是 epoch 80。真实 epoch 只能从
    `val_records_epoch{N}.json` 的文件名锚定（`tools/summarize_run.py::
    reconstruct_epoch_offset` 会从末尾对齐并用 goal_hit 值核对，核对不过就警告，
    不猜）。新 run 的每条记录都带 `epoch` 字段，不会再踩。
12. **`history.index(record)` 不能用来找"最好那次验证的下标**：两条记录内容完全相同时
    `list.index` 返回前一条，best epoch 会指错。改用 `enumerate`。
13. **不要在训练还在跑的时候测推理时间**（`mean_elapsed` / `sec/query`）。
    同一份基线 checkpoint 在训练占着 GPU 时测出来是 0.040 s/query，空闲时是
    0.022 s/query —— 差了一倍，跟模型无关。跨 run 比较速度必须在同一个空闲窗口里测。
14. **改 `model.flow_steps` 会改参数集合**（每轮一个 `flow_slot_embedding`），
    所以 checkpoint 只能用**训练时那个** `flow_steps` 的配置加载；
    想少跑几轮做 ablation 要用 `--eval-flow-steps`（推理期覆盖），不是改 config。

## 10. 第二轮修订（《当前实现修改清单》P0/P1/P2）

按清单完成并逐条验证，详细语义见 `docs/ARCHITECTURE_V2.md` 第 8 节：

| 项 | 内容 | 关键落点 |
|---|---|---|
| P0-1 | 单出口 Source 的被迫段**永久 selected** | `branch_segments.build_source_forced_segment`、`Batch.source_forced_edge_ids`、`edge_state.expand_to_edge_state` |
| P0-2 | train/val/test **按 graph 划分**（无 topology leakage） | `dataset_builder.split_dataset` + `_allocate_group_shares` |
| P1-1 | `weighted=true` 暂时禁用 | `build_dataset` 抛 `NotImplementedError` |
| P1-2 | 统一 node 编号空间 | `relabel_to_contiguous` + `extract_segments(relabel=False)` |
| P1-3 | 两个 residual 子层 LayerNorm 分离 | `graph_flow.attn_out_norm` / `ffn_out_norm` |
| P2-1 | deterministic prior 改名与语义澄清 | `heuristic_prior_deterministic` |
| P2-2 | `d_time` / `encoding` / `conditioning` / `amp` 真正生效 | `time_encoder.TimeEncoder(d_time=...)`、`trainer._autocast` + `GradScaler` |
| P2-3 | sampler 只能跑完整链 | `sample_reverse_chain` 只接受 `max_steps in (None, T)` |

验收（GGMPC 环境，torch 2.9.1+cu126）：`pytest` **215 passed**、
`semantic_check --strict` **0 problems**、tiny overfit 的 full-chain `goal_hit=1.0000`、
按图划分实测 `train ∩ val = train ∩ test = val ∩ test = ∅`。

## 11. 数据集生成（《V2 数据集生成指南》）

第一版不用"先随机生成图、再碰运气筛长路径"，而是 **Controlled Junction Graph**：

```
采样难度档 / 结构模式 → 选可行的 decision 数 K → 由 K 推出 hop 目标
  → 构造骨架 S-J1-...-JK-G → 每段展开成 Branch Segment（插入 degree=2 普通节点）
  → 加干扰分支（dead-end / detour / loop）→ 重新求真实 GT → 难度过滤 → 接受/拒绝
```

难度由 `(N, L_hop, L_decision, K_avg)` 描述，关键是 **L_decision**（GT 路径上要连续做
多少次 Branch Decision），而不是节点数。代码在 `src/data/controlled_graph.py`。

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml
```

输出 `data/controlled/controlled_{train,val,test}.pkl` + `data/controlled/controlled_summary.json`，后者包含
指南第 16 节要求的全部指标（hops / decisions / branch factor 的 mean-std-min-max、NULL 比例、source-as-decision 比例、三类干扰分支占比、难度与模式配比）。

两个为**专项评测集**加的参数（见第 16 节的长链实验）：

```bash
# 只保留 GT 决策数 >= 9 的样本，且不划分 train/val/test（整份存成一个文件）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml --data-dir data/long --name controlled_long --no-split --min-decisions 9 --set data.num_samples=400 --set seed=7 --set data.difficulty_mix.hard=1.0 --set data.difficulty_mix.easy=0.0 --set data.difficulty_mix.medium=0.0
```

`--min-decisions N` 走 `build_controlled_dataset(min_decisions=N)`：难度过滤通过但决策数
不够的样本会被丢掉，并在 summary 的 `filter.rejected_by_min_decisions` 里记数（长链集的
接受率只有 0.39%，这个数说明"长链样本为什么稀少"）。`--no-split` 只写
`<data-dir>/<name>.pkl` + `_summary.json`（`--data-dir` 决定落进哪个组目录，默认取配置里的
`paths.data_dir`），不碰现有的 train/val/test。

### 与指南表格的两处必要偏离

1. **Easy 档的 `gt_decisions` 下界抬到 5**：指南表格写 2–5，但第 9.2 节的验收契约
   要求 `decisions >= 5`，二者直接冲突。以契约优先，否则 Easy 样本会被 100% 丢弃。
2. **Hard 档收窄为 `hops∈[25,33]`、`decisions∈[6,9]`**：表格写的 `hops<=35 ∧
   decisions>=8` 在数学上几乎无交集 —— 每段至少 1 个 ordinary 节点意味着
   `hop >= 2(K+1)`，`K=12` 时 hop 至少 26 且上界 35，实测接受率只有 0.5%。
   收窄后 hard 档可稳定产出，三档仍保持清晰梯度。

### 生成质量（实测 3000 样本）

| 指标 | 值 |
|---|---|
| 接受率 | 3000 / 28462 次尝试 ≈ 10.5% |
| 生成速度 | 36 samples/s（83.6s 生成 3000 条） |
| GT hop length | mean 19.1（契约 15–35） |
| GT decision depth | mean 6.53（契约 5–12） |
| Average branch factor | mean 4.37（契约 2–5） |
| NULL fraction | 0.177 |
| source-forced / source-as-decision | 56% / 44%（目标 70/30，单出口更容易通过过滤） |
| 干扰分支比例 | dead-end 70% / detour 23% / loop 7% |
| topology leakage | **无**（三个 split 的 graph_id 两两不相交） |
| 可复现性 | 同 seed 重跑得到完全相同的 `attempts=28462 / accepted=3000` |

### 目标 mix 与实测 mix 不一致（重要）

`configs/graph_flow.yaml` 里的 `difficulty_mix` / `structure_mix` 是**每次 attempt 重新
采样**的（`generate_controlled_junction_graph` 内部采样，`build_controlled_dataset`
的外层循环只判断 `accepted`）。被留下的样本带着"成功那次 attempt"的标签，于是
**各标签的接受率不同会重新加权 mix**。实测（把生成器包一层计数器、用 seed=0 跑完整
3000 条；attempts/accepted 与 `data/controlled/controlled_summary.json` 完全一致）：

| 标签 | 采样占比 | 接受率 | 实测占比 | 目标占比 |
|---|---|---|---|---|
| difficulty=easy | 19.9% | 0.148 | **28.0%** | 20% |
| difficulty=medium | 60.3% | 0.112 | 64.4% | 60% |
| difficulty=hard | 19.8% | **0.041** | **7.7%** | 20% |
| mode=branch_heavy | 40.2% | 0.202 | **77.1%** | 40% |
| mode=long_chain | 39.3% | **0.036** | **13.4%** | 40% |
| mode=loop_detour | 20.5% | 0.049 | **9.5%** | 20% |

采样本身没问题（20/60/20 与 40/40/20 都打得很准），偏差全部来自接受率：

- **hard 最难通过**（0.041）：hard 要求 `gt_decisions ∈ (6,9)`，但大量 attempt 实际只
  走出 3~4 个 decision（`gt_decisions=3/4 outside (5, 12)` 是第一大拒绝原因），
  hop 预算与 K 的耦合又限制了 K 的上沿。
- **long_chain 最难通过**（0.036）：它的 `candidates_per_decision=(2,3)` 让每个 junction
  的出口更少，更容易跌破 `gt_decisions >= 5` 的下限。
- **branch_heavy 最容易通过**（0.202），所以它从 40% 涨到 77%。

**对评测的直接影响**：test split 只有 300 条，其中 hard 只有 21 条、`gt_decisions=9-11`
只有 11 条（见第 14 节的逐 bucket 对比）。这两个 bucket 上的 delta 噪声约 ±0.1，
下结论时必须看绝对条数而不是只看比例。

**如果要"真正的 20/60/20"**：需要把难度/模式的采样提到重试循环之外（一个输出样本固定
一个标签，只重试图结构），并给每个标签单独的 attempt 上限与兜底。这会改变数据集，
所以要重新生成 + 重新训练，目前**没有做**。

## 12. 实现生成器时踩到的坑（新增）

1. **loop 分支不能连回 start / 更早的 junction**：会造出 `start → helper → J2`
   这种绕过整段骨架的捷径，把 GT 从 22 跳压到 17 跳、路径上只剩 2~3 个 junction。
   现在 loop 只能连"至少隔两跳"的下游 junction。
2. **被迫段用完的 chain 末端必须就是 J1**：否则 skeleton 节点和被迫段是两套节点，
   图会不连通（`gt_hops` 直接报 `NetworkXNoPath`）。
3. **K 与 hop 目标必须耦合**：每段至少 1 个 ordinary 节点 ⇒ `hop >= 2(K+1)`。
   两者独立随机采样会造出大量自相矛盾的样本（接受率只有 3~6%），
   现在改为由 K 推出可行 hop 区间、每次重试重新抽 `forced_source`。
4. **每个 junction 必须至少有一条干扰分支**，否则它的度数停在 2、压根不是 decision；
   但不能强制成 dead-end（会让 dead-end 占到 78%，而指南要 30~40%）。
   兜底用的是"两跳纯 dead-end"，它的终点是叶子，不可能造出捷径。

## 13. 正式训练（100 epoch）

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_controlled_100ep --data data/controlled/controlled_train.pkl --val-data data/controlled/controlled_val.pkl
```

第一版正式配置：`num_samples=3000`、`batch_size=48`、`T=50`、`d_model=128`、`amp=true`、
`epochs=100`。实测约 **43 s/epoch**（50 步 × ~0.86 s，RTX 4070），100 epoch ≈ **72 分钟**。

评测：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/graph_flow.yaml --checkpoint outputs/runs/v2_controlled_100ep/best.pt --data data/controlled/controlled_test.pkl --baselines --out outputs/runs/v2_controlled_100ep/eval_test.json
```

### 第一版单轮结果（flow_steps = 1，作为对照基线）

`best.pt` 取在第 80 epoch。test 300 条查询：goal_hit 0.893、optimal 0.733、
cost_ratio 1.016、loop 0.003、broken 0.103，单条 22 ms。按 decision 条数拆分：
3–5 = 1.000、6–8 = 0.893、9–11 = 0.545 —— 误差沿决策链复利放大，每步约 97%。

## 14. 单个 reverse step 内部的多轮图信息交流（flow_steps）

**动机**：第一版每个 reverse step 只做**一轮**信息交流，即
`H_{t-1} = F_theta(H_t, E_t, tau_t)` 只算一次。这意味着一条长度为 L_decision 的决策链
要跨越 T = 50 个 timestep 才能把信息传完，而且每轮只能看一跳。把"一轮"改成"一个
step 内部连续交流 k 轮"可以让远距离信息在同一个 timestep 内多次传播。

配置（`configs/graph_flow.yaml`）：

```yaml
model:
  flow_steps: 3              # 每个 reverse step 内部的交流轮数（1 = 旧行为）
  flow_slot_embedding: true  # 每轮一个"第几轮"的 embedding
  flow_slot_scale: 1.0
```

实现（`models/graph_flow.py::GraphFlowBlock.forward_multi`）：

```text
H_0 = H_t
for k in 0 .. flow_steps-1:
    H_{k+1} = F_theta(H_k, E_t, tau_t + SlotEmbedding(k))
H_{t-1} = H_{flow_steps}
```

四个必须说明的点：

1. **不是"重复作用同一个映射"**。如果每轮条件完全相同，`F` 反复作用于同一输入只会
   收敛到不动点，多轮几乎等于白算。所以每轮额外加一个**只跟轮次有关**的
   `SlotEmbedding(k)`（与 timestep 无关），加在 `tau_t` 上一起进 AdaLN 条件器。
2. **仍然只有一个 Cell 实例**。`F_theta` 在所有 timestep、所有轮次上共享参数；新增
   的只有 `nn.Embedding(flow_steps, d_model)`。`d_model=128, flow_steps=3` 时
   是 3 × 128 = 384 个参数（相对 348K 约 0.1%）。
3. **状态在轮之间继续累积**（`H_k` 而不是每轮从 `H_t` 重启），Start/Goal 每轮都被
   clamp 回输入，attention 仍然是"按 dst 分组"的 softmax。
4. **`flow_steps=1` 与旧实现逐位一致**（`forward_multi` 退回单次 `forward`），
   所以旧 checkpoint 与 `outputs/runs/v2_controlled_100ep` 的一切结论都仍然可复现。

代价：每个 reverse step 的计算量约为原来的 `flow_steps` 倍。`flow_steps=3` 时实测
**67 s/epoch**（batch 48、T=50、50 步；GPU 利用率不高，所以实际只比单轮的 43 s 慢 1.55 倍），
100 epoch ≈ **1.9 小时**；训练用的是独立 run 名 `v2_controlled_100ep_flow3`，不会覆盖单轮基线。

诊断：`DenoiserOutput.attn_per_slot`（长度 = `flow_steps`）与 `DenoiserOutput.flow_steps`
可用于确认多轮真的发生了（`tests/test_graph_flow.py` 里逐条断言了"多轮 ≠ 不动点迭代"、
"每轮 Start/Goal 都被 clamp"、"梯度能回到每一轮的 slot embedding"）。

还有一个专门的诊断工具，直接量"多轮有没有做事"：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe tools/round_diagnostic.py outputs/runs/v2_controlled_100ep_flow3 --samples 8
```

它打印一个 reverse step 内部每轮对 H 的相对改动 `||H_k - H_{k-1}|| / ||H_{k-1}||`
（只在自由节点上算），以及"少跑几轮"时最终状态离得有多远。实测（flow_steps=3、
epoch 6 的 checkpoint）：

```text
round 0: ||dH_free||/||H_free|| = 0.4576
round 1: ||dH_free||/||H_free|| = 0.3963
round 2: ||dH_free||/||H_free|| = 0.3112
1 round instead of 3: ||H_k - H_K||/||H_K|| = 0.6817
2 rounds instead of 3: ||H_k - H_K||/||H_K|| = 0.3111
```

三点解读：每轮改动都在 30%~46% 量级，**不是**在求不动点（否则会迅速塌到 0），也说
明 slot embedding 这个通道没有被初始化尺度压死（虽然 `std=0.019`、`mean_norm=0.217`
比 `tau_t` 的范数小一个量级，梯度会把它放大到有用的程度）；而只跑 1 轮时最终 H 与
3 轮版本相差 68%，所以"提前退出"是一个真问题、值得用 `--eval-flow-steps` 量化。

### 配套的可复现性改动

改 `flow_steps` 会改变参数集合（每轮一个 `flow_slot_embedding`），所以"这个
checkpoint 到底是几轮训出来的"必须能从 run 目录里读出来，否则以后所有对比都只能靠回忆：

- `scripts/train.py` 现在把**解析后**的配置（含 `--set` 覆盖）写进
  `outputs/runs/<run>/run_config.json`，并在启动日志里打印
  `flow steps  : 3 round(s) per reverse step (shared-cell+slot)`。
- `tools/summarize_run.py` 优先用该 run 自己的 `run_config.json`（而不是命令行传的
  `--config`）来解释它，并在 `summary.json` 里写入 `model.flow_steps / parameters / T /
  batch_size`。基线 run 的这份快照是从 commit `825f4dd` 的 config 事后重建的，
  文件里带 `_reconstructed_from` 说明。
- `load_checkpoint` 在结构不匹配时不再抛裸的 state_dict 报错，而是直接告诉你
  "checkpoint 的 model_config 是 X、当前模型是 Y，flow_steps 必须一致，请用那个 run
  自己的 `run_config.json`"（见 `tests/test_checkpoint_mismatch.py`）。
  评测基线 checkpoint 时要显式换配置：

  ```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_controlled_100ep/run_config.json --checkpoint outputs/runs/v2_controlled_100ep/best.pt --data data/controlled/controlled_test.pkl --baselines --out outputs/runs/v2_controlled_100ep/eval_test.json
  ```

- **曲线对比必须按真实 epoch 对齐**。基线 `v2_controlled_100ep` 的 `history.json`
  只剩 epoch 21-100 且没有 `epoch` 字段，直接按下标对齐会拿"flow3 的 epoch 5"去比
  "基线的 epoch 25"。读取 / 校准 / 对齐统一在 `src/evaluation/history.py`
  （`reconstruct_epoch_offset` 用 `val_records_epoch{N}.json` 的文件名 + goal_hit 值
  核对，核不过就返回 `None` 并警告），`tools/summarize_run.py` 与
  `tools/compare_curves.py` 共用它。**因此基线只有 epoch ≥ 21 的训练曲线和
  epoch ≥ 25 的验证点可用于对比；前 20 个 epoch 已永久丢失。**

### 推理时的提前退出（"训练 3 轮、推理只跑 1 轮"）

多轮交流的代价在推理时是线性的（实测 flow_steps=3 约 0.06 s/query，1 轮约 0.03 s/query）。
"训练时见识过多轮、推理时少跑几轮"完全合法（第 k 轮用的 slot embedding 就是训练时
那一行），所以留了一个口子做"多少轮才够"的 ablation：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_controlled_100ep_flow3/run_config.json --checkpoint outputs/runs/v2_controlled_100ep_flow3/best.pt --data data/controlled/controlled_test.pkl --eval-flow-steps 1 --out outputs/runs/v2_controlled_100ep_flow3/eval_test_flow1.json
```

约束（`GraphFlowDenoiser.set_inference_flow_steps`）：只能**减少**轮数，超过训练时的
round 数会直接 `ValueError`（slot embedding 没有那么多行）；评测结果 json 里会写上
`inference.flow_steps` 与 `inference.trained_flow_steps`，避免事后分不清这是几次前向的结果。

## 15. 结果可视化（预测路径 vs GT 路径）

三个工具，分工不同：

| 想干什么 | 用哪个 |
|---|---|
| 看图（预测路径 vs GT 画在一起） | `tools/visualize_paths.py` |
| 要路径本身（节点序列、结局、分岔点、机器可读 JSON） | `tools/predict_path.py` |
| 公开图数据（DIMACS9/10，`data/public_graphs/`）长什么样 | `tools/visualize_public_graphs.py` |

```bash
# ---- 1) 只要路径本身（文本，不画图）------------------------------------------
# 单条 query：--index 就是 dataset[i] 的编号，也是画图标题里的 #170
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --index 170

# 多条（逗号分隔或重复 --index）+ 导出机器可读 json
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --index 0,47,170 --out-json paths.json

# 换采样种子 / 要确定性输出（推理是随机的，见下文）
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --index 170 --seed 3
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --index 170 --deterministic

# 同一个 checkpoint 只跑 1 轮图信息交流（推理轮数 ablation）
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --index 170 --flow-steps 1

# 换另一个 run（单轮基线）
E:/CondaEnvData/envs/GGMPC/python.exe tools/predict_path.py --run outputs/runs/v2_controlled_100ep --data data/controlled/controlled_test.pkl --index 47

# ---- 2) 画图（预测路径 vs GT 路径）------------------------------------------
# 自动抽样 8 条：覆盖 easy/medium/hard × 到达/断掉
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --num 8 --cols 2 --labels --out outputs/figures/paths_flow3_test.png

# 只看指定几条（便于复现同一组图，或与别的 run 画同一批 query 做对照）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --select indices --indices 170,47,79,140 --cols 2 --labels --out outputs/figures/paths_pick4.png

# 只看 hard 难度
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --num 6 --only-difficulty hard --cols 2 --labels --out outputs/figures/paths_hard.png

# 换布局（默认 spring，见下表）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --num 4 --layout kamada_kawai --out outputs/figures/paths_kk.png
```

`predict_path.py` 的实测输出（`best.pt`，epoch 80）：

```text
#170  medium/branch_heavy  58 节点 / 5 决策
  结果     : 断掉（node 27 has no decision variable）
  跳数     : 预测 5 / GT 25
  预测路径 : [0, 1, 2, 3, 4, 27]
  与 GT 分岔: 第 5 跳（节点 4）
  GT 路径  : [0, 1, 2, 3, 4, 28, 29, 30, 5, 10, 11, 12, 13, 6, 21, 22, 23, 7, 14, 15, 16, 8, 24, 25, 26, 9]
```

它给出四件事：**结局**（到达 goal / 断掉 / 走进环 + 具体原因）、**预测与 GT 的跳数**、
**预测路径的完整节点序列**、**与 GT 分岔的位置**（第几跳、哪个节点）。`--max-nodes-print N`
（默认 64）控制长路径打印时截断显示。

**推理是随机采样的**：整条 reverse chain 有 T=50 个扩散步，每一步都从模型预测的后验分布里
采样"走哪条 branch segment"，所以同一条 query 换个 `--seed` 可能走出不同路径（这也是评测要
多种子平均的原因，见第 16 节）。想固定结果就加 `--deterministic`（用 posterior argmax）。

一张图 2 列 4 行，每个面板一条 query：

* **橙色粗实线 = 模型预测路径**（断掉/成环时只画到断掉那一段）
* **蓝色细虚线 = 标注的 GT 最短路径**
* 绿色星 = start，红色星 = goal，白色小方块 = decision node（模型真正做选择的位置），
  红 X = **预测与 GT 最后一次相同的位置**（也就是分岔点）
* 标题给出：难度/模式、GT 跳数/决策数、预测跳数、结局（到达 goal / 断掉 / 走进环），
  以及"与标注 GT 完全一致 / 同代价但走了另一条等长路 / 比 GT 多几跳"

布局（`--layout`，默认 `spring`）：**按图本身的结构画**，坐标等比例（`set_aspect("equal")`），
不做任何拉伸。同一个 `seed` 下同一张图每次画出来一样。

| 取值 | 说明 |
|---|---|
| `spring`（默认） | Fruchterman-Reingold 力导向，正常图结构、各向同性 |
| `forceatlas2` | networkx 的 ForceAtlas2；链状分支多时会被拉成长条 |
| `kamada_kawai` | 距离保持布局，形状最"正"，节点多时稍慢 |
| `spine` | 把 GT 路径摊成一条水平直线、干扰分支垂直伸出。**只适合盯单条路径的走向，会扭曲真实结构**，不是默认值 |

> 早先的版本默认用 `spine`，画出来"整张图变成一条线"，看轨迹方便但看不出图的结构，
> 已经改成默认 `spring`；`spine` 作为可选项保留。

选择的样本（`--select`）：

| 取值 | 含义 |
|---|---|
| `auto`（默认） | 先跑完整个 pool，再按 (难度, 结局) 分桶，按约 6:4 混着抽成功/失败案例 |
| `random` | **在整个 pool 里均匀随机抽** `--num` 条。默认用系统熵，所以每次跑都不一样；把打印出来的 `select_seed` 填回 `--select-seed` 就能复现同一组图 |
| `indices` | `--indices 277,47,249` 指定样本下标，便于复现同一组图（也可用来和别的 run 画同一批图对比） |
| `goal` / `broken` / `loop` / `optimal` | 只看某一类结局 |
| `--only-difficulty hard` / `--only-mode loop_detour` | 先在某个子集里抽（对 `random` 同样生效，即"在某个子集里随机抽"） |

**两种随机性要分清楚**（命令输出和图的标题里都会打印出来，方便追溯）：

| 参数 | 控制什么 | 默认 |
|---|---|---|
| `--seed` | **推理采样**的随机数种子（`make_generator`）。改它 = 换一副骰子重新解码整个 pool，pool 指标和结局分布都会变 | 取 `run_config.json` 里的 `seed`（本项目是 0） |
| `--select-seed` | 只控制**从 pool 里挑哪几条**（`--select random` 用） | 随机（系统熵），除非显式指定 |

所以同一个命令**每次跑出来的图完全一样**（推理种子固定 + 选择规则确定 + 布局确定性），
这是刻意为之：图可以被引用、可原样复现。想"每次看点不一样的"，用
`--select random`（选样本随机）或 `--seed 1/2/3...`（连解码都换一副骰子）：

```bash
# 每次随机抽 8 条（打印 select_seed，可复现）
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 --data data/controlled/controlled_test.pkl --num 8 --cols 2 --select random --out outputs/figures/paths_random.png

# 固定抽样种子 -> 每次都得到同一组随机样本
... --select random --select-seed 42 --out outputs/figures/paths_random42.png

# 换推理种子 -> 连 pool 结果都变
... --num 8 --seed 1 --out outputs/figures/paths_seed1.png
```

注意两点：`auto` 是**故意偏置**的（约 40% 面板是失败案例），不能拿它估计准确率 ——
准确率看 pool 那一行（`pool result: goal=...`）或正式评测；另外**图里可能有多条等长的
最短路**（实测 sample 170 有 2 条），模型走了另一条时标题会写"同代价最优（与标注 GT
是另一条等长路）"，这不是错。

### 在自己的代码里生成路径

路径的生成分两步，和评测用的是同一条链路：

```
z_T ~ 先验  --sample_reverse_chain-->  z_0（每个 decision 选哪条 branch segment）
z_0        --decode_flat（Path Decoder）-->  节点序列 path + 结局 status
```

```python
from src.data.collate import collate_samples
from src.data.dataset import GraphQueryDataset
from src.diffusion.sampler import sample_reverse_chain
from src.evaluation.path_decoder import candidate_offsets, decision_offsets, decode_flat
from src.training.checkpoint import load_checkpoint
from src.training.setup import build_diffusion, build_model
from src.utils.config import load_config
from src.utils.seed import make_generator, set_seed

run = "outputs/runs/v2_controlled_100ep_flow3"
config = load_config(f"{run}/run_config.json")          # run 自己的配置，保证结构匹配
dataset = GraphQueryDataset.load("data/controlled/controlled_test.pkl")

set_seed(0)
model = build_model(config, "cuda")
load_checkpoint(f"{run}/best.pt", model=model, map_location="cuda")
model.eval()

samples = [dataset[170]]
batch = collate_samples(samples, device="cuda")
z0 = sample_reverse_chain(                              # 整条 reverse chain
    build_diffusion(config), model, batch,
    generator=make_generator(0, device="cpu"), stochastic=True,
)["z0"]
result = decode_flat(                                   # z_0 -> 路径
    samples[0], z0,
    decision_offset=decision_offsets(samples)[0],
    candidate_offset=candidate_offsets(samples)[0],
)
print(result.status, result.path, result.reason)        # 结局 / 节点序列 / 原因
```

三个注意点：

1. **批量解码必须同时传两个 offset**（`decision_offset` / `candidate_offset`），因为
   `z_0` 与候选表在 batch 里是拼接起来的（`collate_samples` 的 paged 布局）。
2. **`z_T ~ 先验` 只在 `t = T` 成立**，所以 sampler 只接受完整链（`P2-3`），不能"从中间起跑"。
3. **推理是随机采样**：想复现就固定 `set_seed` + `make_generator(seed)`，或用
   `stochastic=False`（posterior argmax）。

## 16. 最终对照结论：多轮图信息交流到底有没有用

两个 run 都是 100 epoch、同一份数据、同一套超参（`T=50`、`batch=48`、`lr=1e-4`），
唯一差别是 `model.flow_steps`。各取**验证集最优** checkpoint（两个 run 恰好都是 epoch 80）：

| | `flow_steps=1`（基线） | `flow_steps=3` |
|---|---|---|
| 参数量 | 348,162 | 348,546（+384，+0.11%） |
| best val（epoch 80）goal_hit | 0.9067 | 0.8900 |
| **test goal_hit** | 0.8933 | **0.9167** |
| test optimal | 0.7333 | 0.7233 |
| test broken | 0.1033 | 0.0833 |
| 推理耗时（GPU 空闲实测） | **0.025 s/query** | 0.053 s/query（**≈2.1×**） |
| train loss（80 个共同 epoch 的均值） | 0.3493 | **0.3004**（一致更低） |

**统计检验（这才是关键）**

1. 单次评测（300 条 test、seed 0）配对检验：goal_hit `+0.023`（p=0.32）、
   optimal `−0.010`（p=0.78）、broken `−0.020`（p=0.41）—— **都不显著**。
2. 5 个随机种子的配对检验（每个 query 取 5 次采样均值，再做配对 t 检验）：
   flow1 = 0.8953、flow3 = 0.8873，**Δ = −0.0080**，p = 0.44，
   bootstrap 95% CI `[−0.0287, +0.0120]`。单个种子的 delta 在 `[−0.0333, +0.0233]`
   之间跳（std 0.0216）——**单跑一次得到的 ±2~3 个点全是采样噪声**。
3. 验证集逐 epoch 配对比较是**混合**的：epoch 40 强烈支持 flow3（p=1e-4）、
   epoch 45/50/100 也支持；但 epoch 75（p=0.006）、85（p=0.013）反过来支持基线。

**结论：没有证据表明"每个 reverse step 内部多轮交流"提升了测试集泛化。**
它把训练 loss 压得更低（−0.049，80 个共同 epoch 上无一例外），但这份额外的拟合能力
没有转化成更好的测试表现；代价是推理约 2.1 倍、训练每 epoch 约 1.5 倍。

**但多轮不是摆设**：同一个 `flow_steps=3` 的 checkpoint，推理时只跑 1/2/3 轮分别是
`0.5167 / 0.8133 / 0.9167`（broken `48% / 19% / 8%`）—— 训练时按 3 轮学出来的表示，
少跑一轮就崩。也就是说多轮传播在这套表示里承担了实际职责，只是"训练 3 轮 vs 训练 1 轮"
没有拉开差距。

**可能的原因与下一步（都还没做）**：`T=50` 本身已经提供了 50 轮传播，每步再多 3 轮的
边际收益被稀释；更值得试的方向是**减少扩散步数 + 增加每步轮数**（例如 `T=10` +
`flow_steps=5`，总传播量相近但推理更省），以及给多轮配置配套的正则/学习率调整
（多轮增加了有效容量，训练 loss 更低而测试集没有变好，有轻微过拟合的迹象）。

### "多轮交流缓解长链决策"——**已用专门的长链测试集证实**

之前 300 条标准测试集里决策数 ≥9 的只有 **11 条**（±1 条 = ±9 个点），什么都测不出来。
于是专门造了一个长链评测集：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml --data-dir data/long --name controlled_long --no-split --min-decisions 9 --set data.num_samples=400 --set seed=7 --set data.difficulty_mix.hard=1.0 --set data.difficulty_mix.easy=0.0 --set data.difficulty_mix.medium=0.0
```

生成结果（`data/long/controlled_long.pkl` + `_summary.json`）：**400 条 query / 400 张图**、
决策数 9–10（均值 9.20）、GT hops 21–33（均值 26.0）、全部 hard 档；总共
**102,678 次 attempt** 才凑出这 400 条（约 0.39% 的接受率——这就是长链样本稀少的根源）。
用**图结构指纹**（排序后的边集合哈希）核对过：与 train / val / test 的交集都是 **0**，无泄漏。

用现有的两个 checkpoint 在这个长链集上各跑 5 个随机种子、逐 query 配对检验：

| 指标 | flow1（1 轮） | flow3（3 轮） | Δ | 配对 t 检验 | bootstrap 95% CI |
|---|---|---|---|---|---|
| **goal_hit** | 0.6010 | **0.6685** | **+0.0675** | **p < 0.0001** | **[+0.042, +0.093]** |
| **optimal** | 0.4530 | **0.5335** | **+0.0805** | **p < 0.0001** | **[+0.057, +0.105]** |

**5 个种子全部偏向 flow3**（逐种子 Δ = +0.0425 / +0.0650 / +0.0700 / +0.0950 / +0.0650），
不像标准测试集那样有个别种子反向。所以结论是：

> **多轮图信息交流确实能缓解长决策链上的误差**：在 400 条长链 query 上 goal_hit
> +6.8 个点（相对 +11%）、optimal +8.1 个点，p<1e-4。这个收益在标准混合测试集上
> **测不出来**（Δ=−0.008，p=0.44），因为那种测试集里 84% 的样本只有 6–8 个决策。

必须写在旁边的三条限制：

1. **这是"罕见但合法"的区间，不是完全分布外**。训练集里决策数 ≥9 的样本只占 ~3.7%，
   模型见过这类图但很少；长链集用的是同一套生成器与验收契约（hops 15–35、决策 5–12）。
2. **长链集的难度/模式构成偏**：全部 hard 档（构造使然），模式上 **96% 是 branch_heavy**
   （`long_chain` 只有 3 条、`loop_detour` 13 条——这两行没有统计意义，别引用）。
   这是各模式接受率差异造成的（branch_heavy 0.202 vs long_chain 0.036）。
3. **只测了两个 checkpoint 的推理表现**，没有重训。所以"训练时也用长链偏重的数据会不会
   更好"仍未回答；那是下一步（需要新数据 + 新 run）。

复现：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe tools/multiseed_eval.py --a outputs/runs/v2_controlled_100ep --b outputs/runs/v2_controlled_100ep_flow3 --data data/long/controlled_long.pkl --seeds 0,1,2,3,4 --metric goal_hit --bucket-data data/long/controlled_long.pkl --out outputs/runs/v2_controlled_100ep_flow3/multiseed_long_goal_hit.json
```

### 长链偏重的训练集已备好（**只造数据，尚未训练**）

训练分布的问题已经量化：现有训练集里 ≥9 决策的样本只有 **119/2400 = 5.0%**
（6–8 决策占 82%），所以 82% 的算力花在中短链上。按"让长链占 30% 左右"补了数据：

```bash
# 1) 先另外造一批长链样本（不要动现有 train/val/test）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml --data-dir data/long --name controlled_longpool --no-split --min-decisions 9 --set data.num_samples=1050 --set seed=11
# 实测：1050 条 / 1050 图，决策数 9–10（均值 9.22）、hops 16–33（均值 22.0），
#       201,352 次 attempt 才凑出来（接受率 0.52%），耗时 ~10 分钟

# 2) 与现有训练/验证集合并（graph_id 全局重编号 + 固定 seed 打乱）
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/long/controlled_longmix_train.pkl --input data/controlled/controlled_train.pkl --input data/long/controlled_longpool.pkl --limit 900:1 --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/long/controlled_longmix_val.pkl --input data/controlled/controlled_val.pkl --input data/long/controlled_longpool.pkl --skip 900:1 --seed 0
```

| 文件 | queries / graphs | 决策数直方图 | ≥9 占比 |
|---|---|---|---|
| `controlled_longmix_train.pkl` | 3300 / 3300 | 5:310, 6:1037, 7:669, 8:265, **9:807, 10:212** | **30.9%** |
| `controlled_longmix_val.pkl` | 450 / 450 | 5:38, 6:131, 7:85, 8:28, **9:128, 10:40** | **37.3%** |
| （原）`data/controlled/controlled_train.pkl` | 2400 / 2400 | 5:310, 6:1037, 7:669, 8:265, 9:101, 10:18 | 5.0% |

**无泄漏**（按图结构指纹即排序边集合的哈希核对）：longmix_train ↔ longmix_val = 0、
longmix_train ↔ controlled_test = 0、longmix_train ↔ controlled_long = 0，
longmix_val ↔ 两个 test 也都是 0。原来三个 split 与两个 test 集**一个字节都没动**，
所以前面所有结论继续有效。

**将来要（用户批准后）训练时**：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_longmix_100ep --data data/long/controlled_longmix_train.pkl --val-data data/long/controlled_longmix_val.pkl
```

想验证的两件事（先记下来，免得事后凑解释）：
1. 在 `controlled_long.pkl`（400 条长链测试集）上，新模型应该明显高于现在 flow1 的 0.601
   / flow3 的 0.669 —— 如果没提高，说明瓶颈不是"没数据"而是别的（优化/容量/表示）；
2. **flow1 与 flow3 在长链上的差距可能会缩小**：多轮交流现在的价值有一部分来自
   "长链样本太稀有、学不好"，一旦长链被训够，单轮模型可能自己就能覆盖，多轮的边际收益
   会下降。这是个可证伪的预测，下次直接对比。

注意长链补充样本的构成仍有偏：难度标签是 medium 86% / hard 14%（自然采样），
结构模式 branch_heavy 89% / long_chain 5% / loop_detour 6% —— 这是各模式接受率差异
造成的（见第 11 节），不是刻意设计。

复现全部结论：

```bash
cmd /c tools\postprocess.bat v2_controlled_100ep_flow3 v2_controlled_100ep
E:/CondaEnvData/envs/GGMPC/python.exe tools/multiseed_eval.py --a outputs/runs/v2_controlled_100ep --b outputs/runs/v2_controlled_100ep_flow3 --seeds 0,1,2,3,4 --metric goal_hit
```

产物：`outputs/runs/v2_controlled_100ep_flow3/{eval_test.json, summary.txt,
breakdown_test.json, paired_*.json, curves_vs_baseline.json, multiseed_goal_hit.json,
ablation_flow_steps.json}`。

## 17. 本轮修订：节点/边联合 Attention + Soft Goal Reachability

这一轮只做两件事，其它已经稳定的结构（persistent `H_t`、Start/Goal 只出不进与
clamp、AdaLN、residual、两套 LayerNorm、edge selected/unselected embedding、
一个 reverse step 内多轮交流 + slot embedding、Branch Mean Pool、grouped softmax）
全部没动。**改完只做了测试与冒烟验证，没有真的开始训练。**

### 17.1 Graph Attention：节点 + 边共同决定权重（A）

```text
旧：K_{uv} = W_K e_uv^t                    边状态相同 -> attention 必然相同
新：K_{uv} = W_{K,n} h_u + W_{K,e} e_uv^t  sender 身份（含 Start/Goal）也能进权重
    score  = <q_v, (k_node + k_edge)/sqrt(2)> / sqrt(d)
```

* 代码：`src/models/graph_flow.py`（`k_proj` 拆成 `k_node_proj` + `k_edge_proj`）；
* `1/sqrt(2)` 只是让两路相加后的初始方差与单路一致，不改变表达能力；
* **旧 checkpoint 不能直接加载**（`k_proj` 已被拆开），需要重新训练；
  加载失败时 `load_checkpoint` 会给出可读的结构不匹配报错。

### 17.2 Loss：CE + Soft Goal Reachability（B）

```text
L = x0_ce * L_CE + goal_reach_weight * L_goal
L_goal = sum_t omega_t * L_goal^{(t)} / sum_t omega_t ,  omega_t = alpha_bar_t
L_goal^{(t)} = -(1/B) * sum_b log(P_goal,b^{(t)} + eps)
```

* 每个 reverse timestep **同时**算 CE 与 Soft Goal，只在最后反传一次；
* `P_goal` 用 Branch Scorer 的概率分布做有限 horizon 的 value iteration
  （`src/training/soft_goal.py`）：`branch -> Goal = 1`、`branch -> decision j = V_j`、
  `NULL / dead-end = 0`，只有 softmax 概率 + gather + 乘法 + `segment_sum`，
  梯度能回到 Branch logits；
* 拓扑信息在数据层预先翻译成整数编号（`collate.py` 新增四个字段
  `candidate_next_decision / candidate_hits_goal / reach_start_decision /
  reach_start_is_goal`），Soft Goal 内部**不做任何 NetworkX 遍历**；
* `deg(s) > 1` 时从 source 自己开始算，`deg(s) == 1` 时从 forced segment 的终点开始算；
* **Soft Goal 只是可微训练代理指标**，验证阶段仍然报告
  `Hard Goal Hit / Optimal Path Rate / Loop Rate / Broken Rate`，并额外报告
  `soft_goal_reachability`（评测与每个 epoch 的 `val_*` 都会写进 `history.json`）。

配置（`configs/graph_flow.yaml`）：

```yaml
loss:
  x0_ce: 1.0
  null_weight: 1.0
  active_weight: 1.0
  goal_reach_weight: 0.1        # =0 时严格退化成纯 CE baseline
  goal_reach_eps: 1.0e-8
  goal_timestep_weighting: alpha_bar   # 或 uniform（消融）
  goal_horizon_cap: 24          # value iteration 轮数上限（见第 18 节；controlled 数据无影响）
```

训练日志拆成：

```text
[epoch 1] batch 1/2 loss=1.8581 ce=1.4822 goal=3.7588 soft_goal=0.0265 x0_acc=0.800
```

`history.json` 里对应 `train_loss / train_ce_loss / train_goal_loss /
train_soft_goal / train_x0_acc`，验证部分多一个 `val_one_step_soft_goal` 与
`soft_goal_reachability`。

### 17.3 本轮新增的测试

`tests/test_soft_goal.py`（16 个）覆盖：`0.8 x 0.7 = 0.56`、NULL / dead-end 贡献为 0、
两条路径概率相加、`SoftGoalLoss.backward()` 给 Branch 概率非零梯度、
degree-1 Source 从 forced 终点起算 / 多出口 Source 从自己起算、多图 batch 不串图
（含不同 horizon）、`goal_reach_weight=0` 与纯 CE 逐位一致。
`tests/test_graph_flow.py` 新增 2 个：边状态相同但 sender 不同时 attention 必须能不同、
sender 相同而 edge 不同时也必须能不同。

```bash
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q        # 252 passed
E:/CondaEnvData/envs/GGMPC/python.exe tools/semantic_check.py --strict   # checked 61 modules, 0 problems
```

## 18. 旧数据（V1）融合训练

`data/processed/v1/`（原 `data_old/`，5000 图 / 25000 query）是 V1 时代的预处理数据，
它的候选是"下一跳边"，V2 用的是"branch segment"，两者语义不同，所以必须转换：

```bash
# 1) V1 -> V2（GT 沿用 V1 自己的 gt_path，decision/active 集合逐条核对为 0 不一致）
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/test.pt --out data/oldv1/oldv1_test.pkl
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/train.pt --out data/oldv1/oldv1_train_sub.pkl --sample 2000 --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/val.pt --out data/oldv1/oldv1_val_sub.pkl --sample 400 --seed 0

# 2) 与现有训练/验证集合并（不合并 test；"融入训练"只动 train/val）
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_train.pkl --input data/long/controlled_longmix_train.pkl --input data/oldv1/oldv1_train_sub.pkl --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_val.pkl --input data/long/controlled_longmix_val.pkl --input data/oldv1/oldv1_val_sub.pkl --seed 0

# 3) 图级泄漏检查（必须 0 重叠才能训）
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --pair data/mixed/mixed_oldv1_train.pkl data/mixed/mixed_oldv1_val.pkl --pair data/mixed/mixed_oldv1_train.pkl data/oldv1/oldv1_test.pkl --pair data/mixed/mixed_oldv1_train.pkl data/controlled/controlled_test.pkl

# 4) 训练
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow.yaml --name v2_rev2_mixed --data data/mixed/mixed_oldv1_train.pkl --val-data data/mixed/mixed_oldv1_val.pkl
```

要点：

* **`--sample` 而不是 `--limit`**：V1 的 query 按图类型分块存储（er/ba/ws/geometric
  各占一段），取前缀只会拿到 `er` 图。`tools/convert_v1_dataset.py` 与
  `tools/merge_datasets.py` 都支持 `--sample N[:INDEX]`。
* **约 2.2% 的 V1 query 转换不了**：随机图里存在"由两个 degree=2 节点构成的三角"
  （例如 `5-11-12-5`），V2 的 branch segment 会判定"回到 owner"并断言失败，这些样本
  会被跳过并打印原因。
* **`loss.goal_horizon_cap`（新增，默认 24）**：soft goal 的 value iteration 轮数取
  "batch 内最大 decision 数"（旧 V1 单图最多 79 个路口）。不设上限时**一个这样的样本
  会把整批 50 个 timestep 的轮数全抬到 79**，训练变成 CPU-bound（实测 GPU 利用率
  ~20%，混合 batch 4.43 s → 设 24 后 3.10 s）。对 decision 数 ≤ 上限的图**行为逐位
  不变**（controlled 数据最多 10），所以旧 run 的结果不受影响。
* **分布差异要有预期**：混合集每 query 决策数 38.2（V1）/ 7.3（longmix），active
  decision 占比从 91.5% 掉到 **31.9%** —— NULL 类样本大幅变多，CE 里 `null_weight`
  默认还是 1，模型可能更倾向"在该走的路口选 NULL"。必要时用
  `--set loss.null_weight=0.3` 做对照。

### 18.1 legacy checkpoint 兼容（第二轮修订的连带改动）

第二轮把 `k_proj` 拆成 `k_node_proj` + `k_edge_proj` 后，旧 checkpoint 本来无法加载。
`load_checkpoint` 现在会自动检测并做**无损映射**（`k_node = 0`、
`k_edge = sqrt(2) * k_proj`，数学上新 K 恒等于旧 K），并打印一行提示；这样旧 run 的
`best.pt` 仍可用来评测/对比（实测复现了旧 run 的 history 数字）。

## 19. 多分支（存活路径表）解码

单路径解码每个 decision 只取一个候选（采样或 argmax），一步选错整条 query 就判失败；
而模型经常给出"两条 branch 概率差不多"，只留一条会把另一条本来能到终点的路丢掉。

```text
frontier = [起点]
while frontier 非空:
    每条存活路径在当前 decision 取概率最高的 top-k 条 branch，各自分叉
    终止的（到 goal / 选 NULL / 撞 dead-end / 重复节点）移出表，记入 finished
    其余留在表里继续；按累计 log 概率排序，超过 beam_width 的尾部丢掉
-> finished = 所有活到终止条件的路径（按概率排序）
```

实现：`src/evaluation/multi_path_decoder.py`（`decode_multi_path`）。两种 NULL 策略：

* `stop`（默认，忠实于数据语义）：NULL 参与排名，被选中则该路径终止；
* `skip`：NULL 不参与排名，只在非 NULL 候选里取 top-k —— 即"NULL 不停，继续走"。

### 19.1 命令行

```bash
# 单独评测（三套口径对比：单路径 / 多分支 best / 多分支 coverage）
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_multipath.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --top-k 2 --beam-width 64 --null-policy skip

# 接进主评测脚本（其余工具链不变：输出的 eval json 记录口径完全一致）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_rev2_mixed/run_config.json --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/long/controlled_long.pkl --device cuda --decode multi --top-k 2 --null-policy skip --out outputs/runs/v2_rev2_mixed/eval_long_multi.json

# 多种子配对检验（两个 run 用同一模式评测，比较的才是同一件事）
E:/CondaEnvData/envs/GGMPC/python.exe tools/multiseed_eval.py --a outputs/runs/v2_rev2_longmix/v2_rev2_longmix --b outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --seeds 0,1,2 --metric goal_hit --decode multi --top-k 2 --null-policy skip

# 可视化：细线 = 整张路径表，粗线 = 累计概率最高的那条
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_paths.py --run outputs/runs/v2_rev2_mixed --data data/long/controlled_long.pkl --num 8 --cols 2 --labels --select random --select-seed 0 --multi-k 2 --null-policy skip --multi-highlight 5 --out outputs/figures/paths_mixed_long_multi_k2.png
```

`--decode multi` 时主指标取"累计概率最高的那条路径"（与单路径口径可直接对比），
并额外报告集合语义的 `coverage_rate`（至少一条路径到终点）与
`optimal_coverage_rate`（至少一条最短路）。

**图上怎么区分不同路径**（`--multi-k` 时）：按概率名次上色，而不是全画成橙色 ——

| 画面元素 | 含义 |
|---|---|
| 粗橙线 | 第 1 名：累计 log 概率最高的路径 |
| 蓝 / 绿 / 红 / 紫 细线 | 第 2..5 名备选（`--multi-highlight N` 控制给前几名上色，默认 4） |
| 同色圆圈 + 数字 | 该备选**与主路径分叉的那个节点**（数字 = 名次） |
| 淡灰细线 | 路径表里的其它路径 |
| 蓝虚线 | GT 最短路 |
| 标题末行 | `路径表 N 条（到终点 M）· coverage ✓/✗ · 最优 XX 跳 · 备选 a/b/c/d 跳` |

### 19.2 实测（`v2_rev2_mixed`，best.pt = epoch 90，seed 0）

controlled_long（400 条长链）：

| 口径 | goal_hit | optimal | cost_ratio | loop | broken |
|---|---|---|---|---|---|
| 单路径（现状：采样 z₀） | 0.8950 | 0.7900 | 1.0061 | 0.0075 | 0.0975 |
| 多分支 best（k=1，确定性贪心） | 0.9700 | 0.9200 | 1.0021 | 0.000 | 0.0300 |
| 多分支 best（k=2） | 0.9675 | 0.9175 | 1.0022 | 0.000 | 0.0325 |
| 多分支 best（k=2，NULL 不停） | **1.0000** | 0.9450 | 1.0024 | 0.000 | 0.0000 |
| coverage（k=2） | **1.0000** | optimal coverage 0.9975 | | | |
| coverage（k=3） | 1.0000 | **optimal coverage 1.0000** | | | |

controlled_test（300）：单路径 0.9300/0.8833 → 多分支 best 0.9233/**0.9000**，
coverage **1.0000**（optimal coverage 0.9967）。
oldv1_test（2444）：单路径 0.9763/0.9677 → 多分支 best **0.9890/0.9824**，
coverage 0.9988（optimal coverage 0.9939）。

三点结论：

1. **"分布里有路"被证实**：controlled 两个测试集上 coverage = 1.0000，即每个 query
   的路径表里都至少有一条能到终点 —— 之前的失败不是"模型不知道路"，而是单路径解码
   把它丢了（对应"NULL 提前停 ≠ 走不到"）；
2. **收益的大头来自"确定性解码"**：k=1（按累计概率贪心、不采样）就把 long 从
   0.895 抬到 0.970、optimal 从 0.790 抬到 0.920，多分叉的边际价值主要在 coverage；
3. `--null-policy skip` 在 long 上把 best 路径的 goal_hit 抬到 1.0000、broken 归零，
   但它**在训练目标之外**（属于推理侧策略），报告里要单独标注，不能和训练得到的
   指标混为一谈。

### 19.3 新增测试

`tests/test_multi_path_decoder.py`（10 个）：k=1 等价贪心、k=2 救回 k=1 丢掉的路、
NULL stop/skip 的差异、loop 路径被剪掉但不影响其它路径、beam 截断只丢低概率分支、
finished 按概率排序、可复现、参数校验、`to_decode_result` 的口径转换。

---

## 20. Weighted 扩展（带权图：BFS -> Dijkstra，可选、零破坏）

目标是把任务从

```
(G, s, g) -> minimum-hop path        # 旧的无权 V2
(G, w, s, g) -> minimum-cost path    # 新的 Weighted 扩展
```

并且**只做加法**：`weighted=false` 时模型结构、参数集合（state_dict 的 key 逐个相同）、
数值路径、旧 checkpoint、旧数据集、旧训练曲线全部照旧可用。

### 20.1 两个开关

| 开关 | 作用 | 缺省 |
|---|---|---|
| `data.weighted` | 给边赋正 cost，并把 GT 从 BFS 换成 Dijkstra | `false` |
| `model.use_edge_cost` | 实例化 `EdgeCostEncoder` / `k_cost_proj` / `v_cost_proj`，让 cost 真的进网络 | `false`（老配置里没这个键，读出来就是 False） |

`model.use_edge_cost=false` 时**不会创建任何** cost 参数，所以老 checkpoint 的 key 集合
与现在的无权模型逐 key 相同（`tests/test_weighted.py::test_legacy_checkpoint_still_loads_with_the_unweighted_config`
就是拿 `outputs/runs/v2_rev2_mixed/best.pt` 直接验的）。

配置：`configs/graph_flow.yaml`（无权，行为一个字节没改）与 **`configs/graph_flow_weighted.yaml`**（新增，加权）。

### 20.2 模型：cost 作为第三路 Key / 第二路 Value

```
无权：K_uv = (W_{K,n} h_u + W_{K,e} e^state_uv) / sqrt(2)      V_uv = W_V h_u
加权：K_uv = (W_{K,n} h_u + W_{K,e} e^state_uv + W_{K,c} e^cost_uv) / sqrt(3)
      V_uv = (W_V h_u + W_{V,c} e^cost_uv) / sqrt(2)
```

`e^cost_uv = MLP(w_hat_uv)`，`w_hat = w / mean_G(w)`（**per-graph mean 归一化**）。
之所以只允许 graph_mean：目标函数是 edge cost 的累加，纯比例缩放不改变 `argmin_P sum w_e`；
min-max / z-score 带平移，会把"不同跳数路径"之间的排序改掉，所以配了就直接报错。

### 20.3 数据：先拓扑验收，再赋权，最后重算 GT

```
生成 topology -> 难度过滤（hops / decisions / branch factor，拓扑层面）
             -> attach_edge_weights(w ~ U(1,10)) -> graph.graph["weighted"]=True
             -> build_sample() 里用 nx.shortest_path(weight="weight") 重算 GT
             -> decision field / target candidate（不变）
```

Batch 里新增四个 tensor（`src/data/collate.py`）：`physical_edge_cost` / `physical_edge_cost_norm`
（`[E_phys]`）与 `candidate_branch_cost` / `candidate_branch_cost_norm`（`[C]`，NULL=0）。
物理边编号沿用 `graph.edges()` 顺序，message 方向靠现成的 `msg_to_phys_edge` 映射，
所以一条无向边的两个方向必然拿到同一个 cost，**Branch Segment 的编号系统一行没改**。
`candidate_branch_cost` 第一版只用于统计/诊断，没有接进 Branch Scorer（一次只改一个模块）。

### 20.4 数据有效性自检（写进 dataset summary）

实测 3000 条 `data/weighted_controlled`（U(1,10)）：

| 指标 | 实测 | 含义 |
|---|---|---|
| `weighted_conflict_rate` | **0.4597** | 加权最优 != 跳数最优的样本比例（越高越说明"必须看 cost"） |
| `bfs_cost_ratio` | 1.0344 | `C(P*_hop) / C(P*_weighted)` 的均值（方案里的参考值是 1.2+） |
| `bfs_cost_ratio_p90` | 1.1140 | 同上，90 分位 |
| `dijkstra_cost_ratio` | 1.0 | oracle 自证 |
| `gt_path_is_weighted_optimal_fraction` | 1.0 | GT 确实就是 Dijkstra 解 |
| `weight_mean / std / min / max` | 5.49 / 2.60 / 1.00 / 10.00 | 权重分布 |

`bfs_cost_ratio` 只有 1.03：controlled 图的"绕行/环路"候选大多比主干更长，
i.i.d. U(1,10) 下跳数最优路径的 cost 通常只贵几个百分点。**冲突率 46% 说明监督信号是够的**
（近一半样本的决定目标与"只看拓扑"不同），但 `success_cost_ratio` 这个指标本身区分度有限。
想要更陡的 cost 信号就把分布换成 loguniform（实测 conflict 0.71 / cost ratio 1.19）：

```bash
# 生成时换分布（同一套代码，只是把 U(1,10) 换成 exp(U(log 0.1, log 10))）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow_weighted.yaml --data-dir data/weighted_controlled_logu --set data.edge_weight.distribution=loguniform --set data.edge_weight.range=[0.1,10.0]
```

生成脚本会在 `bfs_cost_ratio < 1.05` 或 `weighted_conflict_rate < 0.20` 时打印
`[weighted WARNING] ...`（U(1,10) 会触发前者，这是**如实报警**，不是 bug）。

### 20.5 命令

```bash
# 1) 生成加权数据集（3000 条，U(1,10)；约 50 秒）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow_weighted.yaml --name weighted_controlled --data-dir data/weighted_controlled

# 2) 主实验：weighted GT + Edge Cost Encoder
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow_weighted.yaml --name v2_weighted_controlled --data data/weighted_controlled/weighted_controlled_train.pkl --val-data data/weighted_controlled/weighted_controlled_val.pkl

# 3) 关键消融：同一份 weighted GT，但**不给模型 edge cost**（应当明显变差）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/train.py --config configs/graph_flow_weighted.yaml --name v2_weighted_controlled_cost_ablated --data data/weighted_controlled/weighted_controlled_train.pkl --val-data data/weighted_controlled/weighted_controlled_val.pkl --set model.use_edge_cost=false

# 4) 评测（Dijkstra baseline 的 cost_ratio 必须是 1.0）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config outputs/runs/v2_weighted_controlled/run_config.json --checkpoint outputs/runs/v2_weighted_controlled/best.pt --data data/weighted_controlled/weighted_controlled_test.pkl --device cuda --no-progress --baselines --out outputs/runs/v2_weighted_controlled/eval_test.json
```

评测侧的两处修正：`baselines.shortest_path()` 在 weighted 图上用 `weight="weight"`（否则它根本不是
cost ratio 的 oracle），`metrics.evaluate_sample()` 的 optimal 判定从绝对阈值改成
`math.isclose(rel_tol=1e-6, abs_tol=1e-6)`。

### 20.6 零破坏验证（本轮实测）

```bash
# 旧 checkpoint + 旧数据集 + 旧 config：路径级指标与改动前逐位相同
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --config configs/graph_flow.yaml --checkpoint outputs/runs/v2_rev2_mixed/best.pt --data data/controlled/controlled_test.pkl --baselines --no-progress --out outputs/runs/v2_rev2_mixed/eval_after_weighted_extension.json
```

结论见 `outputs/runs/v2_rev2_mixed/regression_after_weighted_extension.json`：

* `goal_hit 0.9300 / optimal 0.8833 / cost_ratio 1.0030 / broken 0.0700 / soft_goal 0.9110` ——
  与改动前的 `eval_test.json` 逐位一致（`records` 去掉 `elapsed` 后完全相同）；
* `soft_goal_reachability` 相差 4.5e-8，**小于同一份代码连跑两次的噪声（5.4e-8）**，是 GPU 浮点累加噪声；
* 旧数据集 `controlled / long / oldv1 / mixed` 全部 `load -> collate -> loss.backward()` 正常，
  无权 batch 的 `physical_edge_cost` 恒为 1.0；
* `data/smoke/*.pkl` 是 2026-09-13 生成的旧格式（缺 `source_forced_edge_ids`），与本次改动无关。

### 20.7 已知限制（第一版故意没做的东西）

* `--decode multi`（存活路径表）里的 `optimal_coverage_rate` 仍然按**跳数**算
  （`multi_path_decoder.PathResult.cost` 是 hop count），所以在 weighted 数据集上它衡量的是
  "至少有一条跳数最短路"，不是"至少有一条最小 cost 路"。第一轮故意不改它，
  weighted 的主指标看 `--decode single` 的 `optimal_path_rate` / `success_cost_ratio`。
* `candidate_branch_cost` 只进了 Batch 与统计，没有进 Branch Scorer（方案第 8 节：
  先只做 EdgeCost -> GraphFlow 一条链路，提升归因才干净）。
* 没有 cost-aware loss（expected cost / path length loss），loss 仍然是 `CE + 0.1 * SoftGoal`。
* Edge cost 只按 `msg_to_phys_edge` 映射进 message，Branch Segment 的编号系统一行没改。

### 20.8 实测结果（3000 条 U(1,10) 数据，T=50，flow_steps=3，100 epoch，CE + 0.1 SoftGoal）

两个 run 只差一个开关（`model.use_edge_cost`），数据、seed、超参完全相同：

| 口径 | `goal_hit` | `optimal` | `success_cost_ratio` | `broken` | 说明 |
|---|---|---|---|---|---|
| **Ours weighted**（best.pt = ep95） | **0.8633** | **0.6900** | **1.0070** | 0.1367 | weighted GT + Edge Cost Encoder（single = 最终 argmax readout，见第 23 节） |
| Ours，cost 消融（best.pt = ep75） | 0.8767 | 0.3667 | 1.0557 | 0.1233 | 同一份 weighted GT，但模型看不到 edge cost |
| Greedy-BFS（按跳数贪心） | 1.0000 | — | 1.0350 | — | 永远能到终点，但路线不最优 |
| **Dijkstra oracle** | **1.0000** | **1.0000** | **1.0000** | — | `baselines.shortest_path()`（oracle 口径自证） |

逐条配对（同一批 300 条 query，McNemar 精确检验，`outputs/weighted_paired_*.json`）：

* `optimal`：**+0.393**（0.287 → 0.680），只有消融达标 20 条、只有 weighted 达标 138 条，**p = 7.0e-23**；
* `goal_hit`：+0.080（0.760 → 0.840），p = 0.0138（可达性受 Soft Goal 保护，差距远小于最优性）；
* 分桶（`difficulty=easy/medium/hard`、三种 `mode`、`source=forced/decision`、`gt_decisions=3-5/6-8`）
  全部同向且 p ≤ 0.002，只有 `gt_decisions=9-11` 那一格 n=16、两边打平。

换 `last.pt`（不做 best-checkpoint 选择）结论不变：weighted `optimal 0.6833 / cost_ratio 1.0042`，
消融 `optimal 0.2867 / cost_ratio 1.0615`。

**结论**：Edge Cost Encoder 不是装饰 —— 拿掉它以后，同一个模型在同一个数据集上"恰好走最小 cost 路"
的比例从 68% 掉到 29%，cost ratio 从 1.007 退化到 1.067（比 Greedy-BFS 的 1.035 还差）。
这直接证明模型真的在用 edge weight，而不是靠拓扑猜答案。

复现命令见 20.5；一键复跑收尾评测：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_weighted_experiment.py            # 立刻评测
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_weighted_experiment.py --wait     # 等训练跑完再评测
```

### 20.9 新增测试（`tests/test_weighted.py`，18 个）

weighted GT = Dijkstra（`S-1-A-1-G` vs `S-10-G` 必须是 `S-A-G`）、贵桥被绕开、
两个 message 方向同 cost、`[2,4,6] -> [0.5,1.0,1.5]` 与 argmin 不变、branch cost = 边权和、
无权 batch 的 cost 恒为 1、改一条边的 cost 必须改变 weighted 模型输出而无权模型逐位不变、
`edge_cost_encoder / k_cost_proj / v_cost_proj` 有非零梯度、拿不到 cost 时显式报错、
无权 GraphFlow 拒绝 cost 输入、Dijkstra baseline 是完美 oracle、weighted summary 字段齐备、
`weighted=false` 模型没有任何 cost 参数、老配置缺 `use_edge_cost` 时为 False、
weighted 配置真的启用、非 graph_mean 归一化被拒、旧 checkpoint 逐 key 加载。

---

## 21. 多分支解码增强（必死 branch 预筛选 + 多条 Goal 路径 + 真实 path cost）

来源：《Graph-Junction-Diffusion：Multi-Path Decoder 增强修改指南》。本轮只改**推理解码与评测**，
没有重新训练、没有动 GraphFlow / Edge Cost Encoder / Branch Scorer / Soft Goal / diffusion / 单路径解码。

### 21.1 新增的三件事

| 能力 | 开关 / 接口 | 默认 | 说明 |
|---|---|---|---|
| 必死 branch 预筛选 | `decode_multi_path(..., filter_dead_branches=True)` / `--filter-dead-branches` | **关** | top-k 之前剔除「终点既不是 Goal 也不是 decision node」的非 NULL branch（走完必然 broken，却白占名额）。NULL 不参与筛选，loop 也不在预筛选里处理 |
| 真实 path cost | `PathCandidate.path_cost` | — | `sum_{(u,v) in P} w_uv`；无权图没有 `weight` 属性 → 自动退化成跳数。**只用于记录/排序/评测，绝不参与 beam 剪枝** |
| 多条 Goal 路径 | `best_goal_cost_path` / `goal_paths_by_prob` / `goal_paths_by_cost` / `to_dict()` | — | Goal 里真实 cost 最低（同 cost 取概率更高者）；另有两种排序的 top-N 导出 |

历史语义一个都没改：`multi.best` 仍然是「finished 中累计 log probability 最高的路径」（broken 也能是 best），
`multi.best_goal` 仍然是「Goal 路径里概率最高」，`PathCandidate.cost`（跳数）保留，beam 仍然只看累计 log 概率。

### 21.2 输出口径

`scripts/evaluate.py --decode multi` 与 `tools/evaluate_multipath.py` 现在并排给三条口径
（同一个存活路径表，都走 `evaluate_sample`，所以指标定义完全一致）：

```
multi_best            = multi.best                （历史口径，= metrics 里的主指标）
multi_best_goal       = multi.best_goal           （Goal 里概率最高）
multi_best_goal_cost  = multi.best_goal_cost_path （Goal 里真实 cost 最低）
```

额外集合语义指标：`coverage_rate`（表里至少有一条到终点）、`optimal_coverage_rate`
（表里至少有一条**真实最优**：weighted 按 Dijkstra 最小 cost，无权图退化成跳数，与旧口径等价）、
`mean_goal_paths` / `mean_finished_paths` / `mean_filtered_dead_branches`。
weighted 数据集上额外暴露 `weighted_optimal_coverage_rate`（同一个值，名字点明口径）。
`scripts/evaluate.py` 的结果 JSON 里放在 `multi` 键下；`format_metrics` 增加了
`w_opt_cov` / `goal_paths` / `dead_filtered` 三个短标签。

```bash
# weighted 模型 + 新解码器（4 个组合：null_policy × filter_dead_branches）
E:/CondaEnvData/envs/GGMPC/python.exe tools/evaluate_multipath.py --run outputs/runs/v2_weighted_controlled --data data/weighted_controlled/weighted_controlled_test.pkl --top-k 2 --beam-width 64 --null-policy skip --filter-dead-branches --out outputs/runs/v2_weighted_controlled/mp_weighted_skip_on.json
```

### 21.3 实测（唯一权威表；300 条 weighted test，top_k=2，beam_width=64，seed 0）

两个模型（weighted = 带 Edge Cost Encoder，ablated = 同一份数据但看不到 cost）、
两种 NULL 策略、4 种解码器，全部同一批 query：

| model | null_policy | decoder | goal_hit | optimal | cost_ratio |
|---|---|---|---|---|---|
| weighted | stop | single（新 readout） | **0.8633** | **0.6900** | 1.0070 |
| weighted | stop | multi_best | 0.8667 | 0.6933 | 1.0070 |
| weighted | stop | best_goal | 1.0000 | 0.7833 | 1.0081 |
| weighted | stop | best_goal_cost | 1.0000 | 0.9133 | 1.0026 |
| weighted | skip | single（新 readout） | **0.8633** | **0.6900** | 1.0070 |
| weighted | skip | **multi_best** | **1.0000** | 0.7800 | 1.0081 |
| weighted | skip | best_goal | 1.0000 | 0.7800 | 1.0081 |
| weighted | skip | **best_goal_cost** | **1.0000** | **0.9867** | **1.0002** |
| ablated | stop | single（新 readout） | **0.8767** | **0.3667** | 1.0557 |
| ablated | stop | multi_best | 0.8400 | 0.3733 | 1.0548 |
| ablated | stop | best_goal | 1.0000 | 0.4500 | 1.0551 |
| ablated | stop | best_goal_cost | 1.0000 | 0.7833 | 1.0158 |
| ablated | skip | single（新 readout） | **0.8767** | **0.3667** | 1.0557 |
| ablated | skip | multi_best | 0.9967 | 0.4367 | 1.0558 |
| ablated | skip | best_goal | 1.0000 | 0.4367 | 1.0558 |
| ablated | skip | best_goal_cost | 1.0000 | 0.9300 | 1.0036 |

上表是 `filter_dead_branches=False`（历史默认）。同一批 query 的**集合语义**与搜索统计：

| model | null_policy | filter | coverage | w_opt_cov | mean_goal_paths | mean_finished | mean_filtered |
|---|---|---|---|---|---|---|---|
| weighted | stop | off | 1.0000 | 0.9133 | 3.33 | 10.44 | 0.00 |
| weighted | stop | on | 1.0000 | 0.9133 | 3.34 | 10.02 | 12.12 |
| weighted | skip | off | 1.0000 | 0.9867 | 6.13 | 13.83 | 0.00 |
| weighted | skip | on | 1.0000 | 0.9867 | 6.14 | 13.05 | 16.08 |
| ablated | stop | off | 1.0000 | 0.7833 | 3.64 | 10.29 | 0.00 |
| ablated | stop | on | 1.0000 | 0.7867 | 3.64 | 9.85 | 12.21 |
| ablated | skip | off | 1.0000 | 0.9300 | 7.42 | 13.99 | 0.00 |
| ablated | skip | on | 1.0000 | 0.9233 | 7.51 | 13.32 | 16.55 |

**`filter_dead_branches` 的开关效应**（on − off，只有非零的才列出来）：

| model | null_policy | decoder | Δgoal_hit | Δoptimal | Δcost_ratio |
|---|---|---|---|---|---|
| ablated | stop | multi_best | +0.0000 | −0.0167 | +0.0014 |
| ablated | stop | best_goal | +0.0000 | −0.0200 | +0.0012 |
| ablated | stop | best_goal_cost | +0.0000 | +0.0033 | −0.0002 |
| ablated | skip | multi_best | +0.0000 | −0.0033 | +0.0005 |
| ablated | skip | best_goal | +0.0000 | −0.0033 | +0.0005 |
| ablated | skip | best_goal_cost | +0.0000 | −0.0067 | +0.0001 |

* **weighted 模型：三种配置（off / on / on 重跑）的全部 outcome 指标逐位相同**，预筛选是纯安全网；
* **ablated 模型那几行差值不能归因于过滤器** —— 它们是**跑间噪声**。同一配置连跑两次
  （都用 `--filter-dead-branches --null-policy stop`）：

  ```
  weighted: 两次 difference = none（逐位相同）
  ablated : 两次 difference = multi_best.optimal 0.0033 / best_goal_cost.optimal 0.0067
  ```

  噪声源已定位：同一批输入、同一 seed，`candidate_prob` 在 **cuda 上两次相差 8.6e-07**
  （`z0` 相同），在 **cpu 上两次逐位相同（0.0）**。原因是 GraphFlow 的
  `message.index_add_` / EdgeState 的 `scatter_reduce_` 在 CUDA 上走原子加、
  归约顺序每次不同。weighted 模型的决策概率余量大，1e-6 抖动不会翻 top-k；
  消融模型分布平，会翻掉 2~3 条 query —— 所以只有它抖。
  `single` 一列在所有运行里都相同（采样对 1e-6 抖动不敏感），是这条解释的对照。
* 不变量自检（`outputs/multipath_weighted_summary.json` 生成时跑）：每个 run 都满足
  `best_goal_cost.optimal ≤ weighted_optimal_coverage_rate`，`single` 与 filter 无关 —— 0 violations。
* 只跑开启过滤的结果另有独立产物：`outputs/multipath_weighted_filteron_summary.json`
  （weighted 的数字与上表逐位相同；ablated 有上述噪声）。

### 21.4 这条链路把「模型能力」拆成了三层

以 weighted 模型（skip）为例：

```
single 0.6900  ->  multi_best 0.7800  ->  best_goal_cost 0.9867
   ↑ 最终 argmax readout  ↑ 确定性贪心           ↑ 表里有最优路（上界）
weighted optimal coverage = 0.9867
```

两个模型的同一指标对照（skip，取自 21.3 的表）：

| decoder | 带 edge cost | cost 消融 | 差值 |
|---|---|---|---|
| single optimal（新 readout） | **0.6900** | 0.3667 | +0.323 |
| multi_best optimal | **0.7800** | 0.4367 | +0.343 |
| best_goal optimal | **0.7800** | 0.4367 | +0.343 |
| best_goal_cost optimal | 0.9867 | 0.9300 | +0.057 |
| weighted optimal coverage（**不是解码器**，是表级集合语义） | 0.9867 | 0.9300 | +0.057 |

**必须注意**：`best_goal_cost` / `weighted_optimal_coverage_rate` 在消融模型上也有 0.93，
因为它们衡量的是「表里有没有一条最优路」，而 beam 会把每个 decision 的 top-2 分支都展开、
hop-最优与 cost-最优通常只差一两个 decision —— 表里自然容易包含最优路。而且
`best_goal_cost` 本身就是**按真值 cost 选择**的，已经用了答案。
所以**判断模型是否真的在用 edge weight，要看概率排名那几个口径**（single / multi_best / best_goal，
差距 34~39 个点），不能拿 best_goal_cost 当证据。

### 21.5 beam 宽度对照：beam=64 vs beam=3（CPU，开过滤）

跑在 **CPU** 上（CPU 逐位可复现：同配置连跑两次，除 `wall_time` 外结果完全相同），
`--filter-dead-branches`，`top_k=2`，300 条 weighted test。`single` 不受 beam 影响，两列恒等，故不重复列。

| model | null | 指标 | beam=64 | beam=3 | Δ |
|---|---|---|---|---|---|
| weighted | stop | multi_best optimal | 0.6933 | 0.6933 | 0 |
| weighted | stop | best_goal optimal | 0.7833 | 0.7833 | 0 |
| weighted | stop | best_goal_cost optimal | 0.9133 | 0.9100 | −0.0033 |
| weighted | stop | weighted optimal coverage | 0.9133 | 0.9100 | −0.0033 |
| weighted | skip | multi_best optimal | 0.7800 | 0.7800 | 0 |
| weighted | skip | best_goal optimal | 0.7800 | 0.7800 | 0 |
| weighted | skip | best_goal_cost optimal | 0.9867 | **0.9800** | −0.0067 |
| weighted | skip | weighted optimal coverage | 0.9867 | **0.9800** | −0.0067 |
| ablated | stop | best_goal_cost optimal | 0.7833 | 0.7533 | −0.0300 |
| ablated | stop | weighted optimal coverage | 0.7833 | 0.7533 | **−0.0300** |
| ablated | skip | best_goal_cost optimal | 0.9267 | 0.8767 | −0.0500 |
| ablated | skip | weighted optimal coverage | 0.9267 | 0.8767 | **−0.0500** |

| model | null | beam | coverage | w_opt_cov | mean_goal_paths | mean_finished | mean_pruned |
|---|---|---|---|---|---|---|---|
| weighted | stop | 64 | 1.0000 | 0.9133 | 3.34 | 10.02 | 0.00 |
| weighted | stop | 3 | 1.0000 | 0.9100 | 2.86 | 8.78 | 0.33 |
| weighted | skip | 64 | 1.0000 | 0.9867 | 6.14 | 13.05 | 0.00 |
| weighted | skip | 3 | 1.0000 | 0.9800 | 4.48 | 9.60 | 0.91 |
| ablated | stop | 64 | 1.0000 | 0.7833 | 3.64 | 9.85 | 0.00 |
| ablated | stop | 3 | 1.0000 | 0.7533 | 3.13 | 8.78 | 0.29 |
| ablated | skip | 64 | 1.0000 | 0.9267 | 7.51 | 13.32 | 0.00 |
| ablated | skip | 3 | 1.0000 | 0.8767 | 5.24 | 9.48 | 1.07 |

**读数**：

* `multi_best` / `best_goal`（概率排名口径）在 beam=3 下**一个点都不掉** —— 最优路本来就排在前面，
  beam 再窄也留着；`coverage` 也恒为 1.0000；
* 掉的是「表里有最优路」这一族：weighted 只掉 0.33~0.67pt，ablated 掉 **3.0~5.0pt**；
* 于是**两个模型在 w_opt_cov 上的差距被窄 beam 拉开**：stop 13.0pt → 15.7pt，skip 6.0pt → **10.3pt**。
  这说明宽 beam 之前在替消融模型兜底：它靠扫更多分支把最优路捡进表里；把 beam 收到 3、
  只保留模型自己排得高的分支后，消融模型就漏了，weighted 模型不漏。
* 产物：`outputs/multipath_weighted_beam_compare.json`（8 次运行的完整记录）+ 各 run 目录下的
  `mp_cpu_beam{64,3}_{stop,skip}.json`。

### 21.6 零破坏验证

同一个未加权历史产物（`v2_rev2_mixed` × `controlled_long` × top_k=2 skip）**逐位复现**：

```
coverage_rate 1.0000 / optimal_coverage_rate 0.9975 / mean_goal_paths 7.0275
mean_finished_paths 21.455 / mean_expanded 40.91 / max_depth 10
multi_best: goal_hit 1.0000 / optimal 0.9450 / cost_ratio 1.0024
本轮重跑的每一项 delta = 0.0
```

### 21.7 新增测试（`tests/test_multi_path_decoder.py` 24 个，其中 14 个是本轮加的）

对应指南第 11 节：dead 不占 top-k、关筛选逐位复现旧行为、全部候选被筛掉判 broken 不崩、
loop 逻辑不变、beam 剪枝不变、weighted path cost（26.0 / 10.0）、无权退化成跳数、
`best` 保持旧语义（broken 也能是 best）、`best_goal` 取概率最高、
`best_goal_cost_path` 取真实 cost 最低（2 跳 cost=21 vs 3 跳 cost=6 → 后者）、
weighted optimal coverage 需要真正的最小 cost 路。另外加了 3 个 evaluator 接线测试
（`report.multi` 三条口径 + 无权不暴露 weighted 别名 + weighted 上暴露）。
全部测试：`pytest tests -q` → **307 passed**；`semantic_check.py --strict` → **0 problems**。

### 21.8 全模型统一评测报告

`docs/REPORT_multipath_and_weighted.md`：8 个模型配置（加权 / 消融 / 无权 / 跨任务）× 2 个测试集
（加权 300 条、无权 300 条）× 单分支 + 多分支（`top_k=2, beam_width=3, filter=True`，stop/skip 各一遍），
CPU 跑、逐位可复现；含完整指标定义、解码器定义、参考基线、结论与读数陷阱。
机器可读汇总在 `outputs/all_models_multipath_summary.json`，16 份原始评测在 `outputs/multipath_report/`。

### 21.9 复现用的产物

```
outputs/multipath_weighted_summary.json                        # 6 次评测的汇总表
outputs/runs/v2_weighted_controlled/mp_weighted_{stop,skip}_{off,on}.json
outputs/runs/v2_weighted_controlled_cost_ablated/mp_weighted_{stop_off,skip_on}.json
```

---

## 22. 交互看板（`dashboard/`）：加权模型 + 指标报告

```bash
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py            # http://127.0.0.1:8765/
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py --no-browser --device cpu
```

三个页签：**路径可视化** / **测试指标可视化** / **实验报告**。

### 22.1 加权模型与多分支口径怎么进面板的

| 位置 | 内容 |
|---|---|
| 模型筛选器 | 每个 run 按 `run_config.json` 的 `data.weighted` + `model.use_edge_cost` 打徽章：**带权** / **带权·无cost**（消融）/ **无权** |
| 指标页 · 口径 | 一次评测会展开成多行：`single`、`multi`、`multi · best_goal`、`multi · best_goal_cost`、`multi k=2 / stop|skip` —— 直接对照方案第 8 节的三条口径 |
| 指标页 · 新列 | **类别**（徽章）、**W-opt cov**（`weighted_optimal_coverage_rate`）、**Goal paths**（`mean_goal_paths`） |
| 指标页 · 数据集 | `weighted_controlled_{train,val,test}.pkl` 与无权数据集并列，按 `data/` 子目录分组 |
| 路径页 · 多分支 | 控制条新增 **必死 branch 预筛选** 开关（透传 `filter_dead_branches`），summary 里显示剔除数量 |
| 路径页 · 带权图 | 每条路线同时显示 **跳数** 与 **真实 cost**（后端 `_route_payload` 新增 `path_cost` / `weighted`） |

数据集归属不再靠文件名猜：`scripts/evaluate.py` 现在把 `data` 写进评测 JSON，看板优先用它；
加权 run 里的老产物（没有 `data` 字段）按 `data.weighted=true` 兜底判到加权测试集。另外同一口径
有多份产物时，**`best.pt` 的评测优先于 `*_last.json`**。

### 22.2 实验报告页签

`GET /api/reports` 返回固定清单（不遍历目录），`GET /api/reports/<id>` 返回内容：
Markdown 原样返回（前端用内置小渲染器画标题/表格/列表/代码块），JSON 解析后格式化展示。

| id | 文件 |
|---|---|
| `report` | `docs/REPORT_multipath_and_weighted.md` |
| `all_models` | `outputs/all_models_multipath_summary.json` |
| `weighted_experiment` | `outputs/weighted_experiment.json` |
| `multipath_weighted` / `multipath_filteron` | 过滤 on/off 的 beam=64 对照 |
| `beam_compare` | beam=64 vs beam=3（CPU） |
| `regression` | 零破坏回归：旧 checkpoint 逐位复现 |

### 22.3 API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/catalog` | 模型（含 `kind` / `weighted` / `use_edge_cost` / `flow_steps`）、数据集、指标行、报告清单 |
| GET | `/api/datasets/<id>` | 数据集规模 |
| GET | `/api/reports/<id>` | 报告内容（Markdown 文本 / JSON 对象） |
| POST | `/api/path` | 一次真实反向扩散 + 单路径/多分支解码，返回图、路线（含 `path_cost`）与 summary |

截图（Chromium 实跑，`outputs/figures/dashboard_*.png`）：指标页按加权测试集筛选、报告页渲染本文第 21 节报告。

---

## 23. single-path 最终 readout 变更（采样 -> 最终 argmax）

### 23.1 为什么改

旧默认的 single 解码直接吃 reverse chain **采样**出来的 `z0`。采样只是"从模型分布里抽到的一条"，
不等于模型最可能的那条：weighted 测试集 #92 上，模型在最后一个路口（node 8）对 `NULL` 和 `->9` 都给了
概率，**采样抽到 NULL** → 16 跳中止；而模型自己的 argmax 明明是 `->9` → 18 跳到达。
用户看到"扩散过程连上了、单步解码没连上"，根因就在这里。

### 23.2 改成了什么（只动最终 readout）

```
旧: stochastic reverse chain -> sampled z0                -> decode_flat
新: stochastic reverse chain -> 最后一步 candidate_prob
                             -> 每个 decision 组内 argmax -> z0_argmax -> decode_flat
```

* **reverse chain 一个字没改**：仍然逐步按 posterior 采样（`stochastic=True` 默认不变），
  `z_T ~ pi`、`q(z_{t-1} | z_t, ẑ_0)` 的数学定义都没动；
* 变的是 **readout**：`grouped_argmax(candidate_prob)`（每个 decision 只在自己的候选里取最大，
  不是 flat 全局 argmax），NULL / source decision 的候选语义沿用 DecisionField；
* **multi-path 完全没动**：仍然直接用整组 `candidate_prob` 展开 top-k，历史结果逐位不变
  （已验证：同一配置的新旧产物 `multi_best` / `best_goal` / `best_goal_cost` / coverage 全等）；
* **sampled z0 仍然保留**：扩散可视化的"链状态"档、调试都用它，只是不再驱动 single 解码；
* **`stochastic=False` 的全程 argmax rollout 也保留**，它是另一条路径（每一步都取 argmax），
  与新的 single 是两回事；面板上的开关现在叫「全程 argmax（对照）」。

### 23.3 三个状态的命名（别再混）

| 名字 | 是什么 | 谁在用 |
|---|---|---|
| `single`（默认） | 最后一步 `candidate_prob` 的**组内 argmax** | 评测主口径；面板"解码使用"那一行 |
| `single_sampled`（诊断） | 采样出来的 `z0` | 旧行为，保留用于对照；面板"链状态（仅诊断）" |
| deterministic rollout | **每一步** posterior 取 argmax | `stochastic=False` / 面板「全程 argmax（对照）」 |

代码入口：`src/evaluation/readout.py`（`single_readout_state` / `single_path_state`）。
`evaluate_dataset(decode=...)` 现在接受 `single` / `single_sampled` / `multi`；
`tools/evaluate_multipath.py` 同时输出 `single_path`（新）与 `single_sampled`（诊断）。

### 23.4 #92 回归（weighted 测试集，`v2_weighted_controlled`，seed 0，CPU）

```
readout z0 = [2, 5, 10, 16, 20]   -> goal    18 跳 / cost 101.29
sampled z0 = [2, 5, 10, 16, 19]   -> broken  16 跳 / cost  92.35   NULL selected at 8
（decision 4 上 readout=->9，采样=NULL）
```

测试：`tests/test_single_readout.py`（8 个）覆盖 grouped argmax 正确性、single 不再读采样状态、
采样状态变化不影响 single、multi 与采样状态无关、采样链仍是 stochastic、#92 回归。

### 23.5 300 条上的口径变化（CPU，seed 0）

| 模型 / 测试集 | single（新 readout） | single_sampled（旧） |
|---|---|---|
| weighted 模型 / weighted test | **0.8633 / 0.6900** | 0.8400 / 0.6800 |
| cost 消融 / weighted test | **0.8767 / 0.3667** | 0.7600 / 0.2867 |
| 无权模型跨任务 / weighted test | **0.9433 / 0.5067** | 0.9367 / 0.4867 |
| rev2_mixed / controlled test | 0.9233 / **0.9000** | 0.9300 / 0.8833 |
| rev2_longmix / controlled test | **0.9567 / 0.9500** | 0.9300 / 0.9033 |
| controlled_flow3 / controlled test | **0.9733 / 0.8567** | 0.9167 / 0.7233 |
| controlled_flow1 / controlled test | **0.9967 / 0.8767** | 0.8933 / 0.7333 |

（格式 `goal_hit / optimal_path_rate`。）总体规律：**argmax readout 让 `optimal` 普遍上升**
（weighted 0.6800→0.6900、消融 0.2867→0.3667、flow1 0.7333→0.8767），`goal_hit` 多数也升；
少数模型（rev2_mixed、weighted-跨无权）`goal_hit` 略降 0.7~1.3 个点 —— 因为 argmax 链会更"自信"地
走它认为对的那条，而采样有时会瞎走到终点。两种口径都保留，报告里并排列出。

> **注**：第 0.2 节模型表里较早那些 `controlled_test` / `long` / `oldv1_test` 的 single 数字是在本次
> 改动**之前**测的（等价于现在的 `single_sampled`），没有重跑；要看新口径请用第 23.5 节或
> `docs/REPORT_multipath_and_weighted.md`。

---

## 24. DiDi 成都真实道路数据接入（真实车辆历史路径当 GT）

对应实施方案 `Graph_Junction_Diffusion_DiDi_RealData_Implementation_Plan.md`。
这一节是这条链路的**唯一使用说明**：数据语义、实测结论、四个命令、与方案的偏离。

### 24.1 任务变了：GT 不再是最短路

合成数据那条链是 `随机图 -> BFS/Dijkstra -> 最短路当 GT`。真实数据这条链是：

```
固定成都路网 G + 道路长度 W + OD(s,g)  ->  CSV 里真实车辆走过的 junction 路径（GT）
                                      ->  OD corridor 子图
                                      ->  Branch Segment -> z0 -> Diffusion
```

**必须区分"边权进入模型"和"GT 必须是最短路"**：

| | 合成数据 | DiDi 真实数据 |
|---|---|---|
| 模型输入 | 可选 edge weight | 道路长度（`length`，米） |
| GT | Dijkstra 最小 cost 路径 | **真实司机历史路线** |
| Dijkstra 的角色 | 生成 GT | 只当 `C*` 标尺（CostRatio 的分母） |

实测成都数据 **93.7% 的 GT 都不是最短路**，GT/Dijkstra cost ratio 中位数 1.12、
max 2.24。所以 `Optimal Path Rate` 在真实数据上不再是"模型对不对"的判据 —— 它必须
降级成 secondary metric，主指标换成路径相似度（见 24.6）。

### 24.2 原始文件与语义

数据在 `data/DiDiChengduXian/didi_datasets/datasets/didi_chengdu/`：

| 文件 | 用途 |
|---|---|
| `dicts.pkl` | `road_id -> (u, v, key)`，重建 junction graph + road path 转 junction path |
| `edge_features.csv` | 道路静态属性；第一版只取 **`length`**（已确认列名）当 edge weight |
| `20161010~19.csv` | 真实车辆轨迹，核心字段 `path` = **road id 序列**（不是 junction id） |
| `line_graph_edge_idx.npy` | 只用于校验 road 转移；**不作为主模型图** |
| `transition_prob_mat.npy` | 第一版不进模型 |

### 24.3 实测结论（这些数决定了实现方式，改代码前先读）

* **折叠后的成都路网只有 2891 节点 / 4403 边**，单连通分量。6639 条 road segment
  里有 2226 对平行路段（同一 `(u,v)` 多个 `key`）按方案第 3.3 节折叠成最短的那条，
  另丢弃 10 条自环 road。
* 2466 / 2891 个节点度 >= 3 —— 这是一张**很密**的城市图，"junction" 几乎就是全部
  路口。所以 corridor 里 decision 数天然就大（中位数 ~300）。
* 轨迹相邻 road 在 `idx2edge` 的**存储方向**上 100% 连续（80000 行实测，没有一条
  需要反向）。所以 road -> junction 转换的主算法用存储方向（精确、天然处理掉头），
  方案第 4.1 节的端点集合写法作为兜底（它的前两条 road 必须**一起消费**，否则每条
  road 都会算错一次）。
* 约 1.6% 的相邻 road 对是"平行路段掉头"（`{u,v}` 相同、方向相反），会形成重复
  junction，由 `require_simple_gt` 过滤；实测 80000 行里 37.8% 的轨迹含环路。
* 清洗漏斗（80000 行）：parse 100% -> road id 100% -> 连续 100% -> 长度区间与简单
  路径各 36610/80000 = 45.8% -> 去重后 27570（轨迹重复率 24.7%）-> 抽样 8000 ->
  corridor 保留 81.4%。

### 24.4 corridor 的 rho：方案里的 98% 在这张网上拿不到

方案第 6.3 节要求"train GT containment >= 98% 的最小 rho"。实测（train split
**5760** 条候选 = `max_dataset_samples=8000` 按 72/8/20 划分后的 train，weighted
距离口径，见 `data/didi_chengdu_gjd/scan_corridor.json`）：

| rho | train GT containment | mean corridor nodes | mean decisions |
|---:|---:|---:|---:|
| 1.1 | 0.486 | 153 | 114 |
| 1.2 | 0.685 | 264 | 211 |
| 1.3 | 0.808 | 365 | 296 |
| 1.4 | 0.872 | 458 | 377 |
| **1.5** | **0.914** | **546** | **452** |
| 1.8 | 0.966 | 789 | 661 |
| 2.0 | 0.977 | 938 | 789 |
| 2.5 | 0.991 | 1268 | 1073（44% 的整张城图）|

**98% 只有 rho >= 2.4 才够，那时 corridor 已经吞掉半座城市、每个样本 1000+ decision，
"走廊"这个设计本身失效了。** 所以第一版取 **rho = 1.5**：corridor 只保留 ~19% 的
城市节点、train containment 0.914，剩下 ~9% 按方案第 6.3 节记 `corridor_miss`
并从数据集里剔除，`stats.json` 里单独报 `corridor_retention`。

想更贴近方案的 98% 就把 `data.corridor.rho` 改成 1.8（retention 0.97，但 decision
数 +47%、样本体积与训练时间按比例上升）。rho 只在 train 上选、选定后冻结，val/test
不得重新按 GT 调整。

附带一个反直觉但重要的结论：**hop 距离口径比 weighted 差**。同样 containment 下
hop 椭球要大得多（rho_hop=2.5 才 98.7%，而 weighted rho=1.5 就 91.2%、节点数只有
它的 44%）。所以 corridor 用道路长度算距离是对的。

### 24.5 四个命令

```bash
# 阶段 0：扫描（不改数据）—— 确认 edge_features 列名、道路长度统计、转换率、
#         invalid road id、连续性失败率、环路 GT 率、唯一路径数、GT cost ratio 分布
python scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --scan-only

# 阶段 1：corridor rho 扫描，选出并冻结 rho
python scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --scan-corridor

# 阶段 2：生成正式数据集
python scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml --build

# 自检（不需要 pytest / torch；20 条断言 + 已生成数据集复核）
python tools/verify_didi_pipeline.py --data data/didi_chengdu_gjd
```

`--max-samples N` 覆盖 `data.max_dataset_samples`（最终数据集规模）。它和
`data.max_candidates` **不是一回事**：后者只是解析期的内存/时间闸门。候选要读满
10 个日期文件才有 OD/时段多样性，所以先尽量多收集、再按固定 seed 均匀抽到目标规模。

`--set data.corridor.rho=1.8` 之类的覆盖照旧可用。候选轨迹会缓存到
`data/didi_chengdu_gjd/_didi_candidates.pkl`（签名含文件 mtime/size + 过滤条件），
换配置会自动失效重读，`--refresh-cache` 强制重读。

### 24.6 产出与指标

```
data/didi_chengdu_gjd/
├── graph_global.pkl          全局无向有权 junction graph + 建图统计
├── train.pkl / val.pkl / test.pkl
├── test_1000.pkl             GDP 风格固定 1000 条
├── shuffled_od_1000.pkl      OD 打乱重配；没有真实 GT，只测 GoalHit/Loop/Broken/CostRatio/时间
├── split_manifest.csv        sample_id/order_id/date/split/长度/decision/gt_cost/gt_cost_ratio
├── metadata.json             建图统计 / rho / 过滤条件 / 划分 / 规模
├── stats.json                funnel（每步保留率）/ corridor_retention / 各 split 摘要
├── scan_only.json            --scan-only 的产物
└── scan_corridor.json        --scan-corridor 的产物
```

当前这份（seed 0、rho 1.5、`max_dataset_samples=8000`）：

| split | n | decision 均值 | corridor 节点均值 | NULL 候选占比 | corridor retention |
|---|---:|---:|---:|---:|---:|
| train | 4687 | 348 | 426 | 0.230 | 0.814 |
| val | 533 | 361 | — | 0.230 | 0.833 |
| test | 1275 | 361 | — | 0.230 | 0.797 |
| test_1000 | 1000 | — | — | — | — |
| shuffled_od_1000 | 741 | — | — | — | 无真实 GT |

train/val/test 的轨迹键两两零交集（`stats.json` 里 `split_overlaps` 全 0，自检脚本
也会断言）。三个 split 的 GT 都带 `gt_source == "observed"`。

真实数据多出来的指标（只在 `meta['gt_source'] == 'observed'` 时计算，**旧实验的
JSON 结构与字段名一个字节都没改**）：

* `path_similarity_score = mean( 1(goal_hit) * nLCS )`，`nLCS = LCS(pred, gt)/|gt|`
  —— 没到终点记 0，用来选 `best.pt`；
* `normalized_lcs_success`（只在成功样本上算）、paired `edge_precision/recall/f1`
  （无向边规范化后比较）；
* `gt_cost_ratio` / `pred_cost_ratio`（辅助指标，不是主指标）；
* `klev` / `jsev`（dataset-level 的 edge visit 分布散度，DiffPath 风格）；
* `buckets.length_buckets`（按 GT 长度等量三分）、`buckets.decision_buckets`
  （按 `num_decisions` 分 1-3 / 4-6 / 7-9 / >=10）。

### 24.7 训练与评测

```bash
# 正式训练（flow_steps=1、T=50、batch 16、AdamW lr 1e-4、AMP，100 epoch）
python scripts/train.py --config configs/graph_flow_didi_weighted.yaml \
  --name didi_chengdu_flow1_weighted \
  --data data/didi_chengdu_gjd/train.pkl --val-data data/didi_chengdu_gjd/val.pkl

# 完整 test + Dijkstra baseline
python scripts/evaluate.py --config configs/graph_flow_didi_weighted.yaml \
  --checkpoint outputs/runs/didi_chengdu_flow1_weighted/best.pt \
  --data data/didi_chengdu_gjd/test.pkl --deterministic --baselines \
  --out outputs/runs/didi_chengdu_flow1_weighted/eval_test.json

# GDP 风格主表
python scripts/evaluate.py ... --data data/didi_chengdu_gjd/test_1000.pkl

# shuffled OD（自动跳过相似度指标，只报 GoalHit/Loop/Broken/CostRatio/时间）
python scripts/evaluate.py ... --data data/didi_chengdu_gjd/shuffled_od_1000.pkl

# 最干净的消融：同一份数据、同一网络、只切 model.use_edge_cost true/false
python scripts/train.py --config configs/graph_flow_didi_weighted.yaml \
  --name didi_chengdu_flow1_noedgecost --set model.use_edge_cost=false ...
```

配置里三处与 synthetic 不同的关键项：`split.split_by_graph: false`（真实数据是
**一张固定城市图**，必须按 path 划分）、`flow_steps: 1`（推理必须与训练同轮数）、
`training.selection_metric: path_similarity_score`（`GoalHit` 在真实数据上会较早
饱和，而路径相似度还在改善）。**loss 第一版没有改**：仍然是
`L = CE + 0.1 * SoftGoal` —— 真实 GT 是司机选择而不是最优解，直接加 cost loss 会把
模型拉回"只追最短路"，改变研究任务本身。

OOM 时按 `16 -> 8 -> 4` 降 batch，**不要**先降模型维度。

**先做小规模验证再上全量**（方案第 14 节阶段 3/4）。用 `--max-samples` 造小数据集，
它控制的是"最终抽多少条候选"，不要用 `--max-rows-per-file` 去凑：

```bash
# 阶段 3：smoke test（约 1 分钟出数据）—— train ~580 / val ~80 / test ~150
python scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml \
  --build --max-samples 1000 --out-dir data/didi_chengdu_smoke

# 阶段 4：小样本 overfit（32~64 条，反复训到 x0 acc 接近 1、PathSim 明显升高）
python scripts/prepare_didi.py --config configs/graph_flow_didi_weighted.yaml \
  --build --max-samples 220 --out-dir data/didi_chengdu_overfit
```

真实小样本都过拟合不了，**不要**直接跑全量。

### 24.8 新增 / 修改的文件

| 文件 | 状态 | 内容 |
|---|---|---|
| `src/data/didi_dataset.py` | 新增 | dicts/edge_features 读取、建全局有权图、road->junction 转换、corridor、距离缓存、过滤/去重/划分、funnel 统计 |
| `scripts/prepare_didi.py` | 新增 | `--scan-only` / `--scan-corridor` / `--build` 三个阶段 + 候选缓存 |
| `configs/graph_flow_didi_weighted.yaml` | 新增 | DiDi 配置（含 rho 取舍的实测表） |
| `src/evaluation/real_path_metrics.py` | 新增 | LCS / nLCS / paired Edge PRF / PathSimilarityScore / KLEV / JSEV / cost ratio / 分桶 |
| `tests/test_didi_dataset.py` | 新增 | 转换 / corridor 无泄漏 / observed GT 不被替换 / 划分无交集 / flow_steps=1 |
| `tests/test_real_path_metrics.py` | 新增 | 每个指标的手算小样例 |
| `tools/verify_didi_pipeline.py` | 新增 | 不依赖 pytest / torch 的自检脚本（23 条断言） |
| `tools/visualize_didi_samples.py` | 新增 | 把数据集样本画在**真实经纬度底图**上（见 24.11） |
| `src/data/dataset_builder.py` | 改 | **新增** `relabel_graph_and_path_to_contiguous()` / `build_sample_from_observed_path()`；**旧函数语义一字未改** |
| `src/data/decision_field.py` | 改 | 校验第 6 条放宽：允许"绕回 owner 的 loop branch"（见 24.9） |
| `src/data/dataset.py` | 改 | docstring 说明 GT 语义；`summary()` 增 `gt_cost` / `gt_cost_ratio` / `gt_source` |
| `src/training/trainer.py` | 改 | `training.selection_metric` / `selection_mode`，默认仍是 `goal_hit_rate` / `max` |
| `src/evaluation/evaluator.py` | 改 | `EvaluationReport.real`；有观测 GT 时把 `path_similarity_score` 等提到顶层 metrics |
| `src/evaluation/baselines.py` | 改 | 新增 `real_baseline_summary()`：Dijkstra/greedy 的 nLCS / Edge F1 / CostRatio |
| `scripts/evaluate.py` | 改 | 输出 `real_path_metrics` / `distribution_metrics` / `buckets`；shuffled OD 自动跳过相似度指标 |

### 24.9 实现时踩到的坑

1. **corridor 裁剪会造出"绕回 owner 的 loop branch"。** 子图提取会**降低节点度**，
   把原本度 >= 3 的节点降成度 2，于是出现 `O -> X -> Y -> O` 这样的分支。它在语义上
   完全合法（`branch_reach_target` 认出这是"绕行回到自己"的候选，而且永远不可能成为
   GT branch），但 `validate_decision_field` 的"branch 节点不得重复"断言会把整条样本
   丢掉 —— 实测会丢掉 **13.4%** 的样本。修法是只放行"首尾相同、内部不重复"这一种
   情形，这是**严格更宽松**的改动：合成管线的 branch 本来就无重复节点，旧行为逐位不变。
   修完 build 失败率从 13.4% 降到 0，这类 branch 只占全部 branch 的 0.02%。
2. **端点集合兜底的前两条 road 必须一起消费**，否则每条 road 都会算错一次位置。
3. **`target_gt_containment` 不能照抄 0.98**（见 24.4）。
4. **样本体积**：一个 ~350 decision 的 corridor 样本 pickle 之后约 110KB，其中
   `segments` 与 `field` 远大于 `graph` 本身。所以 8000 条落地约 1GB。
   `data.max_dataset_samples` 就是为这个加的闸门。

### 24.10 方案的刻意偏离与未完成项

**刻意偏离（都有实测依据）**

* `rho` 从"方案要求 98%"改成"train containment >= 0.91 的最小 rho = 1.5"（24.4）。
* `data.length_column` 直接写成 `length`（已用 `--scan-only` 核实），而不是留
  `null` —— 留 null 会让 `--build` 直接报错；代码里"不许猜列名、不许退化成
  `weight=1`"的硬校验一条都没少。
* 放宽了 `validate_decision_field` 的 branch 唯一性断言（24.9 第 1 条）。

**一处必须更正的早期结论：坐标其实是有的**

第一版我在 README / 方案偏离里写过"`dicts.pkl` 只有 OSM node id，没有 junction
经纬度，所以不伪造 DTW"。**这是错的**，错因是 `ChengDu.pkl` / `graph.pkl`
（`data/DiDiChengduXian/data/data/cd/`，两份内容完全相同）是 OSMnx 1.1.1 导出的
`MultiDiGraph`，**边属性里带 `shapely.geometry.linestring.LineString`**；本机没装
`shapely`，`pickle.load` 直接抛 `No module named 'shapely'`，于是"打不开"被误读成
"没有坐标"。实际情况：

| 事实 | 数值 |
|---|---|
| 图上 `crs` | `epsg:4326`（WGS84 经纬度） |
| 节点属性 | `x` = 经度，`y` = 纬度，`street_count` |
| 节点 id | 与 `dicts.pkl` 的 `(u, v)` **同一套 OSM node id** |
| 对**我们用的那张图**的覆盖率 | **2780 / 2891 = 96.2%**（缺的 111 个用邻居坐标迭代填充） |
| 覆盖范围 | 经度 104.0354~104.1313，纬度 30.6502~30.7399 ≈ **9.1 km × 9.9 km** |

注意两份 `dicts.pkl` **并不相同**：我们用的是 `didi_datasets/datasets/didi_chengdu/`
（6639 road / 2891 节点），而 `ChengDu.pkl` 配的是 `data/data/cd/dicts.pkl`
（6566 road / 2848 节点），所以覆盖率不是 100%，必须处理缺失。

`src/data/didi_dataset.py::load_node_coordinates()` 内建了一个最小 `shapely` 替身
（只在 `ImportError` 时注入，只需要能被 pickle 的 `__setstate__` 接住，不解析几何），
`attach_coordinates()` 负责写回 `x`/`y` 并补缺失。于是：

* **可视化可以用真实地理底图** —— 见 `tools/visualize_didi_samples.py`
  （默认就是 `--geo`，`--topology` 才退回拓扑布局）；
* **方案第 12.5 节的 km-based DTW 现在具备实现条件**（尚未实现，见下）。

**未完成（本机环境限制）**

* 本机 `torch` 未安装（也没有网络装包），所以**方案第 14 节的阶段 3（256 样本
  smoke test）、阶段 4（32~64 样本 overfit）、阶段 5（正式训练）以及第 16 节的
  全部评测都没有跑**。数据侧（阶段 0~2）已完整跑通并自检通过；
  `tests/` 里的两个新测试文件也**没有被执行过**（`tests/conftest.py` 要 import
  torch —— 本机 pytest 也没装）。请在有 torch 的环境里先跑：

  ```bash
  python -m pytest tests/test_didi_dataset.py tests/test_real_path_metrics.py -q
  python tools/verify_didi_pipeline.py --data data/didi_chengdu_gjd   # 不需要 torch，已通过
  ```

### 24.11 样本可视化（真实地理底图）

```bash
# 默认：真实经纬度底图 + 裁到每条样本自己的范围（街道细节清楚）
python tools/visualize_didi_samples.py --per-split 3 --out outputs/figures/didi_samples_geo.png

# 不裁剪：看这些样本落在整座成都的什么位置（能看到环路结构）
python tools/visualize_didi_samples.py --no-crop --per-split 3 --out outputs/figures/didi_city_overview.png

# 只画 corridor 本身，放大看 decision node 与 GT 拐弯
python tools/visualize_didi_samples.py --zoom --per-split 2 --out outputs/figures/didi_zoom.png

# 退回拓扑 spring 布局（对比用：形状与真实地图完全不同）
python tools/visualize_didi_samples.py --topology --out outputs/figures/didi_topology.png
```

每个 panel：浅灰 = 整张成都路网（真实经纬度）；浅蓝 = 该样本的 OD corridor；
红粗 = GT 真实司机历史路径；绿虚线 = 同一 OD 的 Dijkstra 最短路；
绿星/红星 = start / goal；橙点 = decision node。

**坐标怎么来的**：`ChengDu.pkl` / `graph.pkl` 是 OSMnx 导出的 `MultiDiGraph`，
节点自带 `x`(lon) / `y`(lat)。本机没装 `shapely`，而该 pickle 的边属性带
`LineString` 几何，直接 `pickle.load` 会抛 `No module named 'shapely'` ——
`load_node_coordinates()` 会在这种情况下注入一个最小替身再读，**只取节点 x/y，
不解析几何**。装不装 shapely 都不影响结果。

画图用等距圆柱投影（`x = lon·cos(lat0)`，`y = lat`），在这个 9km × 10km 的范围内
形变可忽略；panel 之间锁了等比例，所以长度可以直接互相比较。

**从图上能直接读出来的事**：

1. **GT 不是最短路** —— 15 条抽样里只有 1 条 `GT == Dijkstra`。红线在大量路口和
   绿虚线分叉，有的样本（如 train #199，ratio 1.60）绕得相当明显。
2. **decision 数远大于 GT 长度** —— 例：train #250 的 GT 只有 15 跳、但 corridor 里
   有 402 个 decision、1762 个 candidate；test #432 更是 682 decision。橙点铺满整片
   corridor，红线只穿过其中十几个，即 **95%+ 的 decision 是 NULL**。
   （候选层面的 NULL 占比只有 0.23，因为每个 decision 组里的 NULL 只占 1/(1+branch)。）
3. **corridor 覆盖的是一片连续城区**，不是一条细走廊 —— 这是 rho=1.5 在这张密网上
   的必然结果（见 24.4）。
4. **真实司机路径有折返和绕行**，且这些样本仍然满足 `require_simple_gt`
   （折返发生在不同 junction 之间，不是回到同一个路口）。

### 24.12 第二轮评审修复（4 个逻辑问题 + 1 个文档错误）

这一轮**没有改任何实验设计**（rho / flow_steps / loss / 数据规模都没动），只修了
会让论文数字变错的地方。数据集已按修正后的代码重建。

#### ① KLEV / JSEV 的编号空间错了 —— 最严重的一个

每个真实样本的 corridor 都被独立 relabel 成 `0..N-1`。对**单样本**指标
（nLCS / Edge F1 / CostRatio）没问题，但 evaluator 把**不同样本**的局部编号路径
直接汇总去做 edge visit 分布，于是"样本 A 的边 (0,1)"和"样本 B 的边 (0,1)"
被当成了同一条城市道路 —— 实际上它们是两条完全无关的路。

修法：`build_sample_from_observed_path()` 现在把反查表写进
`sample.meta['local_to_global']`（`local_to_global[new_id] = old_osm_id`），
evaluator 用 `real_path_metrics.to_global_path()` 映射回**全局 OSM id** 之后才算
KLEV / JSEV。`evaluate.py` 的 JSON 里 `real_path_metrics.distribution_node_space`
会标明 `global_osm_id`。

测试：`test_distribution_metrics_use_global_node_ids` —— 构造两个 relabel 后都是
`[0,1,2]` 的样本，断言局部口径只有 2 条"边"（错）、全局口径有 4 条（对）。

#### ② `count_decisions()` / `corridor_decision_count()` 把 Start 多算一次

口径是集合运算 `{deg>=3} ∪ {s : deg(s)>1} \ {g}`，但实现写成了
"先数 `deg>=3`，再 `if deg(start)>1: +1`" —— 当 `deg(start) >= 3` 时 start
已经在第一个集合里，被算了两次。

实测影响：**504/600 = 84% 的 corridor 样本被高估 1 个 decision**
（平均 +0.84）。不影响 `sample.num_decisions`（它来自 branch segment，定义一直是对的），
但 `scan_corridor.json` / `metadata.json` / `candidate.num_decisions` 全部偏大。
现在两个函数都改成 set 写法，并加了和 `branch_segments.build_decision_nodes()`
逐一对齐的测试。

#### ③ `target_null_fraction` 算的是候选级，不是标签级

原来算的是"候选表里 NULL 候选占多少"。但训练标签 `z_0` 的口径是
"每个 decision 选了 NULL 还是某条 branch"，两者差一个量级：

| 口径 | 数值（train 前 50 条实测） |
|---|---|
| candidate 级（旧，误标为 target） | 0.229 |
| **标签级（新，真 target）** | **0.938** |

真实 corridor 有 ~300 个 decision，GT 只经过其中几十个。所以
**`target_null_fraction ≈ 0.936`**（train/val/test 分别是 0.9356 / 0.9345 / 0.9342），
这才是判断 `null_weight / active_weight` 会不会造成 NULL collapse 时要看的数。

现在 `summary()` 同时给 `candidate_null_fraction` 与 `target_null_fraction`
（后者按 decision 加权：`Σ空标签 / Σdecision`），`stats.json` 里也分开记，不再混用。

#### ④ shuffled OD "必须 1000" 现在由代码强制

新增 `split.strict_shuffled_size: true`（DiDi 配置已打开）。凑不满就在 `--build`
阶段直接 `raise RuntimeError`，而不是打个 warning 继续 —— 论文主表要固定
`test_1000 = 1000` 且 `shuffled_od_1000 = 1000`，一次 991 一次 1000 会让两组实验
口径不可比。当前生成结果：**1000/1000，2 轮 shuffle，attempts=1430**，
拒绝原因全部记账（corridor 392 / 原 OD 24 / 重复 13 / s==g 1）。

#### ⑤ 注释里的过期数字

`didi_dataset.py` 头部注释写的是 `2891 节点 / 4408 边`，实际是 **4403 边**
（4408 是没丢自环时的数：6639 road segment − 10 自环 − 2226 平行折叠 = 4403）。
已改正，避免写论文时自己把数字弄混。

#### 附：km-based DTW 已实现（方案第 12.5 节）

坐标齐了之后把 DTW 补上了：

* `real_path_metrics.dtw_distance_km(pred, gt, coordinates, band=None)` ——
  haversine 代价 + 标准 DP，返回 `DtwResult(total_km, mean_km, warping_steps)`。
  **对外一律报 `mean_km`**（总代价/对齐点对数），因为原始 DTW 随路径长度线性增长，
  不比归一化就不能在长短路径之间比较；`band` 是 Sakoe-Chiba 窗口，防退化对齐。
* 接进 `pair_record` / aggregate：`dtw_km`、`dtw_km_success`、`dtw_km_p50`、
  `dtw_num_finite`；`evaluate.py` 与 `Trainer.validate` 都会带上。
* 坐标加载走 `load_node_coordinates_filled()`：原始 OSMnx 图只覆盖 2891 个节点里的
  **2780 个（96.2%）**，剩下 111 个用邻居坐标迭代填充。不补的话，
  **test 集里就会有样本的 DTW 静默变成 NaN** —— 这是实测踩到的。
* 手算样例测试：`test_dtw_handles_different_sampling_density` 里能对上
  `d(p0,g0)+d(p0,g1)+d(p1,g2)+d(p2,g3)`，并断言 DTW 严格优于逐点硬比。

在 test 集上跑 GT-vs-Dijkstra 的 DTW 可以看出它与离散指标互补：

| idx | GT 跳数 | Dijkstra 跳数 | nLCS | Edge F1 | DTW (km) |
|---:|---:|---:|---:|---:|---:|
| 2 | 24 | 24 | 1.000 | 1.000 | 0.0000 |
| 4 | 14 | 11 | 0.571 | 0.522 | 0.1061 |
| 7 | 12 | 12 | 0.167 | 0.000 | 0.5347 |
| 5 | 27 | 34 | 0.444 | 0.339 | 0.3992 |

idx 7 最典型：两条路**跳数完全一样、没有一条边重合**（Edge F1 = 0），但几何上
平均只差 0.53 km —— 这正是 DTW 要补的那块信息。
