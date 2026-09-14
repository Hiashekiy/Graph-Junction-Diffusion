# `data/` 数据集索引

数据集**不进版本库**（`.gitignore` 忽略 `data/*`，只保留本文件）。这里按来源分成 5 组，
文件名一律**不要改**：`outputs/runs/**/eval*.json`、`mp_*.json` 都是按**文件名**记录数据集
的，dashboard 也拿文件名当数据集 id；改名会让历史指标和数据集对不上号。

```
data/
├── README.md            本文件（唯一进版本库的东西）
├── controlled/          主数据集：controlled_junction 生成器（骨架 + 干扰分支）
├── long/                长链专测集 / 长链候选池 / 长链混合训练集
├── oldv1/               由 V1 旧数据转换来的 V2 数据集
├── mixed/               当前最新模型（v2_rev2_mixed）用的混合训练集
├── smoke/               冒烟小数据集，只用来快速跑通流程
├── public_graphs/       公开图数据（DIMACS9/10、SNAP、CLRS30），由 scripts/download_graph_datasets.py 下载
└── processed/v1/        V1 时代的原始 .pt …… 本机工作副本里没有保留
```

规模与语义的完整说明见仓库根 `README.md` 第 0.1 节；本文件只做"目录导航 + 怎么重新生成"。

## controlled/ —— 主数据集

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `controlled_train.pkl` | 12 MB | 2400 条 / 6.53 决策 / 34.7 候选 | 主训练集 |
| `controlled_val.pkl` | 1.6 MB | 300 条 / 6.55 决策 | 主验证集 |
| `controlled_test.pkl` | 1.6 MB | 300 条 / 6.49 决策 | **标准测试集**：所有 run 都在它上面报 test 指标 |
| `controlled_summary.json` | 4 KB | 3000 条 | 上面三份的生成统计 |

```bash
# 默认输出目录 = 配置里的 paths.data_dir = data/controlled
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml
```

## long/ —— 长链（决策数 ≥ 9）

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `controlled_long.pkl` | 2.8 MB | 400 条 / 9.20 决策 / 26.0 跳 | 长链专测集（全 hard） |
| `controlled_longpool.pkl` | 7.1 MB | 1050 条 / 9.22 决策 | 长链候选池，用来给训练集补长链样本 |
| `controlled_longmix_train.pkl` | 19 MB | 3300 条 / 7.26 决策（≥9 占 30.9%） | `v2_rev2_longmix` 的训练集 |
| `controlled_longmix_val.pkl` | 2.5 MB | 450 条 / 7.44 决策 | 对应的验证集 |
| `*_summary.json` | 1–4 KB | — | 上面各份的统计 |

```bash
# 长链专测集 / 池子：--min-decisions 9 且不划分 split
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml --data-dir data/long --name controlled_long --no-split --min-decisions 9 --set data.num_samples=400 --set seed=7 --set data.difficulty_mix.hard=1.0 --set data.difficulty_mix.easy=0.0 --set data.difficulty_mix.medium=0.0
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow.yaml --data-dir data/long --name controlled_longpool --no-split --min-decisions 9 --set data.num_samples=1050 --set seed=11

# 长链混合集：controlled_{train,val} + 池子里的 900 / 150 条
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/long/controlled_longmix_train.pkl --input data/controlled/controlled_train.pkl --input data/long/controlled_longpool.pkl --limit 900:1 --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/long/controlled_longmix_val.pkl --input data/controlled/controlled_val.pkl --input data/long/controlled_longpool.pkl --skip 900:1 --seed 0
```

## oldv1/ —— V1 旧数据转成 V2 格式

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `oldv1_train_sub.pkl` | 36 MB | 1970 条 / 37.9 决策 / 275 候选 | V1 train 抽 2000 条转换，混入训练集 |
| `oldv1_val_sub.pkl` | 6.8 MB | 390 条 / 36.5 决策 | V1 val 抽 400 条转换，混入验证集 |
| `oldv1_test.pkl` | 42 MB | 2444 条 / 38.2 决策 / 268 候选 | **跨分布测试集** |

```bash
# 需要先有 V1 原始 .pt（见下面的 "processed/v1"）
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/test.pt  --out data/oldv1/oldv1_test.pkl
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/train.pt --out data/oldv1/oldv1_train_sub.pkl --sample 2000 --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/convert_v1_dataset.py --input data/processed/v1/val.pt   --out data/oldv1/oldv1_val_sub.pkl  --sample 400  --seed 0
```

> V1 的候选是"下一跳的边"，V2 是"走到下一个 structural endpoint 的 branch segment"，
> 所以必须转换；约 2.2% 的 query 因为"两个 degree=2 节点构成的三角"无法用 V2 语义表示而被跳过。

## mixed/ —— 当前最新模型的混合训练集

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `mixed_oldv1_train.pkl` | 55 MB | 5270 条 / 18.7 决策 / 127 候选 | `v2_rev2_mixed` 的训练集 |
| `mixed_oldv1_val.pkl` | 9.3 MB | 840 条 / 20.9 决策 | 对应的验证集（模型选择用它） |
| `mixed_oldv1_{train,val}_summary.json` | 1–4 KB | — | 合并统计（组成、决策数直方图、长链占比） |

```bash
# longmix + oldv1 合并（只动 train/val，不合并 test）
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_train.pkl --input data/long/controlled_longmix_train.pkl --input data/oldv1/oldv1_train_sub.pkl --seed 0
E:/CondaEnvData/envs/GGMPC/python.exe tools/merge_datasets.py --out data/mixed/mixed_oldv1_val.pkl   --input data/long/controlled_longmix_val.pkl   --input data/oldv1/oldv1_val_sub.pkl   --seed 0

# 训前必须过的图级泄漏检查（要求 0 重叠）
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --pair data/mixed/mixed_oldv1_train.pkl data/mixed/mixed_oldv1_val.pkl --pair data/mixed/mixed_oldv1_train.pkl data/oldv1/oldv1_test.pkl --pair data/mixed/mixed_oldv1_train.pkl data/controlled/controlled_test.pkl
```

## weighted_controlled/ —— 带权图（Weighted 扩展，README 第 20 节）

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `weighted_controlled_train.pkl` | 12 MB | 2400 条 | 加权主训练集 |
| `weighted_controlled_val.pkl` | 1.6 MB | 300 条 | 加权验证集（模型选择用它） |
| `weighted_controlled_test.pkl` | 1.6 MB | 300 条 | 加权标准测试集 |
| `weighted_controlled_summary.json` | 4 KB | 3000 条 | 生成统计 + weighted sanity check |

```bash
# 3000 条，w ~ U(1,10)，GT = Dijkstra 最小 cost 路径（约 50 秒）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/graph_flow_weighted.yaml --name weighted_controlled --data-dir data/weighted_controlled
```

实测（`weighted_controlled_summary.json` 的 `weighted` 节）：`weighted_conflict_rate` 0.460、
`bfs_cost_ratio` 1.034、`gt_path_is_weighted_optimal_fraction` 1.0，权重 5.49 ± 2.60（1.00–10.00）。
拓扑难度契约（hops ∈ [15,35]、decisions ∈ [5,12]、branch factor ∈ [2,5]）与无权数据集完全一致 ——
赋权只改变"哪条路最优"，不改变拓扑验收。

## smoke/ —— 冒烟小数据集

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `smoke_train.pkl` | 176 KB | 26 条 | 流程自检 |
| `smoke_val.pkl` | 28 KB | 3 条 | 同上 |
| `smoke_test.pkl` | 24 KB | 3 条 | 同上 |
| `smoke_summary.json` | 4 KB | — | 三个 split 的统计 |

## public_graphs/ —— 公开图数据（下载而来）

由 `scripts/download_graph_datasets.py` 下载（只用标准库，零第三方依赖）。默认 profile
`recommended` 会把文件放到 `<output>/<group>/<原始文件名>`，并写一份 `DATASETS.txt` 清单
（含每份的 URL 与来源页）。当前已下载：

| 文件 | 大小 | 规模 | 说明 |
|---|---|---|---|
| `dimacs9/rome99.gr` | 0.13 MB | 3353 节点 / 8870 弧 | DIMACS9 罗马有向路网，未压缩文本（`p sp 3353 8870`），**无坐标** |
| `dimacs9/USA-road-d.NY.gr.gz` | 3.5 MB | 264346 节点 / 733846 弧 | DIMACS9 纽约，弧长 = 旅行时间 |
| `dimacs9/USA-road-d.NY.co.gz` | 2.0 MB | 264346 行 | 上面那份的节点经纬度（`v id lon lat`，1e-6 度） |
| `dimacs9/USA-road-d.BAY.gr.gz` | 3.9 MB | 321270 节点 / 800172 弧 | DIMACS9 旧金山湾区 |
| `dimacs9/USA-road-d.BAY.co.gz` | 2.5 MB | 321270 行 | 湾区节点经纬度 |
| `dimacs9/USA-road-d.COL.gr.gz` | 5.5 MB | 435666 节点 / 1057066 弧 | DIMACS9 科罗拉多 |
| `dimacs9/USA-road-d.COL.co.gz` | 3.8 MB | 435666 行 | 科罗拉多节点经纬度 |
| `dimacs10/luxembourg.osm.graph.bz2` | 0.5 MB | 114599 节点 / 119666 边 | DIMACS10 卢森堡 OSM 街道图（**邻接表**格式，无向无权） |
| `dimacs10/luxembourg.osm.xyz.bz2` | 0.5 MB | 114599 行 | 上面那份的节点坐标（`x y z`，UTM 米） |
| `DATASETS.txt` | 1 KB | — | 上述各份的 URL / 来源页 / 描述 |

```bash
# 默认 recommended = DIMACS9 的 5 份（约 14 MB）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py

# 关键：DIMACS9 的 .gr 只有弧长，没有坐标画不出地图；--with-coords 会补下 *.co.gz / *.xyz.bz2
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --with-coords

# 看看有哪些可选 / 下载后顺手解压 / 换输出目录
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --list
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --decompress --remove-archives
E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --output data/public_graphs

# 更大的组：SNAP roadNet-CA/PA/TX（--profile snap）、再加 CLRS30（--profile all）、
# 或全美 unit-cost 图（--datasets dimacs9_usa_unit，约 225 MB 压缩）
```

**画出来看**（全貌 + 最热闹路口的局部放大 + 度数/边权分布）：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py
E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py --only rome99 luxembourg --hops 4
```

输出 `outputs/figures/public_graphs_map.png`（每份数据一行：左全图、右 3 跳放大）与
`outputs/figures/public_graphs_stats.png`（度数分布、边权分布、规模统计表）。

> 两种文本格式：DIMACS9 是 `c` 注释 + `p sp N M` + `a u v w`（弧），坐标另存于 `*.co`；
> DIMACS10 streets 归档是 `%` 注释 + `N M` + **N 行邻接表**，坐标另存于 `*.xyz`。
> 重跑同一命令会跳过已下载的文件，中断后直接再跑即可。

## processed/v1/ —— V1 原始数据（本机未保留）

`data/processed/v1/{train,val,test,ood_size}.pt` + `manifest.json` / `*_meta.json` 是 V1 时代
（原 `data_old/`）的预处理数据，5000/500/500/300 张图。**当前工作副本里没有这几个文件**，
要重跑上面的 `convert_v1_dataset.py` 得先从归档恢复。

## 快速自检

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/inspect_dataset.py --data data/controlled/controlled_train.pkl --limit 3
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
```
