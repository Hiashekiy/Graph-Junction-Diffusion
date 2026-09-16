# `data/` 数据集索引

数据集**不进版本库**（`.gitignore` 忽略 `data/*`，只保留本文件）。文件名一律**不要改**：
`outputs/runs/**/eval*.json`、`mp_*.json` 都是按**文件名**记录数据集的，dashboard 也拿文件名
当数据集 id；改名会让历史指标和数据集对不上号。

```
data/
├── README.md            本文件（唯一进版本库的东西）
├── unweighted/          唯一的**无权重**合成数据集（train/val/test，按图重新划分）
├── weighted/            唯一的**带权**合成数据集（train/val/test）
├── didi/                所有 DiDi 真实道路数据（2026-09-16 归拢到这一个父目录）
│   ├── raw/             原始数据：轨迹 CSV + dicts / edge_features / line_graph / 坐标图
│   │   ├── chengdu/       ← 管线依赖（config 的 data.root / data.coords_file 指这里）
│   │   └── xian/          ← 当前不用，跨城市实验备用
│   └── graph/           处理好的图数据集（Branch Segment + decision field + split）
│       └── chengdu/       ← config 的 paths.data_dir 指这里（README 第 24 节）
└── processed/v1/        V1 时代的原始 .pt …… 本机工作副本里没有保留
```

> **2026-09-16 大清理 + 归拢。** 已删除：
>
> * `controlled/`、`long/`、`oldv1/`、`smoke/`（131 MB）—— 它们的训练数据全部包含在
>   `unweighted/` 里；`unweighted/` 已用 `tools/resplit_dataset.py` **按图重新划分**成 train/val/test
>   （原来是"合并时焊死"的 train/val，没有 test）。重划分的硬要求与验证见仓库根 README §0.1 A。
> * `public_graphs/`（DIMACS9/10）与配套的 `scripts/download_graph_datasets.py`、
>   `tools/visualize_public_graphs.py`、`tools/dimacs9_external_eval.py` —— 该方向放弃，
>   且唯一的结论是负面的（受控路口图的 checkpoint 零样本迁移不到真实路网）。
>
> ⚠️ **这四个目录不可再生**：`oldv1` 的上游 `data/processed/v1/*.pt` 本来就不在仓库里。
> 本文档下方仍保留了它们当年的生成命令，作为历史记录 —— **那些命令现在跑不了**。
>
> 另外把三份 DiDi 道路数据从 `data/` 顶层归拢进了 `data/didi/`，并分成 `raw/`（原始）
> 与 `graph/`（处理好的图数据集）两类。原下载包 `data/DiDiChengduXian/` 已删除 ——
> 里面的成都/西安核心文件已提取到 `raw/`，其余是另两个城市（`pt`）和本项目从不使用的
> 预计算距离矩阵。

规模与语义的完整说明见仓库根 `README.md` 第 0.1 节；本文件只做"目录导航 + 怎么重新生成"。

## mixed/ —— 唯一的无权重数据集（2026-09-16 按图重新划分）

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `unweighted_train.pkl` | 51 MB | 4905 条 / 4564 图 | 训练集 |
| `unweighted_val.pkl` | 6.3 MB | 601 条 / 570 图 | 验证集（模型选择用它） |
| `unweighted_test.pkl` | 6.3 MB | 604 条 / 570 图 | **标准测试集** |
| `unweighted_summary.json` | 2 KB | — | 划分统计（组成占比、决策数/候选数、图数） |

```bash
# 重新划分（按 graph_id 切 + graph_id 全局重编号 + 分层；自带 graph_id 交集的泄漏自检）
E:/CondaEnvData/envs/GGMPC/python.exe tools/resplit_dataset.py ^
  --input <旧 train>.pkl --input <旧 val>.pkl ^
  --out-dir data/unweighted --name unweighted --seed 0

# 独立的指纹核对（sha1 over 图结构，比 graph_id 更硬；要求 0 重叠）
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --all ^
  data/unweighted/unweighted_train.pkl data/unweighted/unweighted_val.pkl data/unweighted/unweighted_test.pkl
```

实测：train/val/test 两两重叠图数 **0**；四个来源在三个 split 里的占比一致
（`controlled_longmix` 61.2/62.4/62.1%，`oldv1` 38.8/37.6/37.9%）。
样本 `meta['source_file']` 仍写着 `controlled_longmix_train.pkl` 之类的**原始出处**，那是血缘记录。

## weighted_controlled/ —— 带权图（Weighted 扩展，README 第 20 节）

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `weighted_train.pkl` | 12 MB | 2400 条 | 加权主训练集 |
| `weighted_val.pkl` | 1.6 MB | 300 条 | 加权验证集（模型选择用它） |
| `weighted_test.pkl` | 1.6 MB | 300 条 | 加权标准测试集 |
| `weighted_summary.json` | 4 KB | 3000 条 | 生成统计 + weighted sanity check |

```bash
# 3000 条，w ~ U(1,10)，GT = Dijkstra 最小 cost 路径（约 50 秒）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/generate_dataset.py --config configs/controlled_weighted.yaml --name weighted_controlled --data-dir data/weighted
```

实测（`weighted_summary.json` 的 `weighted` 节）：`weighted_conflict_rate` 0.460、
`bfs_cost_ratio` 1.034、`gt_path_is_weighted_optimal_fraction` 1.0，权重 5.49 ± 2.60（1.00–10.00）。
拓扑难度契约（hops ∈ [15,35]、decisions ∈ [5,12]、branch factor ∈ [2,5]）与无权数据集完全一致 ——
赋权只改变"哪条路最优"，不改变拓扑验收。

## didi/graph/chengdu/ —— DiDi 成都真实道路数据（README 第 24 节）

来源：`data/didi/raw/chengdu/`（10 天轨迹 CSV + `dicts.pkl` + `edge_features.csv`）。**GT 是真实车辆历史路径，不是最短路**：93.7%
的 GT 都不是 Dijkstra 最优，cost ratio 中位数 1.12。

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `graph_global.pkl` | 227 KB | 2891 节点 / 4403 边 | 折叠后的全局无向有权成都路网 |
| `train.pkl` / `val.pkl` | 519 MB / 62 MB | 4687 / 533 条 | 训练 / 模型选择 |
| `test.pkl` | 149 MB | 1275 条 | 完整 test 指标 |
| `test_1000.pkl` | 117 MB | 1000 条 | GDP 风格固定子集（论文主表） |
| `shuffled_od_1000.pkl` | 95 MB | 741 条 | OD 打乱重配，**无真实 GT** |
| `split_manifest.csv` | 904 KB | 6495 行 | 逐样本 order_id / split / gt_cost / gt_cost_ratio |
| `metadata.json` | 2 KB | — | 建图统计、rho、过滤条件、划分 |
| `stats.json` | 9 KB | — | 清洗漏斗 + corridor retention + 各 split 摘要 |
| `scan_only.json` / `scan_corridor.json` | 3 KB / 3 KB | — | 阶段 0 / 阶段 1 的扫描产物 |
| `_didi_candidates.pkl` | 10 MB | 36610 条 | 候选缓存（换配置自动失效） |

规模偏大是因为真实 corridor 比 synthetic 大两个量级：一个 ~350 decision 的样本
pickle 之后约 110 KB，`segments` + `field` 占大头。缩小的办法是调小
`data.max_dataset_samples` 或 `data.corridor.rho`（见 README 第 24.4 / 24.5 节）。

```bash
# 三个阶段（约 4 分钟）+ 自检（不需要 torch）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/didi_chengdu.yaml --scan-only
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/didi_chengdu.yaml --scan-corridor
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/didi_chengdu.yaml --build
E:/CondaEnvData/envs/GGMPC/python.exe tools/verify_didi_pipeline.py --data data/didi/graph/chengdu
```

## didi/graph/chengdu_long/ —— 同一张路网的**长轨迹 / 大图**泛化测试集（README 第 24.15 节）

和上面 `chengdu/` **同一张路网、同一份原始 CSV、同一套流程**（`graph_global.pkl` 逐位相同，
md5 `c5eed820…`），只把轨迹长度窗口从 `(10,100)` 搬到 `(59,200)`：每一条样本的
`gt_length ≥ 60`，是主数据集中位长度（22）的 2.7 倍起。用途是**规模泛化**，不是替代主数据集。

| 文件 | 大小 | 规模 | 用途 |
|---|---|---|---|
| `graph_global.pkl` | 227 KB | 2891 节点 / 4403 边 | 折叠后的全局无向有权成都路网（与 `chengdu/` 相同） |
| `train.pkl` / `val.pkl` | 970 MB / 178 MB | 1693 / 314 条 | 在长图上微调（可选） |
| `test.pkl` | 638 MB | 1103 条 | 完整 test 指标 |
| `test_1000.pkl` | 581 MB | 1000 条 | **规模泛化主表** |
| `shuffled_od_1000.pkl` | 347 MB | 1000 条 | OD 打乱重配，**无真实 GT**（`dijkstra_placeholder`） |
| `split_manifest.csv` | 463 KB | 3110 行 | 逐样本 order_id / split / gt_cost / gt_cost_ratio |
| `metadata.json` | 2 KB | — | 建图统计、rho、过滤条件、划分 |
| `stats.json` | 17 KB | — | 清洗漏斗 + corridor retention + 各 split 摘要 |
| `_didi_candidates.pkl` | 2 MB | 4039 条 | 候选缓存（换配置自动失效） |

规模对照（`test_1000`）：corridor 节点 1909 vs 412（**4.63×**）、gt_length 65.3 vs 22.7
（**2.88×**）、decision 1631 vs 336（4.85×）、单样本 567 KB vs 114 KB（4.98×）。

```bash
# 只跑 --build：rho 沿用主数据集冻结的 1.5，不重扫（重扫=用新数据调参，会污染泛化结论）
# 全量读 150 万行，约 16 分钟，产出 2.72 GB
E:/CondaEnvData/envs/GGMPC/python.exe scripts/prepare_didi.py --config configs/didi_chengdu_long.yaml --build
E:/CondaEnvData/envs/GGMPC/python.exe tools/verify_didi_pipeline.py --data data/didi/graph/chengdu_long

# 拿主数据集的模型直接测新规模（--config 保持主数据集，只换 --data）
E:/CondaEnvData/envs/GGMPC/python.exe scripts/evaluate.py --checkpoint outputs/runs/didi_chengdu/best.pt --config configs/didi_chengdu.yaml --data data/didi/graph/chengdu_long/test_1000.pkl
```

⚠️ 两点解读前提：① corridor 在这里**不裁剪**（`max_corridor_nodes: 0`），最大样本 = 2,886 节点，
接近整城 2,891，走廊的局部性已失效，它是压力测试而不是 corridor 语义的复现；② retention 0.80~0.82，
且和主数据集一样存在高绕路 GT 更容易被丢的偏差（1.50~2.00 桶 0.712、≥2.00 桶 0.128），
跨规模比较时两边都不是无偏采样。详见 README 第 24.15 节。

## processed/v1/ —— V1 原始数据（本机未保留）

`data/processed/v1/{train,val,test,ood_size}.pt` + `manifest.json` / `*_meta.json` 是 V1 时代
（原 `data_old/`）的预处理数据，5000/500/500/300 张图。**当前工作副本里没有这几个文件**，
要重跑 V1→V2 转换（`tools/convert_v1_dataset.py`，**已于 2026-09-16 删除**）
得先从归档恢复 V1 的 `.pt`，并重新实现/找回那个转换脚本。
现有的 `data/unweighted/` 里那 38.8% 的 V1 样本是当年转换的产物。

## 快速自检

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/inspect_dataset.py --data data/unweighted/unweighted_train.pkl --limit 3
E:/CondaEnvData/envs/GGMPC/python.exe tools/check_leakage.py --all data/unweighted/unweighted_train.pkl data/unweighted/unweighted_val.pkl data/unweighted/unweighted_test.pkl
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests -q
```
