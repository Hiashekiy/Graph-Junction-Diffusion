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
| `K = W_K e_uv`（边状态） | `graph_flow.py::k_proj(edge_feat)` |
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

## 5. 使用

```bash
# 1) 生成数据（含逐样本语义校验）
python scripts/generate_dataset.py --config configs/graph_flow.yaml

# 2) 检查数据语义（手工核对 branch / z0 / batch 形状）
python scripts/inspect_dataset.py --data data/er_256_train.pkl --limit 3

# 3) tiny overfit（实施指南第 27 节的强制前置步骤）
python scripts/train.py --config configs/graph_flow.yaml --tiny \
    --set diffusion.T=20 --set training.epochs=200 --name tiny

# 4) 正式训练
python scripts/train.py --config configs/graph_flow.yaml --name graph_flow

# 5) 评测
python scripts/evaluate.py --config configs/graph_flow.yaml \
    --checkpoint outputs/runs/graph_flow/best.pt \
    --data data/er_256_test.pkl --baselines --out outputs/runs/graph_flow/eval.json

# 6) 按难度 / 结构模式 / 决策数拆分测试结果
python tools/breakdown_eval.py outputs/runs/graph_flow/eval.json \
    --data data/er_256_test.pkl

# 7) 两次评测并排比较（逐 bucket 给 A / B / delta；不加载模型，训练中也能跑）
python tools/compare_runs.py \
    --a outputs/runs/v2_controlled_100ep/eval_test.json \
    --b outputs/runs/v2_controlled_100ep_flow3/eval_test.json \
    --data data/controlled_test.pkl --label-a "flow_steps=1" --label-b "flow_steps=3"

# 8) 无 torch 也能跑的静态检查
python tools/semantic_check.py --strict
```

命令行可以覆盖任意配置项（`--set` 可以重复出现）：

```bash
python scripts/train.py --config configs/graph_flow.yaml --tiny \
    --set training.lr=3e-3 --set training.epochs=300 \
    --set diffusion.T=10 --set model.d_model=32
```

> **注意**：配置里的 `training.lr=1e-4` 是给正式规模准备的。tiny overfit 验证链路的
> 时候用 `3e-3` 量级收敛快得多（见第 8 节实测）。

## 6. 测试

```bash
python -m pytest tests -q
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
| `test_checkpoint_mismatch.py` | checkpoint 与模型结构不匹配时的可读报错（flow_steps 必须一致） |

## 7. 评测指标

主指标（模型选择用）：

- **Goal Hit Rate** = 到达 goal 的 query 数 / 总 query 数
- **Optimal Path Rate** = 成功且路径 cost 等于 GT 最短路的比例
- **Success Cost Ratio** = `L_pred / L_optimal`，**只在成功样本上统计**
- **Loop Rate / Broken Rate / Inference Time**

`decision accuracy` 只是 debug 指标，**不能用来选模型**（它和端到端表现严重脱节）。
注意它还分两类：NULL 决策占了 80% 以上，所以整体 accuracy 会看起来很漂亮 ——
真正决定路径的是**非 NULL（active）决策**的准确率。

## 8. 已验证的落地状态

在一台有 torch 2.9 / CUDA 的机器上实跑过（数据规模很小，仅用于验证链路）：

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `python -m pytest tests -q` | **129 passed** |
| 数据生成 | `python scripts/generate_dataset.py ...` | 通过，含逐样本语义校验 |
| tiny overfit | `python scripts/train.py --tiny --set training.lr=3e-3 ...` | 见下 |
| 完整链路 | `python scripts/evaluate.py --checkpoint .../best.pt --data ...` | goal_hit **1.0000** |

tiny overfit（16 个固定 query、T=10、d_model=32，300 epochs）的真实轨迹：

```text
epoch 100: train_loss=0.176  train_x0_acc=1.000  goal_hit=0.250
epoch 200: train_loss=0.120  train_x0_acc=0.978  goal_hit=0.875
epoch 300: train_loss=0.047  train_x0_acc=1.000  goal_hit=1.000  loop=0, broken=0
```

也就是说：**loss 下降 → teacher-forced 单步学会 → 完整 reverse chain 的 Goal Hit 上升**
这条链路是通的。注意默认 `training.lr=1e-4` 在 tiny 集上偏小，验链路时可以先调大。

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

验收（GGMPC 环境，torch 2.9.1+cu126）：`pytest` 171 passed、
`semantic_check` 0 problems、tiny overfit 的 full-chain `goal_hit=1.0000`、
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
python scripts/generate_dataset.py --config configs/graph_flow.yaml
```

输出 `data/controlled_{train,val,test}.pkl` + `data/controlled_summary.json`，后者包含
指南第 16 节要求的全部指标（hops / decisions / branch factor 的 mean-std-min-max、
NULL 比例、source-as-decision 比例、三类干扰分支占比、难度与模式配比）。

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
3000 条；attempts/accepted 与 `data/controlled_summary.json` 完全一致）：

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
python scripts/train.py --config configs/graph_flow.yaml --name v2_controlled_100ep \
    --data data/controlled_train.pkl --val-data data/controlled_val.pkl
```

第一版正式配置：`num_samples=3000`、`batch_size=48`、`T=50`、`d_model=128`、`amp=true`、
`epochs=100`。实测约 **43 s/epoch**（50 步 × ~0.86 s，RTX 4070），100 epoch ≈ **72 分钟**。

评测：

```bash
python scripts/evaluate.py --config configs/graph_flow.yaml \
    --checkpoint outputs/runs/v2_controlled_100ep/best.pt \
    --data data/controlled_test.pkl --baselines \
    --out outputs/runs/v2_controlled_100ep/eval_test.json
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
  python scripts/evaluate.py \
      --config outputs/runs/v2_controlled_100ep/run_config.json \
      --checkpoint outputs/runs/v2_controlled_100ep/best.pt \
      --data data/controlled_test.pkl --baselines \
      --out outputs/runs/v2_controlled_100ep/eval_test.json
  ```

### 推理时的提前退出（"训练 3 轮、推理只跑 1 轮"）

多轮交流的代价在推理时是线性的（实测 flow_steps=3 约 0.06 s/query，1 轮约 0.03 s/query）。
"训练时见识过多轮、推理时少跑几轮"完全合法（第 k 轮用的 slot embedding 就是训练时
那一行），所以留了一个口子做"多少轮才够"的 ablation：

```bash
python scripts/evaluate.py \
    --config outputs/runs/v2_controlled_100ep_flow3/run_config.json \
    --checkpoint outputs/runs/v2_controlled_100ep_flow3/best.pt \
    --data data/controlled_test.pkl --eval-flow-steps 1 \
    --out outputs/runs/v2_controlled_100ep_flow3/eval_test_flow1.json
```

约束（`GraphFlowDenoiser.set_inference_flow_steps`）：只能**减少**轮数，超过训练时的
round 数会直接 `ValueError`（slot embedding 没有那么多行）；评测结果 json 里会写上
`inference.flow_steps` 与 `inference.trained_flow_steps`，避免事后分不清这是几次前向的结果。
