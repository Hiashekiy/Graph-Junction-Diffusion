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
| 一步 reverse step 只跑一次 Flow | `denoiser.py::step` 只调一次 `graph_flow` |
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

# 6) 无 torch 也能跑的静态检查
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
