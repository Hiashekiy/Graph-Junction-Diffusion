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

# 8) 两次训练的曲线按**真实 epoch** 对齐比较（train loss / 验证 goal hit）
python tools/compare_curves.py \
    --a outputs/runs/v2_controlled_100ep \
    --b outputs/runs/v2_controlled_100ep_flow3 \
    --label-a "flow_steps=1" --label-b "flow_steps=3" --every 5

# 9) 把预测路径和 GT 路径画在一起（直观看效果，见第 15 节）
python tools/visualize_paths.py \
    --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --num 8 --labels \
    --out outputs/figures/paths_flow3_test.png

# 10) 只要路径本身（文本 / JSON，不画图）
python tools/predict_path.py \
    --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 170

# 11) 无 torch 也能跑的静态检查
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
| 单元测试 | `python -m pytest tests -q` | **215 passed**（含 flow_steps / 分桶口径 / checkpoint 不匹配 / epoch 校准） |
| 静态检查 | `python tools/semantic_check.py --strict` | checked 48 modules, **0 problems** |
| 数据生成 | `python scripts/generate_dataset.py ...` | 通过，含逐样本语义校验；同 seed 重跑 attempts/accepted 逐位一致 |
| tiny overfit | `python tools/smoke_tiny_overfit.py 200 4` | 4/4 goal、optimal **1.0000**、broken 0 |
| 完整链路 | `python scripts/evaluate.py --checkpoint .../best.pt --data ...` | goal_hit **0.8933**（300 条 test、flow_steps=1 基线） |

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
python scripts/generate_dataset.py --config configs/graph_flow.yaml
```

输出 `data/controlled_{train,val,test}.pkl` + `data/controlled_summary.json`，后者包含
指南第 16 节要求的全部指标（hops / decisions / branch factor 的 mean-std-min-max、NULL 比例、source-as-decision 比例、三类干扰分支占比、难度与模式配比）。

两个为**专项评测集**加的参数（见第 16 节的长链实验）：

```bash
# 只保留 GT 决策数 >= 9 的样本，且不划分 train/val/test（整份存成一个文件）
python scripts/generate_dataset.py --config configs/graph_flow.yaml \
    --name controlled_long --no-split --min-decisions 9 \
    --set data.num_samples=400 --set seed=7 \
    --set data.difficulty_mix.hard=1.0 \
    --set data.difficulty_mix.easy=0.0 --set data.difficulty_mix.medium=0.0
```

`--min-decisions N` 走 `build_controlled_dataset(min_decisions=N)`：难度过滤通过但决策数
不够的样本会被丢掉，并在 summary 的 `filter.rejected_by_min_decisions` 里记数（长链集的
接受率只有 0.39%，这个数说明"长链样本为什么稀少"）。`--no-split` 只写
`data/<name>.pkl` + `_summary.json`，不碰现有的 train/val/test。

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

还有一个专门的诊断工具，直接量"多轮有没有做事"：

```bash
python tools/round_diagnostic.py outputs/runs/v2_controlled_100ep_flow3 --samples 8
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
  python scripts/evaluate.py \
      --config outputs/runs/v2_controlled_100ep/run_config.json \
      --checkpoint outputs/runs/v2_controlled_100ep/best.pt \
      --data data/controlled_test.pkl --baselines \
      --out outputs/runs/v2_controlled_100ep/eval_test.json
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
python scripts/evaluate.py \
    --config outputs/runs/v2_controlled_100ep_flow3/run_config.json \
    --checkpoint outputs/runs/v2_controlled_100ep_flow3/best.pt \
    --data data/controlled_test.pkl --eval-flow-steps 1 \
    --out outputs/runs/v2_controlled_100ep_flow3/eval_test_flow1.json
```

约束（`GraphFlowDenoiser.set_inference_flow_steps`）：只能**减少**轮数，超过训练时的
round 数会直接 `ValueError`（slot embedding 没有那么多行）；评测结果 json 里会写上
`inference.flow_steps` 与 `inference.trained_flow_steps`，避免事后分不清这是几次前向的结果。

## 15. 结果可视化（预测路径 vs GT 路径）

两个工具，分工不同：

| 想干什么 | 用哪个 |
|---|---|
| 看图（预测路径 vs GT 画在一起） | `tools/visualize_paths.py` |
| 要路径本身（节点序列、结局、分岔点、机器可读 JSON） | `tools/predict_path.py` |

```bash
# ---- 1) 只要路径本身（文本，不画图）------------------------------------------
# 单条 query：--index 就是 dataset[i] 的编号，也是画图标题里的 #170
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 170

# 多条（逗号分隔或重复 --index）+ 导出机器可读 json
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 0,47,170 --out-json paths.json

# 换采样种子 / 要确定性输出（推理是随机的，见下文）
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 170 --seed 3
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 170 --deterministic

# 同一个 checkpoint 只跑 1 轮图信息交流（推理轮数 ablation）
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --index 170 --flow-steps 1

# 换另一个 run（单轮基线）
python tools/predict_path.py --run outputs/runs/v2_controlled_100ep \
    --data data/controlled_test.pkl --index 47

# ---- 2) 画图（预测路径 vs GT 路径）------------------------------------------
# 自动抽样 8 条：覆盖 easy/medium/hard × 到达/断掉
python tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --num 8 --cols 2 --labels \
    --out outputs/figures/paths_flow3_test.png

# 只看指定几条（便于复现同一组图，或与别的 run 画同一批 query 做对照）
python tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --select indices --indices 170,47,79,140 \
    --cols 2 --labels --out outputs/figures/paths_pick4.png

# 只看 hard 难度
python tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --num 6 --only-difficulty hard --cols 2 --labels \
    --out outputs/figures/paths_hard.png

# 换布局（默认 spring，见下表）
python tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --num 4 --layout kamada_kawai \
    --out outputs/figures/paths_kk.png
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
python tools/visualize_paths.py --run outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_test.pkl --num 8 --cols 2 --select random \
    --out outputs/figures/paths_random.png

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
dataset = GraphQueryDataset.load("data/controlled_test.pkl")

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
python scripts/generate_dataset.py --config configs/graph_flow.yaml \
    --name controlled_long --no-split --min-decisions 9 \
    --set data.num_samples=400 --set seed=7 \
    --set data.difficulty_mix.hard=1.0 \
    --set data.difficulty_mix.easy=0.0 --set data.difficulty_mix.medium=0.0
```

生成结果（`data/controlled_long.pkl` + `_summary.json`）：**400 条 query / 400 张图**、
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
python tools/multiseed_eval.py \
    --a outputs/runs/v2_controlled_100ep \
    --b outputs/runs/v2_controlled_100ep_flow3 \
    --data data/controlled_long.pkl --seeds 0,1,2,3,4 \
    --metric goal_hit --bucket-data data/controlled_long.pkl \
    --out outputs/runs/v2_controlled_100ep_flow3/multiseed_long_goal_hit.json
```

### 长链偏重的训练集已备好（**只造数据，尚未训练**）

训练分布的问题已经量化：现有训练集里 ≥9 决策的样本只有 **119/2400 = 5.0%**
（6–8 决策占 82%），所以 82% 的算力花在中短链上。按"让长链占 30% 左右"补了数据：

```bash
# 1) 先另外造一批长链样本（不要动现有 train/val/test）
python scripts/generate_dataset.py --config configs/graph_flow.yaml \
    --name controlled_longpool --no-split --min-decisions 9 \
    --set data.num_samples=1050 --set seed=11
# 实测：1050 条 / 1050 图，决策数 9–10（均值 9.22）、hops 16–33（均值 22.0），
#       201,352 次 attempt 才凑出来（接受率 0.52%），耗时 ~10 分钟

# 2) 与现有训练/验证集合并（graph_id 全局重编号 + 固定 seed 打乱）
python tools/merge_datasets.py --out data/controlled_longmix_train.pkl \
    --input data/controlled_train.pkl --input data/controlled_longpool.pkl \
    --limit 900:1 --seed 0
python tools/merge_datasets.py --out data/controlled_longmix_val.pkl \
    --input data/controlled_val.pkl --input data/controlled_longpool.pkl \
    --skip 900:1 --seed 0
```

| 文件 | queries / graphs | 决策数直方图 | ≥9 占比 |
|---|---|---|---|
| `data/controlled_longmix_train.pkl` | 3300 / 3300 | 5:310, 6:1037, 7:669, 8:265, **9:807, 10:212** | **30.9%** |
| `data/controlled_longmix_val.pkl` | 450 / 450 | 5:38, 6:131, 7:85, 8:28, **9:128, 10:40** | **37.3%** |
| （原）`data/controlled_train.pkl` | 2400 / 2400 | 5:310, 6:1037, 7:669, 8:265, 9:101, 10:18 | 5.0% |

**无泄漏**（按图结构指纹即排序边集合的哈希核对）：longmix_train ↔ longmix_val = 0、
longmix_train ↔ controlled_test = 0、longmix_train ↔ controlled_long = 0，
longmix_val ↔ 两个 test 也都是 0。原来三个 split 与两个 test 集**一个字节都没动**，
所以前面所有结论继续有效。

**将来要（用户批准后）训练时**：

```bash
python scripts/train.py --config configs/graph_flow.yaml \
    --name v2_longmix_100ep \
    --data data/controlled_longmix_train.pkl \
    --val-data data/controlled_longmix_val.pkl
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
python tools/multiseed_eval.py --a outputs/runs/v2_controlled_100ep \
    --b outputs/runs/v2_controlled_100ep_flow3 --seeds 0,1,2,3,4 --metric goal_hit
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
python -m pytest tests -q        # 246 passed
python tools/semantic_check.py --strict   # checked 59 modules, 0 problems
```
