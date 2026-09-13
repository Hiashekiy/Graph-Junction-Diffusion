# ARCHITECTURE V2 — Edge-State Conditioned Recurrent Graph Flow Denoiser

本文只描述**代码里实际存在**的结构，逐条对应设计报告 V2.1。凡是"第一版不做"的
东西都在文末列出，避免后来者误以为漏了实现。

---

## 1. 两个同步演化的状态

```
(H_t, z_t) -> (H_{t-1}, z_{t-1})
```

- `H_t`：persistent node state，`[N, d]`，**跨 reverse step 直接传递、不重置**；
- `z_t`：每个 decision node 的 categorical branch state，`[M]`，用 **flat candidate
  index** 表示。

代码里的强制机制：`GraphFlowDenoiser.step(batch, H_t, z_t, t) -> DenoiserOutput`，
`DenoiserOutput.H_next` 就是下一步的 `H_t`。H 不作为 module 内部 mutable state 存在
（`tests/test_persistent_state.py` 会检查这一点）。

## 2. 张量约定

| 张量 | 形状 | 说明 |
|---|---:|---|
| `node_type` | `[N]` | 0/1/2/3 |
| `edge_index` | `[2, E_msg]` | 有向消息边（无向边展开成两个方向） |
| `msg_to_phys_edge` | `[E_msg]` | message edge -> physical edge |
| `decision_node` | `[M]` | decision -> 全局节点 id |
| `candidate_owner` | `[C]` | candidate -> decision |
| `candidate_is_null` | `[C]` | 是否 NULL |
| `branch_node_ids` | `[C, Ln]` | branch 节点 membership（**不含 owner**，padded） |
| `branch_node_lengths` | `[C]` | 上面每一行的有效长度 |
| `branch_edge_ids` | `[C, Le]` | branch 覆盖的**物理边** ID（padded） |
| `target_candidate` | `[M]` | GT z_0，flat candidate index |

`E_msg = 2 * E_phys`。padding 位置用 length mask 归零，不参与 mean。

**所有编号空间都是 batch 级的**：节点、decision、candidate 各加自己的 offset，
**物理边 ID 也加 per-graph offset**（`GraphSegments` 内部保持每图局部编号，
`collate_samples` 时平移）。漏掉物理边那一份偏移不会报错，只会让第二张图起
读写到别的图的边状态，属于最危险的一类 bug。

解码时同样要注意三个空间的区别：

- `z_t` / `candidate_log_prob`：flat **candidate** 空间 `[C]`；
- `target_candidate` / decoder 的输入：flat **decision** 空间 `[M]`，值是
  batch 级 candidate 索引；
- `decode_flat(sample, z0, decision_offset, candidate_offset)` 需要后两个偏移
  才能把全局索引换算回"本样本第 i 个 decision 的第 j 个候选"。

## 3. 单步数据流

```
                     z_t [M]
                       |
                       v
  candidate_is_null / branch_edge_* / msg_to_phys_edge
                       |
                       v
        expand_to_edge_state  ->  edge_state_id [E_msg] in {0,1}
                       |
                       v
        EdgeStateEncoder      ->  edge_feat_t [E_msg, d]

  H_t [N,d] --+
              |
     tau_t [B,d] -> TimeConditioner -> (gamma,beta) -> AdaLN(LN(H_t))
              |
              v
        GraphFlowBlock F_theta(H_t, E_t, tau_t)   x flow_steps 轮（见第 9 节）
              |   Q = W_Q h_hat[dst]
              |   K = W_K edge_feat
              |   V = W_V h_hat[src]
              |   alpha = softmax_{u in N(v)} <q_v, k_uv>/sqrt(d)   (dst 为 S/G 的边被剔除)
              |   m_v = sum alpha * V
              |   H~ = LN(H_t + W_O m)
              |   H_new = LN(H~ + FFN(H~))
              |   H_next = where(fixed_mask, H_t, H_new)
              v
        H_{t-1} [N,d]  ---------> 下一步的 H_t

                       |
                       v
        BranchMeanPool(H_{t-1})  ->  h_bar_ik [C, d]   (排除 owner i)
                       |
                       v
        BranchScorer: [h_i, h_bar_ik, tau_t] -> logit
        NULLScorer  : [h_i, tau_t]           -> logit
                       |
                       v
        grouped_log_softmax(logits, candidate_owner, M) over flat table
                       |
                       v
        p_theta(z_0) [C]
```

`EdgeStateEncoder` 的展开是纯索引/散射操作，没有 Python 循环：

1. `selected_candidate = ~candidate_is_null[z_t]` 置 True；
2. `member_active = selected_candidate[branch_edge_owner]`（并按 length mask）；
3. `physical_state.scatter_reduce_(..., reduce="amax")`（OR 合并）；
4. `edge_state_id = physical_state[msg_to_phys_edge]`（两方向共享）；
5. `edge_feat = E_edge[edge_state_id]`。

## 4. Reverse chain

```
H_t = model.init_nodes(batch)        # H_T = TypeEmbedding(V)，只调一次
z_t = sample_prior(...)              # z_T ~ pi

for t in T..1:
    out    = model.step(batch, H_t, z_t, t)
    dense  = dense_from_flat_prob(out.candidate_prob, candidate_owner, M)
    p_prev = model_reverse_posterior(dense, mask, z_t, candidate_owner, M, t)
    z_prev = sample_prev(p_prev)          # 或 sample_prev_deterministic
    H_t    = out.H_next                   # ← persistent
    z_t    = z_prev
```

`p_theta(z_{t-1}) = sum_c q(z_{t-1} | z_t, z_0=c) p_theta(z_0=c)` 的数学与 V1 完全
一致（`diffusion/posterior.py` 原样保留，未改动）。

## 5. 训练

```
z_path = diffusion.sample_forward_trajectory(z_0)     # 单步 alpha_t，coherent Markov chain
H = model.init_nodes(batch)
L = 0
for t in T..1:
    out = model.step(batch, H, z_path[t], t)
    L += clean_state_loss(out.candidate_log_prob, target_candidate)
    H = out.H_next                                     # 不 detach
L = L / T
L.backward()
```

- 不做随机单步训练：那样训练时的 `H_t` 与推理时不一致；
- 第一版不 detach（`truncate_every=0`），梯度沿 recurrent hidden state 反传；
- 损失是**加权** CE，不做组内归一化：

```
L_t = (1/M) * sum_i w_i * ( -log p_theta(z_0^i) )
L   = (1/T) * sum_t L_t
```

  `w_i` 由 `null_weight` / `active_weight` 决定（默认都是 1）。
  不要写成 `-log p / C_i`：均匀分布下会变成 `log(C_i)/C_i` 而不是 `log(C_i)`，
  梯度尺度会随候选数漂移。
- 命中率必须用 `losses.grouped_argmax`（在每个 decision 自己的候选组内取 argmax）。
  对一维 flat 候选表直接用 `argmax(dim=-1)` 得到的是 0-dim 张量，会让准确率
  静默错成"目标恰好是 0 号候选"的比例。

## 6. 与 V1 的本质差异

| | V1（已废弃） | V2 |
|---|---|---|
| candidate | 下一个邻居 | 整条 Branch Segment |
| 节点初始化 | LapPE/RWSE/degree + Static Encoder | 只有 4 类 node-type embedding |
| 条件注入 | z_t 写进节点特征 | z_t 展开成边状态（K） |
| 网络深度 | 每个 t 内堆多层 Graph Transformer | 1 个 Graph Flow Block / reverse step |
| 节点状态 | 每步重新编码 | persistent，跨步保留 |
| 训练 | 随机单步 | teacher-forced recurrent 整链 |

## 7. 第一版明确不实现

LapPE、RWSE、DegreeEmbedding、Node-ID Embedding、Global Pool、Static Encoder、
旧 Routing-State Encoder、multi-head attention（`d_head == d_model` 单头）、
scheduled sampling、hard-junction / path-level loss、edge cost encoder（因此
`data.weighted` 暂时禁用）、accelerated sampling（timestep skipping）。

---

## 8. 第二轮修订（修改清单 P0/P1/P2）

### P0-1 单出口 Source 的被迫段永久 selected

`deg(s) == 1` 时 Source 没有 decision variable，但 `s -> 第一个 endpoint` 这段
是必经之路。现在它在**每个 timestep**都是 selected：

```
E_t = E_source-forced  ∪  Psi(z_t)
```

- `GraphSegments.source_forced_edge_ids` / `source_forced_nodes` 保存这段；
- `Batch.source_forced_edge_ids` 是 batch 级（已加 physical edge offset）；
- `expand_to_edge_state` 先把这些边置 SELECTED，再用 `scatter_reduce(amax)`
  叠加 `z_t` 选中的 branch。

注意 `deg(s) > 1` 时 Source 自己是 decision node，此时 `source_forced_edge_ids`
为空列表 —— 不要和"source 的 branch"混为一谈。

### P0-2 按 graph 划分 train/val/test

同一张底层图的所有 OD query 共享 `meta["graph_id"]`，`split_dataset` 按 graph
分组后再切分，保证：

```
G_train ∩ G_val = G_train ∩ G_test = G_val ∩ G_test = ∅
```

另外 `_allocate_group_shares` 会在"图的数量够"时强制每个 split 至少 1 张图 ——
否则 4 张图按 0.8/0.1/0.1 取整会让 val 为空、`validate()` 静默什么都不返回
（这是真实踩到过的坑）。`build_datasets` 现在还会对空 split 直接报错。

### P1-1 weighted 模式禁用

模型只能看到 selected/unselected，看不到 `w_uv`。拓扑/OD/edge-state 相同但 cost
不同的样本模型输入完全一样，最优路径却可能不同 —— 信息上不可辨识。所以
`build_dataset(weighted=True)` 直接抛 `NotImplementedError`，等 Edge Cost Encoder
落地再打开。

### P1-2 统一 node 编号空间

`build_sample` 一进来就 `relabel_to_contiguous(graph, s, g)`，之后 graph /
start / goal / gt_path / segments 全部使用同一套 `0..N-1` 编号；
`extract_segments` 以 `relabel=False` 调用，不再偷偷产生第二套映射。

### P1-3 两个 residual 子层的 LayerNorm 分离

```
h~      = LN_1(h + W_O m)          attn_out_norm
h^{t-1} = LN_2(h~ + FFN(h~))       ffn_out_norm
```

### P2-1 deterministic prior 语义

`prior_deterministic` 更名为 `heuristic_prior_deterministic`：它（junction 取 NULL、
source 取第一条 branch）**不是** `z_T ~ pi` 的确定性等价物，而是一个明显偏向 NULL
的人造初值。想复现实验请保持 `z_T ~ pi` 但传入固定 seed 的 `torch.Generator`。

### P2-2 配置项真正生效

- `time.d_time`：真正传给 `TimeEncoder`（`d_time != d_model` 时插入
  `Linear(d_time, d_model)`）；奇数维度直接报错。
- `time.encoding` / `time.conditioning`：只支持 `sinusoidal` / `adaln`，
  其它取值构造时就 `NotImplementedError`，而不是被静默忽略。
- `training.amp`：真正用 `torch.amp.autocast` + `GradScaler`；只在 cuda 上启用，
  CPU 上传 true 也不会崩（明确忽略）。

### P2-3 sampler 只能跑完整链

`z_T ~ pi` 只在 `t = T` 成立，一般 `q(z_k) != pi`，所以"从 t=k 起跑 k 步"并不
等价于一条更短的扩散链。`sample_reverse_chain` 现在只接受
`max_steps is None or max_steps == diffusion.T`，其它值直接 `ValueError`。
训练侧的 `recurrent_reverse_loss(max_steps=k)` 是**另一件事**（截断 BPTT 的那条
前向轨迹长度），不受这条限制。

## 9. 单步内部的多轮信息交流（第三轮修订）

第一版每个 reverse step 只跑**一次** `F_theta`。这有两个后果：一是每轮只能看到一跳
邻居，二是长度为 `L_decision` 的决策链要跨 `T` 个 timestep 才能把信息传完。现在把
"一轮"改成"一个 step 内部连续 `flow_steps` 轮"，远距离信息可以在同一个 timestep
内多次传播。

```
H_0 = H_t
for k in 0 .. flow_steps-1:
    H_{k+1} = F_theta(H_k, E_t, tau_t + SlotEmbedding(k))
H_{t-1} = H_{flow_steps}
```

要点：

1. **每轮条件必须不同**。若各轮条件完全一样，`F` 反复作用于同一输入只会收敛到
   不动点，"多轮"退化成白算。所以每轮加一个**只与轮次有关、与 timestep 无关**的
   `SlotEmbedding(k)`，加在 `tau_t` 上后一起进入 `TimeConditioner`（AdaLN）。
2. **参数仍然只有一份 Cell**。`F_theta` 在所有 timestep、所有轮次间共享；新增的只有
   `nn.Embedding(flow_steps, d_model)`（`d_model=128, flow_steps=3` 时 384 个参数）。
3. **状态跨轮累积**：第 k+1 轮读的是第 k 轮写出的 `H`，不是重新从 `H_t` 开始。
   每一轮都执行 `H = where(fixed_mask, H, H_new)`，所以 Start/Goal 在任意轮次都被
   clamp 回输入；attention 仍是按 dst 分组的 softmax（含"Start/Goal 只出不进"）。
4. **向后兼容**：`flow_steps=1` 时 `forward_multi` 等价于原来的单次 `forward`
   （不构造 slot embedding），旧 checkpoint 与旧实验结论照旧可复现。
5. **BPTT 深度**：`flow_steps` 轮都参与反传（不 detach），因此单个 reverse step 的
   有效深度是 `flow_steps x cell_depth`；`training.max_bptt_steps` 的语义不变
   （它管的是时间轴上的截断，不是轮次）。

配置项：`model.flow_steps`、`model.flow_slot_embedding`、`model.flow_slot_scale`
（见 `configs/graph_flow.yaml`）。诊断信息：`DenoiserOutput.flow_steps` 与
`DenoiserOutput.attn_per_slot`。
