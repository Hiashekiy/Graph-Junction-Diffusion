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
        GraphFlowBlock F_theta(H_t, E_t, tau_t)
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
旧 Routing-State Encoder、multi-head attention（第一版固定单头，`full_attention=True`
时才用 `d_head=d_model`）、scheduled sampling、hard-junction / path-level loss、
多卡/AMP 调优。

这些都留给后续阶段；当前第一版的目标是先把

```
tiny overfit -> denoising works -> full chain Goal Hit 上升
```

这条链路跑通。
