# Graph-Junction-Diffusion V2 代码实施指南

**目标版本：Edge-State Conditioned Recurrent Graph Flow Denoiser**  
**配套设计文档：网络设计报告 V2.1**  
**实施原则：先把数据语义和单步 Graph Flow 做对，再接完整 Reverse Chain，最后接训练。**

---

## 0. 本次实现锁定的核心设计

这一版不要再兼容旧网络。代码从新的状态定义出发：

\[
\boxed{
(H_t,z_t)\rightarrow(H_{t-1},z_{t-1})
}
\]

其中：

- \(H_t\)：整张图的 persistent node state；
- \(z_t\)：每个 decision node 当前的 categorical branch state；
- \(z_t\) 每一步被展开成 selected / unselected edge-state field；
- Graph Flow 更新节点特征；
- 更新后的 \(H_{t-1}\) **原样传给下一步**；
- 同一个 Graph Flow Cell \(F_\theta\) 在所有 timestep 共享参数；
- Start / Goal 只发送信息，不接收信息；
- Branch Scorer 使用 Junction state + Branch Mean Pool + timestep embedding；
- categorical diffusion / posterior 数学保持不变。

第一版不要加入 LapPE、RWSE、DegreeEmbedding、Node-ID Embedding、Global Pool、Static Encoder、旧 Routing-State Encoder。

---

# 1. 建议目录

```text
Graph-Junction-Diffusion/
├── README.md
├── requirements.txt
├── configs/
│   └── graph_flow.yaml
│
├── docs/
│   ├── ARCHITECTURE_V2.md
│   └── IMPLEMENTATION_GUIDE_V2.md
│
├── src/
│   ├── data/
│   │   ├── graph_generators.py
│   │   ├── branch_segments.py
│   │   ├── decision_field.py
│   │   ├── dataset_builder.py
│   │   ├── dataset.py
│   │   └── collate.py
│   │
│   ├── diffusion/
│   │   ├── schedule.py
│   │   ├── posterior.py
│   │   ├── categorical.py
│   │   └── sampler.py
│   │
│   ├── models/
│   │   ├── node_embedding.py
│   │   ├── edge_state.py
│   │   ├── time_encoder.py
│   │   ├── graph_flow.py
│   │   ├── branch_scorer.py
│   │   └── denoiser.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── path_decoder.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── baselines.py
│   │
│   └── utils/
│       ├── nn.py
│       ├── segment_ops.py
│       ├── seed.py
│       └── config.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── inspect_dataset.py
│   ├── train.py
│   └── evaluate.py
│
└── tests/
    ├── test_branch_segments.py
    ├── test_decision_field.py
    ├── test_edge_state.py
    ├── test_graph_flow.py
    ├── test_branch_scorer.py
    ├── test_diffusion.py
    ├── test_persistent_state.py
    └── test_reverse_chain.py
```

如果旧项目中已经单独保存了 `categorical.py / posterior.py / schedule.py / segment_ops.py / time_encoder.py / nn.py / seed.py / graph_generators.py`，可以先复制回来，其余模块重新写。

---

# 2. 第一阶段：先把数据语义重新建立

这一阶段**不要写神经网络**。先确保一张 NetworkX 图能稳定转换成：

\[
G,s,g
\rightarrow
\text{Branch Segments}
\rightarrow
D
\rightarrow
\mathcal C_i
\rightarrow
z_0
\]

否则后面所有 tensor 都会错。

---

## 2.1 节点分类

定义：

```python
ORDINARY = 0
JUNCTION = 1
START = 2
GOAL = 3
```

节点类型规则：

```python
if v == s:
    START
elif v == g:
    GOAL
elif degree(v) >= 3:
    JUNCTION
else:
    ORDINARY
```

注意：Start / Goal 优先级高于 degree。

输出：

```text
node_type: LongTensor [N]
```

---

## 2.2 Decision Set

保持 categorical diffusion 的决策语义：

\[
J(G)=\{v:\deg(v)\ge3\}
\]

\[
D(G,s,g)
=
\left(
J(G)\cup\{s\mid \deg(s)>1\}
\right)
\setminus\{g\}.
\]

所以：

- Junction 是 decision node；
- Source 如果有多个出口，也是 decision node；
- Goal 永远没有 outgoing decision；
- Source 若只有一个出口，不需要 decision variable。

实现：

```python
def build_decision_nodes(graph, s, g):
    junctions = {v for v in graph.nodes if graph.degree(v) >= 3}
    if graph.degree(s) > 1:
        junctions.add(s)
    junctions.discard(g)
    return sorted(junctions)
```

---

# 3. Branch Segment 分解

这是 V2 数据层最重要的新逻辑。

旧版 candidate 是：

```text
i -> immediate neighbour
```

新版 candidate 是：

```text
decision node
→ ordinary nodes
→ next structural endpoint
```

---

## 3.1 Branch Endpoint

为了保证 dead-end 也能停止，工程上定义：

\[
A
=
\{v:\deg(v)\neq2\}\cup\{s,g\}.
\]

其中：

- degree \(\ge3\)：Junction；
- degree \(=1\)：dead-end；
- \(s,g\)：无论 degree 都强制作为 endpoint。

节点类型仍然只有四类；degree-1 dead-end 可以继续记作 Ordinary，只是在 Branch tracing 时作为停止点。

---

## 3.2 从一个 decision node 追踪一条 Branch

对于 decision node \(i\) 的每个邻居 \(n\)：

```text
path = [i, n]
prev = i
cur  = n
```

如果 `cur` 不是 endpoint，并且 `degree(cur)==2`：

1. 找到 `cur` 的另一个邻居；
2. `prev = cur`；
3. `cur = next`；
4. append；
5. 直到碰到 endpoint。

伪代码：

```python
def trace_branch(graph, owner, first_hop, endpoints):
    path = [owner, first_hop]
    prev = owner
    cur = first_hop

    while cur not in endpoints:
        nbrs = list(graph.neighbors(cur))
        assert len(nbrs) == 2

        nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
        path.append(nxt)

        prev, cur = cur, nxt

    return path
```

于是：

```text
J1 ─ a ─ b ─ J2
```

得到：

```python
[J1, a, b, J2]
```

而不是旧版的：

```python
J1 -> a
```

---

## 3.3 Branch Candidate 数据结构

建议每个 branch 先保留 Python 结构：

```python
@dataclass
class Branch:
    owner: int
    nodes: list[int]
    physical_edges: list[int]
    end: int
```

其中：

```text
owner = 当前 decision node
nodes = [owner, ..., endpoint]
physical_edges = 这条 branch 覆盖的无向物理边 ID
end = nodes[-1]
```

注意同一条物理 segment 从两个 Junction 看，可能对应两个**不同 candidate**：

```text
J1 -> J2
J2 -> J1
```

candidate 是有 owner 的决策类别，但它们可以覆盖相同 physical edges。

---

# 4. 物理边与 Message Edge 必须分开

建议一开始就区分：

```text
physical edge
message edge
```

对于无向物理边：

\[
\{u,v\}
\]

建立唯一 `physical_edge_id`。

同时 Graph Flow 中展开成：

\[
u\rightarrow v
\]

和：

\[
v\rightarrow u.
\]

需要保存：

```text
edge_index        : [2, E_msg]
msg_to_phys_edge  : [E_msg]
```

通常：

\[
E_{\text{msg}}=2E_{\text{physical}}.
\]

这样 Branch Segment 只记录 physical edge IDs；每个 timestep 再统一映射成 message-edge state。

这个设计能避免 Branch candidate 和 Attention 消息方向混在一起。

---

# 5. Clean Decision Field \(z_0\)

对于每个 decision node \(i\)：

普通 Junction：

\[
\mathcal C_i
=
\{NULL,B_{i1},\ldots,B_{iK_i}\}
\]

Source decision：

\[
\mathcal C_s
=
\{B_{s1},\ldots,B_{sK_s}\}
\]

Source 无 NULL。

---

## 5.1 从 GT path 确定正确 Branch

给定：

\[
P_{GT}=[s,\ldots,g]
\]

对一个位于 GT path 上的 decision node \(i\)，找到从 \(i\) 开始与 GT path 后续节点完全一致的 Branch Segment。

例如：

```text
GT:
... J1 -> a -> b -> J2 -> ...
```

则：

```text
z0[J1] = Branch(J1,a,b,J2)
```

不在 GT path 上的普通 Junction：

```text
z0 = NULL
```

Source 必须 active。

建议 validator 强制检查：

```text
1. every active branch really exists
2. active branch node sequence matches GT path
3. source never NULL
4. goal never decision
5. ordinary off-path decision -> NULL
6. every branch consists only of real graph edges
```

---

# 6. Batch / Collate：一次把后面需要的 tensor 准备齐

这一步非常关键。不要让模型内部重新做 NetworkX 遍历。

建议 `Batch` 至少包含：

```text
# graph
node_type               [N]
edge_index               [2, E_msg]
msg_to_phys_edge         [E_msg]
num_physical_edges

# graph ownership
node_graph_id            [N]
graph_node_ptr           [B+1]
starts                   [B]
goals                    [B]

# decision
decision_node            [M]
decision_graph_id        [M]

# flat candidates
candidate_owner          [C]
candidate_is_null        [C]
target_candidate         [M]

# branch node membership
branch_node_ids          [L_n]
branch_node_owner        [L_n]   # global candidate ID

# branch edge membership
branch_edge_ids          [L_e]   # physical edge ID
branch_edge_owner        [L_e]   # global candidate ID
```

NULL candidate：

- 出现在 `candidate_owner`；
- `candidate_is_null=True`；
- 没有 `branch_node_ids`；
- 没有 `branch_edge_ids`。

Branch candidate：

- `candidate_is_null=False`；
- 有完整 branch membership。

`target_candidate[m]` 建议继续使用**flat candidate index**，这样可直接复用 categorical diffusion。

---

# 7. Node Encoder：实际上只是 Node-Type Embedding

文件：

```text
src/models/node_embedding.py
```

实现不要复杂化：

```python
class NodeTypeEmbedding(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(4, d_model)

    def forward(self, node_type):
        return self.embedding(node_type)
```

输出：

\[
\boxed{
H_T\in\mathbb R^{N\times d}
}
\]

初始化只执行**一次**：

```python
H_t = node_encoder(batch.node_type)
```

随后整个 reverse chain 不再重新调用初始化。

---

# 8. Time Encoder

文件：

```text
src/models/time_encoder.py
```

输入：

```text
t
```

输出：

\[
\boxed{
\tau_t\in\mathbb R^d
}
\]

流程：

\[
t
\rightarrow
SinusoidalEncoding
\rightarrow
MLP
\rightarrow
\tau_t.
\]

建议：

```text
d_model = 128

Sinusoidal(128)
→ Linear(128,128)
→ SiLU
→ Linear(128,128)
```

如果完整 reverse chain 中一个 batch 的所有图共享当前 loop timestep：

```python
for t in range(T, 0, -1):
```

那么可以先生成：

```text
tau_t : [d]
```

再广播。

---

# 9. Edge-State Encoder

文件：

```text
src/models/edge_state.py
```

两个 learnable states：

```python
nn.Embedding(2, d_model)
```

```text
0 = unselected
1 = selected
```

重点不是 embedding 本身，而是：

\[
\boxed{
z_t\rightarrow edge\_state\_id_t
}
\]

---

## 9.1 Vectorized \(z_t\rightarrow E_t\)

输入：

```text
z_t                    [M]
candidate_is_null      [C]
branch_edge_ids        [L_e]
branch_edge_owner      [L_e]
msg_to_phys_edge       [E_msg]
```

第一步，确定当前哪些 candidate active：

```python
selected_candidate = torch.zeros(C, dtype=torch.bool, device=device)
active = ~candidate_is_null[z_t]
selected_candidate[z_t[active]] = True
```

第二步，看每条 branch-edge membership 是否被激活：

```python
member_active = selected_candidate[branch_edge_owner]
```

第三步 OR 到 physical edge：

```python
physical_state = zeros(E_phys)
physical_state.scatter_reduce_(
    0,
    branch_edge_ids,
    member_active.long(),
    reduce="amax"
)
```

第四步复制到 message edges：

```python
edge_state_id = physical_state[msg_to_phys_edge]
```

第五步 embedding：

```python
edge_feat = edge_embedding(edge_state_id)
```

得到：

\[
\boxed{
edge\_feat_t\in\mathbb R^{E_{msg}\times d}
}
\]

---

# 10. Graph Flow Block

文件：

```text
src/models/graph_flow.py
```

这是核心模块。

第一版使用**单头 Attention**，先不要做 multi-head。

---

## 10.1 输入输出

输入：

```text
H_t        [N,d]
edge_index [2,E_msg]
edge_feat  [E_msg,d]
tau_t      [d] / [B,d]
start_mask [N]
goal_mask  [N]
```

输出：

```text
H_next     [N,d]
```

对应：

\[
\boxed{
H_{t-1}=F_\theta(H_t,E_t,\tau_t)
}
\]

---

## 10.2 Time Conditioning

先：

\[
(\gamma_t,\beta_t)
=
MLP_{\rm cond}(\tau_t)
\]

然后：

\[
\hat H_t
=
(1+\gamma_t)\odot LN(H_t)+\beta_t.
\]

这里每个图内部所有节点共享本图对应的 \(\gamma_t,\beta_t\)。

---

## 10.3 Q / K / V

```python
src = edge_index[0]
dst = edge_index[1]
```

接收节点：

\[
Q_v=W_Q\hat h_v
\]

边：

\[
K_{uv}=W_Ke_{uv}^t
\]

发送节点：

\[
V_u=W_V\hat h_u
\]

tensor：

```python
q = W_Q(H_hat[dst])       # [E_msg,d]
k = W_K(edge_feat)        # [E_msg,d]
v = W_V(H_hat[src])       # [E_msg,d]
```

score：

```python
score = (q * k).sum(-1) / sqrt(d)
```

---

## 10.4 Start / Goal 不接收消息

构造：

```python
fixed_mask = start_mask | goal_mask
valid_msg = ~fixed_mask[dst]
```

只对：

```text
valid_msg == True
```

的 message edges 进行 softmax / aggregate。

这意味着：

```text
u -> s  删除
u -> g  删除
s -> u  保留
g -> u  保留
```

不要依赖 Attention 自己学会这一点。

---

## 10.5 Edge Softmax

对每个 destination node 单独做：

\[
\alpha_{uv}
=
Softmax_{u\in N(v)}(a_{uv}).
\]

可以直接复用：

```python
segment_softmax(score, dst, num_nodes)
```

得到：

```text
alpha [E_valid]
```

---

## 10.6 Message Aggregate

```python
weighted_v = alpha[:, None] * v
m = torch.zeros(N, d, device=device)
m.index_add_(0, dst, weighted_v)
```

对应：

\[
m_v
=
\sum_u
\alpha_{uv}W_Vh_u.
\]

---

# 11. Residual Update

绝不能：

```python
H_next = m
```

使用：

\[
\tilde H
=
LN(H_t+W_Om)
\]

\[
H_{new}
=
LN(\tilde H+FFN(\tilde H)).
\]

最后 clamp：

```python
H_new[fixed_mask] = H_t[fixed_mask]
```

更适合 autograd 的写法：

```python
H_next = torch.where(
    fixed_mask[:, None],
    H_t,
    H_new
)
```

所以 Start / Goal 始终保留初始 embedding。

---

# 12. Persistent State 是 Denoiser API 的一部分

这一版 `denoiser.forward()` 不能再是：

```python
model(batch, z_t, t)
```

而应该显式接收：

```python
model.step(batch, H_t, z_t, t)
```

建议输出：

```python
@dataclass
class DenoiserOutput:
    H_next: Tensor
    candidate_logits: Tensor
    candidate_log_prob: Tensor
    candidate_prob: Tensor
```

这是为了从接口层面保证：

\[
\boxed{
H_t\text{ 真的跨 timestep 传递}
}
\]

不要把 H 偷偷藏在 Module 内部 mutable state 中。

---

# 13. Branch Mean Pool

文件：

```text
src/models/branch_scorer.py
```

对于 branch candidate：

\[
B_{ik}=[i,v_1,\ldots,j]
\]

pool 时排除 owner \(i\)：

\[
\bar h_{ik}
=
\frac1{|B_{ik}\setminus\{i\}|}
\sum_{v\in B_{ik}\setminus\{i\}}
h_v.
\]

因此构造 dataset membership 时：

```text
branch_node_ids
```

建议就不要包含 owner。

Vectorized：

```python
node_feat = H_next[branch_node_ids]
branch_sum = segment_sum(
    node_feat,
    branch_node_owner,
    num_candidates
)

branch_count = segment_sum(
    torch.ones(...),
    branch_node_owner,
    num_candidates
)

branch_mean = branch_sum / branch_count.clamp_min(1)
```

NULL candidate 的 count=0，branch_mean 可以保持 0，因为 NULL 不会进入 branch scorer。

---

# 14. Branch Scorer

对于非 NULL candidate \(c=B_{ik}\)：

取 owner decision：

```python
decision_id = candidate_owner[c]
junction_node = decision_node[decision_id]
```

输入：

\[
[
h_i^{t-1},
\bar h_{ik}^{t-1},
\tau_t
]
\in\mathbb R^{3d}
\]

推荐：

```text
Linear(3d, 2d)
→ SiLU
→ Linear(2d, 1)
```

输出：

```text
branch_logit[c]
```

所有 Branch 共用同一个 MLP。

---

## 14.1 NULL Scorer

对于 NULL：

\[
[
h_i^{t-1},
\tau_t
]
\in\mathbb R^{2d}
\]

推荐：

```text
Linear(2d,d)
→ SiLU
→ Linear(d,1)
```

Source 没有 NULL candidate，所以数据层直接保证 Source candidate group 中不存在 NULL。

---

## 14.2 Grouped Softmax

把 NULL / Branch logits 最终填到一个：

```text
candidate_logits [C]
```

然后：

```python
candidate_log_prob = grouped_log_softmax(
    candidate_logits,
    candidate_owner,
    num_decisions
)
```

这样每个 Junction 自己归一化：

\[
\sum_{c\in\mathcal C_i}
p_i(c)=1.
\]

---

# 15. Denoiser 单步完整流程

文件：

```text
src/models/denoiser.py
```

建议只负责 orchestration，不要把所有代码堆在一个文件。

伪代码：

```python
def step(self, batch, H_t, z_t, t):

    tau_t = self.time_encoder(t)

    edge_feat_t = self.edge_state_encoder(
        z_t=z_t,
        candidate_is_null=batch.candidate_is_null,
        branch_edge_ids=batch.branch_edge_ids,
        branch_edge_owner=batch.branch_edge_owner,
        msg_to_phys_edge=batch.msg_to_phys_edge,
    )

    H_next = self.graph_flow(
        H_t=H_t,
        edge_index=batch.edge_index,
        edge_feat=edge_feat_t,
        tau_t=tau_t,
        fixed_mask=batch.start_goal_mask,
    )

    logits = self.branch_scorer(
        H=H_next,
        tau_t=tau_t,
        ...
    )

    log_prob = grouped_log_softmax(...)

    return DenoiserOutput(
        H_next=H_next,
        candidate_logits=logits,
        candidate_log_prob=log_prob,
        candidate_prob=log_prob.exp(),
    )
```

注意顺序固定：

\[
z_t
\rightarrow
E_t
\rightarrow
H_{t-1}
\rightarrow
BranchScore.
\]

Branch Scorer 用的是更新后的：

\[
H_{t-1}
\]

不是更新前的 \(H_t\)。

---

# 16. Categorical Diffusion

以下数学保持原样：

\[
Q_t
=
\alpha_tI+(1-\alpha_t)\mathbf1\pi^T
\]

\[
\bar Q_t
=
\bar\alpha_tI+(1-\bar\alpha_t)\mathbf1\pi^T.
\]

Denoiser 仍预测：

\[
p_\theta(z_0^i|\cdots).
\]

然后：

\[
p_\theta(z_{t-1}^i|z_t)
=
\sum_c
q(z_{t-1}^i|z_t^i,z_0^i=c)
p_\theta(z_0^i=c|\cdots).
\]

所以已经验证过的：

```text
schedule.py
posterior.py
categorical.py
```

尽量不要改数学代码，只改输入数据适配。

---

# 17. Reverse Sampler

文件：

```text
src/diffusion/sampler.py
```

这是 V2 和旧 sampler 最大的差异：

旧 state：

\[
z_t
\]

新 state：

\[
\boxed{
(H_t,z_t)
}
\]

---

## 17.1 初始化

```python
H_t = model.init_nodes(batch)      # 只执行一次
z_t = diffusion.sample_prior(...)
```

即：

\[
H_T=TypeEmbedding(V)
\]

\[
z_T\sim\pi.
\]

---

## 17.2 完整 Reverse Loop

```python
for t in range(T, 0, -1):

    out = model.step(
        batch=batch,
        H_t=H_t,
        z_t=z_t,
        t=t,
    )

    clean_dense, mask = diffusion.dense_from_flat_prob(
        out.candidate_prob,
        batch.candidate_owner,
        batch.num_decisions,
    )

    posterior = diffusion.model_reverse_posterior(
        clean_dense,
        mask,
        z_t,
        ...
        t=t,
    )

    z_prev = diffusion.sample_prev(...)

    H_t = out.H_next
    z_t = z_prev
```

最重要的是：

```python
H_t = out.H_next
```

而不是下一轮重新：

```python
H_t = model.init_nodes(...)
```

---

# 18. Training：不能再随机单步训练

Persistent \(H_t\) 意味着：

\[
H_t
\]

依赖之前已经发生的 reverse propagation history。

所以训练不能继续：

```text
随机 t
→ 独立采 z_t
→ 重置 H
→ 单步 loss
```

第一版使用：

\[
\boxed{
\text{full forward trajectory + teacher-forced reverse unroll}
}
\]

---

# 19. 先生成一条完整 Forward Noising Trajectory

给定 GT：

\[
z_0
\]

按单步 transition：

\[
z_0\rightarrow z_1\rightarrow\cdots\rightarrow z_T
\]

顺序采样。

需要在 `CategoricalDiffusion` 中补一个很小的函数：

```python
sample_forward_step(z_prev, t)
```

逻辑：

\[
z_t=
\begin{cases}
z_{t-1},&\text{prob } \alpha_t\\
Cat(\pi),&\text{prob }1-\alpha_t
\end{cases}
\]

注意这里使用：

\[
\alpha_t
\]

不是：

\[
\bar\alpha_t.
\]

这样生成的是一条真正 coherent 的 forward Markov trajectory。

保存：

```text
z_path[0], z_path[1], ..., z_path[T]
```

---

# 20. Teacher-Forced Reverse Training

初始化：

```python
H_t = model.init_nodes(batch)
```

然后：

```python
losses = []

for t in range(T, 0, -1):

    z_t = z_path[t]

    out = model.step(
        batch,
        H_t,
        z_t,
        t
    )

    loss_t = clean_state_loss(
        out.candidate_log_prob,
        batch.target_candidate
    )

    losses.append(loss_t)

    H_t = out.H_next
```

最终：

\[
\boxed{
L=
\frac1T
\sum_{t=1}^{T}
L_t
}
\]

一次 backward：

```python
loss.backward()
optimizer.step()
```

这时：

\[
H_T\rightarrow H_{T-1}\rightarrow\cdots
\]

整条计算图都存在，梯度可以沿 recurrent hidden state 反向传播。

第一版不要对：

```python
H_t = H_t.detach()
```

否则会直接切断跨 timestep 学习。

---

# 21. 第一版 Loss

先不要一开始加入复杂结构损失。

基础版：

\[
\boxed{
L_t
=
-\frac1M
\sum_i
\log p_\theta(z_0^i)
}
\]

即 clean-state categorical CE。

为了避免大量 NULL 主导，可以保留最简单的 active / NULL weighting：

\[
L_t
=
\frac1M
\sum_i
w_i
\left[
-\log p_\theta(z_0^i)
\right].
\]

但第一目标是验证新网络能否工作，先确保：

```text
tiny graph overfit
→ denoising works
→ full chain Goal Hit rises
```

之后再研究 hard-junction / path-level loss。

---

# 22. 训练与推理分布差异

Teacher forcing 时下一步使用真实 forward trajectory：

```text
z_t, z_{t-1}, ...
```

推理时使用模型自己采样：

```text
z_t -> model posterior -> z_{t-1}
```

两者存在 exposure bias。

第一版先不处理。

等基本模型跑通以后，再做：

```text
scheduled sampling
```

逐渐用模型生成的 `z_prev` 替换 teacher state。

不要在第一版同时引入，否则很难定位问题。

---

# 23. Path Decoder

最终 \(z_0\) 的 category 不再代表“下一个邻居”，而代表：

```text
一整条 Branch Segment
```

所以 decoder 应该：

```text
current = s
path = [s]

while current != g:
    找 current 的 decision
    读取 z0[current]

    if NULL:
        broken

    branch = selected branch
    append branch.nodes[1:]

    current = branch.end

    if repeated structural node:
        loop
```

如果 branch 终点是 dead-end 且不是 \(g\)：

```text
broken
```

Goal Hit：

```text
decoder 最终到达 g
```

---

# 24. Evaluation

第一版主指标只保留：

```text
Goal Hit Rate
Optimal Path Rate
Success Cost Ratio
Loop Rate
Broken Rate
Inference Time
```

其中：

\[
GoalHit
=
\frac{\#\text{到达 }g}{\#\text{queries}}
\]

Optimal Path Rate：

```text
成功路径长度 == GT shortest-path length
```

Success Cost Ratio：

\[
\frac{L_{\text{pred}}}{L_{\text{optimal}}}
\]

只在成功样本上统计。

不要再让 decision accuracy 成为模型选择主指标；它只作为 debug metric。

---

# 25. 必须写的测试

不要训练前才发现基础语义错了。

至少先写以下测试。

### 25.1 Branch Segment

```text
J1-a-b-J2
```

必须得到：

```python
[J1,a,b,J2]
```

不是：

```python
[J1,a]
```

---

### 25.2 Node Encoding

两个 Ordinary：

```text
v1, v2
```

初始化必须：

\[
h_{v1}^T=h_{v2}^T.
\]

Start 与 Goal 必须不同于 Ordinary/Junction 的 embedding entry。

---

### 25.3 Edge State Expansion

选中：

```text
J1-a-b-J2
```

则：

```text
J1-a
a-b
b-J2
```

全部 physical edges = selected。

其它边 = unselected。

两个 message directions 必须使用同一个 state。

---

### 25.4 NULL

如果：

```text
z_t[J1] = NULL
```

则 J1 的 candidate branches 不应产生任何 selected edge。

---

### 25.5 Start / Goal Clamp

无论邻居输入什么：

\[
h_s^{t-1}=h_s^t
\]

\[
h_g^{t-1}=h_g^t.
\]

---

### 25.6 Residual Persistence

构造两步：

```text
H_T -> H_{T-1} -> H_{T-2}
```

检查第二步输入确实是第一步输出。

禁止第二步重新初始化。

---

### 25.7 Parameter Sharing

整个 T-step chain 中只有一个：

```python
model.graph_flow
```

参数数量不能随 T 增长。

---

### 25.8 Grouped Softmax

每个 decision：

\[
\sum_{c\in C_i}p_i(c)=1.
\]

---

### 25.9 Categorical Posterior

旧 `test_diffusion.py` 全部继续通过。

---

# 26. 实施顺序

严格按下面顺序做，不要同时铺开。

### Milestone A：数据结构

完成：

```text
branch_segments.py
decision_field.py
dataset_builder.py
collate.py
```

验收：

```text
手工小图所有 Branch 完全正确
GT path -> z0 完全正确
batch membership 完全正确
```

---

### Milestone B：编码层

完成：

```text
node_embedding.py
edge_state.py
time_encoder.py
```

验收：

```text
H_T shape 正确
selected branch -> edge state 正确
tau_t shape 正确
```

---

### Milestone C：单步 Graph Flow

完成：

```text
graph_flow.py
```

验收：

```text
Q/K/V shape 正确
destination grouped softmax 正确
Start/Goal 不更新
Residual 保留 H
```

先不要接 diffusion。

---

### Milestone D：Branch Prediction

完成：

```text
branch_scorer.py
denoiser.py
```

验收：

```text
branch mean pool 正确
NULL/source category 正确
每个 decision probability sum = 1
```

---

### Milestone E：Reverse Sampler

接：

```text
categorical diffusion
persistent H
```

验收：

```text
(H_t,z_t) -> (H_{t-1},z_{t-1})
```

完整跑 T 步不报错。

---

### Milestone F：Recurrent Training

实现：

```text
full forward trajectory
teacher-forced reverse chain
full BPTT
```

先做 tiny overfit。

---

### Milestone G：正式数据与评估

最后才扩大：

```text
graph size
graph family
batch size
T
```

并比较：

```text
Goal Hit
Optimal Path
Cost Ratio
Loop
Broken
```

---

# 27. Tiny Overfit 是必须的

正式训练前造一个非常小的数据集：

```text
10~50 个固定 query
小图
固定 seed
```

训练直到：

```text
train clean-state prediction 接近 100%
full reverse chain Goal Hit 接近 100%
```

如果 tiny overfit 都做不到，禁止继续扩大数据。

这个阶段最适合检查：

```text
Branch mapping
z_t -> E_t
persistent H
posterior
decoder
```

---

# 28. 第一版推荐配置

```yaml
model:
  d_model: 128
  node_types: 4
  edge_states: 2
  ffn_hidden: 256
  dropout: 0.0

time:
  encoding: sinusoidal
  d_time: 128
  conditioning: adaln

diffusion:
  T: 50
  schedule: linear
  beta_start: 0.02
  beta_end: 0.20
  base_noise: uniform

training:
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  amp: true

loss:
  x0_ce: 1.0

evaluation:
  stochastic_sampling: true
```

这些只是第一版工程默认值，不是论文最终超参数。

---

# 29. 第一版 forward API 建议

最终模型接口尽量保持非常清楚：

```python
class GraphFlowDenoiser(nn.Module):

    def init_nodes(self, batch):
        return self.node_embedding(batch.node_type)

    def step(self, batch, H_t, z_t, t):
        tau_t = self.time_encoder(t)

        edge_feat_t = self.edge_state(
            batch=batch,
            z_t=z_t,
        )

        H_next = self.graph_flow(
            batch=batch,
            H_t=H_t,
            edge_feat_t=edge_feat_t,
            tau_t=tau_t,
        )

        logits, log_prob, prob = self.branch_scorer(
            batch=batch,
            H=H_next,
            tau_t=tau_t,
        )

        return DenoiserOutput(
            H_next=H_next,
            candidate_logits=logits,
            candidate_log_prob=log_prob,
            candidate_prob=prob,
        )
```

ReverseSampler：

```python
H_t = model.init_nodes(batch)
z_t = diffusion.sample_prior(batch)

for t in range(T, 0, -1):
    out = model.step(batch, H_t, z_t, t)

    z_prev = reverse_categorical_step(
        out.candidate_prob,
        z_t,
        t,
    )

    H_t = out.H_next
    z_t = z_prev
```

这个接口本身就把新网络最重要的设计表达出来了：

\[
\boxed{
H_t\text{ 是显式状态，并且被传到下一步。}
}
\]

---

# 30. 最后验收标准

第一版 V2 只有满足下面条件才算“实现完成”：

```text
[ ] Branch 已从 one-hop candidate 改成完整 Segment
[ ] Node 仅由 O/J/S/G embedding 初始化
[ ] Edge 仅有 selected / unselected embedding
[ ] Time 使用独立 tau_t 编码
[ ] Q = receiver node
[ ] K = edge state
[ ] V = sender node
[ ] Start/Goal 只出不进
[ ] Graph Flow 有 residual，不覆盖旧节点状态
[ ] H_{t-1} 直接进入下一 reverse step
[ ] 所有 timestep 共享同一个 Graph Flow Cell
[ ] Branch 用整段节点 Mean Pool
[ ] Source 无 NULL
[ ] grouped softmax 正确
[ ] categorical posterior 保持原数学
[ ] sampler state 是 (H_t,z_t)
[ ] training 使用 recurrent reverse-chain unroll
[ ] tiny overfit 通过
[ ] full-chain Goal Hit 能正常计算
```

如果这些全部满足，才进入后续 loss、scheduled sampling、图规模、OOD 和论文实验阶段。
