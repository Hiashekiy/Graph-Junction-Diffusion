# Graph-Junction-Diffusion：Edge-State Recurrent Graph Flow Denoiser 设计报告

**版本：V2.1（2026-09-13，补全编码与实现细节）**  
**定位：Junction Categorical Diffusion 的新型图信息流去噪器设计**

---

## 1. 核心思想

旧网络是“先编码整张图，再把当前扩散状态写入节点”。新网络完全改成：

\[
\boxed{\text{Node = Information}}
\]

\[
\boxed{\text{Edge State = Information-Flow Condition}}
\]

\[
\boxed{\text{Reverse Diffusion = Recurrent Graph Information Propagation}}
\]

也就是说：

1. 节点负责承载和积累信息；
2. 当前扩散状态 \(z_t\) 被展开成 selected / unselected 的边状态；
3. 边状态控制本轮信息如何传播；
4. 每一轮 Graph Flow 得到的节点状态 \(H_{t-1}\) **完整保留到下一轮 Reverse Diffusion**；
5. 更新后的节点信息用于重新判断各 Junction 的 Branch；
6. 新 Branch 决策形成 \(z_{t-1}\)，再控制下一轮信息传播。

核心闭环：

\[
\boxed{
z_t
\rightarrow
E_t
\rightarrow
H_{t-1}
\rightarrow
p_\theta(z_0)
\rightarrow
z_{t-1}
}
\]

同时：

\[
\boxed{
H_T
\rightarrow
H_{T-1}
\rightarrow
H_{T-2}
\rightarrow
\cdots
\rightarrow
H_0
}
\]

因此真正的递推状态是：

\[
\boxed{
(H_t,z_t)
\longrightarrow
(H_{t-1},z_{t-1})
}
\]

---

## 2. 图结构与 Branch Segment

底层图保持原始节点级结构：

\[
G=(V,E)
\]

不压缩普通节点。节点只有四类：

\[
\boxed{
\text{Ordinary},\quad
\text{Junction},\quad
\text{Start},\quad
\text{Goal}
}
\]

例如：

```text
                ○──○──J2
               /
s──○──○──J1
               \
                ○──○──○──J3──○──g
```

虽然底层图仍由普通节点和边组成，但一个 Junction 的决策变量实际对应一整条 Branch Segment。

例如：

\[
B_{1,2}=[J_1,v_1,v_2,J_2]
\]

对应：

```text
J1 ─ ○ ─ ○ ─ J2
```

普通 Junction 的类别集合：

\[
\mathcal C_i=
\{NULL,B_{i1},B_{i2},\ldots,B_{iK_i}\}.
\]

Source 没有 NULL：

\[
\mathcal C_s=
\{B_{s1},B_{s2},\ldots,B_{sK_s}\}.
\]

---

## 3. 节点初始化与节点编码

新网络完全删除：

\[
LapPE,\qquad RWSE,\qquad DegreeEmbedding
\]

同时删除 Static Graph Encoder。节点在 \(t=T\) 时不携带任何显式图位置，只根据四种节点类型生成初始特征向量。

### 3.1 节点类型 ID

为每个节点分配一个离散类型：

\[
type(v)\in\{0,1,2,3\}
\]

约定：

```text
0 = Ordinary
1 = Junction
2 = Start
3 = Goal
```

Start / Goal 优先级最高。例如 \(s\) 本身即使度数大于 2，也使用 Start 类型，而不是 Junction 类型。

因此一个 batch 最基本的节点输入是：

```text
node_type : LongTensor [N]
```

这里 \(N\) 是 batch 中所有图节点总数。

### 3.2 Learned Node-Type Embedding

定义一个可学习 embedding table：

\[
\boxed{
E_{\text{node}}\in\mathbb R^{4\times d}
}
\]

其中 \(d=d_{\text{model}}\) 是整个 Graph Flow 网络统一使用的节点隐藏维度。第一版取：

\[
\boxed{d=128}
\]

节点初始向量通过查表得到：

\[
\boxed{
h_v^T
=
E_{\text{node}}[type(v)]
}
\]

即：

\[
h_v^T=
\begin{cases}
e_O,&v\text{ 是普通节点}\\
e_J,&v\text{ 是 Junction}\\
e_S,&v=s\\
e_G,&v=g
\end{cases}
\]

其中：

\[
e_O,e_J,e_S,e_G\in\mathbb R^d
\]

都是训练过程中自动学习的参数。

代码上对应：

```python
node_embedding = nn.Embedding(4, d_model)
H_T = node_embedding(node_type)      # [N, d_model]
```

因此这里不是把 `0/1/2/3` 直接作为连续数值送入网络，也不是给每个节点 ID 单独建 embedding，而是只对四种**节点类型**做 embedding lookup。

### 3.3 初始化性质

初始时所有同类型节点严格使用同一个向量：

\[
h_{v_1}^T=h_{v_2}^T=\cdots=e_O
\]

\[
h_{j_1}^T=h_{j_2}^T=\cdots=e_J.
\]

网络不会提前知道：

- 节点位于图中的什么位置；
- 节点距离 Start / Goal 多远；
- 节点属于哪条最终路径；
- 节点 ID 是多少。

节点之间的差异只能由后续：

\[
\boxed{
\text{图拓扑 + Start/Goal 信息传播 + Edge State + 历史状态}
}
\]

逐渐形成。

### 3.4 参数初始化

第一版不需要特殊技巧。推荐：

\[
E_{\text{node}}\sim\mathcal N(0,0.02^2)
\]

或者使用 PyTorch 默认初始化。重点是四种类型使用四个独立、可学习、共享于所有图的向量。

---

## 4. Diffusion State 转换与边编码

当前 categorical diffusion state：

\[
z_t=\{z_t^i\}
\]

表示每个 Junction 当前选择哪一条 Branch。

如果：

\[
z_t^i=B_{ik}
\]

则 Branch \(B_{ik}\) 被认为处于 selected 状态；其余 candidate branch 为 unselected。

如果：

\[
z_t^i=NULL
\]

则该 Junction 的所有 Branch 都为 unselected。

### 4.1 边状态只有两类

定义离散 edge-state ID：

```text
0 = unselected
1 = selected
```

于是每条用于消息传递的底层边都有：

```text
edge_state_id : LongTensor [E_msg]
```

其中 \(E_{\text{msg}}\) 是消息图中的有向边数量。对于无向原图，一条物理边展开为两个 message edges：

\[
u\rightarrow v,\qquad v\rightarrow u.
\]

### 4.2 Learned Edge-State Embedding

定义可学习 embedding table：

\[
\boxed{
E_{\text{edge}}\in\mathbb R^{2\times d_e}
}
\]

第一版直接令：

\[
\boxed{
d_e=d
}
\]

于是：

\[
e_{uv}^{t}
=
E_{\text{edge}}[state_{uv}^{t}]
\in\mathbb R^d
\]

代码：

```python
edge_embedding = nn.Embedding(2, d_model)
edge_feat_t = edge_embedding(edge_state_id)   # [E_msg, d_model]
```

其中：

\[
E_{\text{edge}}[0]=e_{\mathrm{unsel}},\qquad
E_{\text{edge}}[1]=e_{\mathrm{sel}}.
\]

selected / unselected 不对应手工固定的大小关系；两个向量都是独立可学习参数。

### 4.3 从 \(z_t\) 展开到整张图的 Edge-State Field

需要预先保存每个 Branch Segment 覆盖哪些底层物理边：

```text
branch_edge_ids
branch_edge_ptr
```

例如：

```text
B_ik = [J_i, v1, v2, J_k]
```

覆盖：

```text
(J_i,v1), (v1,v2), (v2,J_k)
```

在 timestep \(t\)：

1. 全部物理边先设为 `unselected`；
2. 遍历当前 \(z_t\) 中被选中的 Branch；
3. 将该 Branch 覆盖的所有物理边设为 `selected`；
4. 无向图中两个消息方向共享同一个物理边状态。

因此：

\[
\boxed{
E_t=\Psi(z_t)
}
\]

具体链路：

\[
z_t
\rightarrow
state_t\in\{0,1\}^{E_{\text{msg}}}
\rightarrow
E_{\text{edge}}[state_t]
\rightarrow
\{e_{uv}^t\}.
\]

若同一条物理 Branch 同时被两个端点引用，第一版采用**并集规则**：

\[
\boxed{
state_{uv}^t=1
\iff
\text{至少一个当前 active candidate 覆盖该物理边}
}
\]

即 OR 合并，不引入第三种冲突状态。

### 4.4 NULL 的编码

NULL 不拥有单独的 edge embedding：

\[
\boxed{
z_t^i=NULL
\Longrightarrow
\text{Junction }i\text{ 的 candidate branches 不向图中写入 selected}
}
\]

所以 NULL 完全由“相关 Branch 仍为 unselected”表达。

### 4.5 第一版统一维度与核心张量

第一版统一采用：

\[
\boxed{
d_{\text{model}}=128,\qquad
d_{\text{edge}}=128,\qquad
d_{\text{time}}=128
}
\]

核心张量约定：

| 张量 | 形状 | 含义 |
|---|---:|---|
| `node_type` | `[N]` | 四类节点 ID |
| `H_t` | `[N,d]` | 当前 persistent node state |
| `edge_index` | `[2,E_msg]` | 有向 message edges |
| `edge_state_id_t` | `[E_msg]` | 当前 selected/unselected |
| `edge_feat_t` | `[E_msg,d]` | edge-state embedding |
| `decision_node` | `[M]` | 每个 decision 对应的图节点 |
| `candidate_owner` | `[C]` | 每个 candidate 属于哪个 decision |
| `candidate_is_null` | `[C]` | 是否 NULL |
| `branch_node_ids` | `[L_n]` | 所有 Branch 的节点 membership 扁平表 |
| `branch_node_owner` | `[L_n]` | membership 属于哪个 branch candidate |
| `branch_edge_ids` | `[L_e]` | 所有 Branch 覆盖的物理边 membership |
| `branch_edge_owner` | `[L_e]` | edge membership 属于哪个 branch candidate |
| `z_t` | `[M]` | 每个 decision 当前类别，使用 flat candidate index |
| `tau_graph` | `[B,d]` | 每张图当前 timestep embedding |

其中：

- \(B\)：batch 中图数量；
- \(N\)：batch 中总节点数；
- \(E_{\text{msg}}\)：总有向消息边数；
- \(M\)：总 decision 数；
- \(C\)：总 candidate 数；
- \(L_n\)：所有 branch-node membership 总长度；
- \(L_e\)：所有 branch-edge membership 总长度。

神经网络主路径尽量围绕这些扁平张量工作，避免 Python 循环。

---

## 5. Graph Flow Block：Q 来自接收节点，K 来自边状态，V 来自发送节点

对于有向消息：

\[
u\rightarrow v
\]

Query：

\[
\boxed{
q_v=W_Qh_v^t
}
\]

Key：

\[
\boxed{
k_{u\rightarrow v}=W_Ke_{uv}^t
}
\]

Value：

\[
\boxed{
v_{u\rightarrow v}=W_Vh_u^t
}
\]

角色划分：

```text
h_u
 │
 └── Value：传播什么信息

e_uv^t
 │
 └── Key：当前这条通道是什么状态

h_v
 │
 └── Query：当前节点怎样接收信息
```

因此：

\[
\boxed{\text{Node = Information}}
\]

\[
\boxed{\text{Edge = Information-flow condition}}
\]

---

## 6. Attention 与消息聚合

Attention score：

\[
\boxed{
a_{u\rightarrow v}
=
\frac{
q_v^\top k_{u\rightarrow v}
}{
\sqrt{d_h}
}
}
\]

在节点 \(v\) 的所有入邻居之间做 softmax：

\[
\boxed{
\alpha_{u\rightarrow v}
=
\operatorname{Softmax}_{u\in\mathcal N(v)}
(a_{u\rightarrow v})
}
\]

得到本轮传播来的新信息：

\[
\boxed{
m_v^t
=
\sum_{u\in\mathcal N(v)}
\alpha_{u\rightarrow v}
W_Vh_u^t
}
\]

同状态边拥有相同的 Key；发送节点之间的差异不进入 Key，而体现在 Value 中。这正符合当前设计：同一种 selected/unselected 通道采用同一种传播条件，而节点差异由历史传播自然产生。

---

## 7. 节点更新：必须保留旧信息

不能直接：

\[
h_v^{t-1}=m_v^t.
\]

因为这样会丢掉节点之前已经积累的信息。

采用 Residual Update：

\[
\boxed{
\tilde h_v
=
LN
\left(
h_v^t+W_Om_v^t
\right)
}
\]

再经过 FFN：

\[
\boxed{
h_v^{t-1}
=
LN
\left(
\tilde h_v+FFN(\tilde h_v)
\right)
}
\]

因此：

\[
\boxed{
\text{New Node State}
=
\text{Old Node State}
+
\text{Newly Propagated Information}
}
\]

---

## 8. Persistent Node State：节点状态跨 Reverse Step 直接保留

这是新网络最重要的机制之一。

每一步 Graph Flow 得到的：

\[
H_{t-1}
\]

**直接作为下一步 Reverse Diffusion 的节点输入。**

也就是说：

\[
\boxed{
H_{t-1}
\text{ 不重置、不重新初始化、不重新编码图}
}
\]

完整过程：

```text
H_T , z_T
   │
   ▼
Graph Flow
   │
   ▼
H_{T-1}
   │
   ├── Branch Scoring → z_{T-1}
   │
   ▼
直接作为下一步节点状态
   │
   ▼
Graph Flow
   │
   ▼
H_{T-2}
   │
   ├── Branch Scoring → z_{T-2}
   │
   ▼
...
```

因此：

\[
\boxed{
H_T
\rightarrow
H_{T-1}
\rightarrow
H_{T-2}
\rightarrow
\cdots
\rightarrow
H_0
}
\]

是一条真正的 recurrent hidden-state chain。

与此同时：

\[
\boxed{
z_T
\rightarrow
z_{T-1}
\rightarrow
z_{T-2}
\rightarrow
\cdots
\rightarrow
z_0
}
\]

是 categorical reverse-diffusion chain。

两条链同步演化：

\[
\boxed{
(H_t,z_t)
\rightarrow
(H_{t-1},z_{t-1})
}
\]

其中：

- \(H_t\)：图上已经积累的信息；
- \(z_t\)：当前 Branch 的 noisy selection state。

**节点信息持续保留；边状态持续去噪。**

---

## 9. Start 与 Goal：固定信息源，只流出、不流入

Start 和 Goal 不接受邻居更新：

\[
\boxed{
h_s^{t-1}=h_s^t=e_S
}
\]

\[
\boxed{
h_g^{t-1}=h_g^t=e_G
}
\]

消息传播时直接禁止：

\[
u\rightarrow s
\]

和：

\[
u\rightarrow g.
\]

但允许：

\[
s\rightarrow v,\qquad g\rightarrow v.
\]

即：

```text
邻居 ─X→ s
s ────→ 邻居

邻居 ─X→ g
g ────→ 邻居
```

因此 Start 和 Goal 是两个固定的信息源，它们不会被周围节点污染，但会持续向图中输出信息。

---

## 10. 每一个 Reverse Diffusion Step 只执行一次 Graph Flow

新网络不再在每个 diffusion step 内部堆多层 Graph Transformer。

因为现在：

\[
\boxed{
\text{Reverse Diffusion Step 本身就是 Graph Information Propagation Step}
}
\]

第一版固定为：

\[
\boxed{
1\text{ 个 Graph Flow Block / Reverse Step}
}
\]

例如 \(T=50\)：

```text
t = 50:
H50 + E50 → Flow一次 → H49

t = 49:
H49 + E49 → Flow一次 → H48

t = 48:
H48 + E48 → Flow一次 → H47

...

t = 1:
H1 + E1 → Flow一次 → H0
```

所以完整 reverse chain 本身就提供了 50 轮信息传播。

---

## 11. Graph Flow Block 参数在所有 timestep 上共享

不是：

\[
F_1,F_2,\ldots,F_{50}.
\]

而是只有一个 Graph Flow Cell：

\[
\boxed{
F_\theta
}
\]

沿 reverse diffusion 时间轴重复调用：

\[
\boxed{
H_{t-1}
=
F_\theta(H_t,E_t,\tau_t)
}
\]

即：

```text
                 同一个 Fθ
                    │
H50,E50,e50 ────────┤
                    ▼
                   H49
                    │
H49,E49,e49 ────────┤  同一个 Fθ
                    ▼
                   H48
                    │
                   ...
```

参数：

\[
W_Q,W_K,W_V,W_O,FFN
\]

在所有 reverse steps 上完全共享。

Branch Scorer、NULL Scorer 和 Time Encoder 也同样共享。

因此参数集合只有一套：

\[
\boxed{
\Theta=
\{
E_{\mathrm{node}},
E_{\mathrm{edge}},
TimeEncoder,
GraphFlowBlock,
BranchScorer,
NullScorer
\}
}
\]

不同 timestep 的行为仍然不同，因为每一步输入都不同：

\[
H_t\neq H_{t-1},
\qquad
E_t\neq E_{t-1},
\qquad
\tau_t\neq e_{t-1}.
\]

所以：

\[
\boxed{
\text{Shared parameters}
\neq
\text{identical behaviour at every timestep}
}
\]

更准确地说，网络反复执行同一个规则：

> 根据当前节点信息和当前边状态，再推进一轮信息传播。

---

## 12. 时间编码与 Timestep Conditioning

为了避免和边特征 \(e_{uv}^t\) 混淆，本报告统一把时间步编码记为：

\[
\boxed{
\tau_t
}
\]

节点初始化只使用节点类型，但 Graph Flow 仍然需要知道当前处于 reverse diffusion 的哪个阶段：

\[
t
\rightarrow
TimeEncoder
\rightarrow
\tau_t.
\]

其中：

\[
\tau_t\in\mathbb R^d
\]

是一个**全局 timestep condition**。同一个图、同一个 reverse step 内所有节点共享同一个 \(\tau_t\)，所以它不会区分节点位置。

### 12.1 Sinusoidal Time Encoding

先将整数 \(t\in\{1,\ldots,T\}\) 编成：

\[
s_t\in\mathbb R^{d_t}
\]

其中：

\[
s_t[2k]
=
\sin
\left(
t\cdot 10000^{-2k/d_t}
\right)
\]

\[
s_t[2k+1]
=
\cos
\left(
t\cdot 10000^{-2k/d_t}
\right).
\]

第一版取：

\[
d_t=d.
\]

### 12.2 Time MLP

再通过：

\[
\boxed{
\tau_t=MLP_{\text{time}}(s_t)
}
\]

推荐结构：

```text
Linear(d, d)
→ SiLU
→ Linear(d, d)
```

代码形态：

```python
time_feat = sinusoidal_encoding(t, d_model)
tau_t = time_mlp(time_feat)          # [B, d_model]
```

batch 中不同图可处于不同 \(t\)，先得到：

```text
tau_graph : [B, d]
```

再通过 `node_graph_id` / `decision_graph_id` 广播。

### 12.3 时间如何进入 Graph Flow

\(\tau_t\) 不是 Node ID，也不是图位置编码，只用于告诉共享的 \(F_\theta\) 当前处于高噪声还是低噪声阶段。

第一版使用 AdaLN / FiLM 风格 conditioning：

\[
(\gamma_t,\beta_t)
=
MLP_{\text{cond}}(\tau_t)
\]

其中：

\[
\gamma_t,\beta_t\in\mathbb R^d.
\]

定义：

\[
\boxed{
\hat h_v^t
=
(1+\gamma_t)\odot LN(h_v^t)+\beta_t
}
\]

然后：

\[
Q_v=W_Q\hat h_v^t,
\qquad
K_{uv}=W_Ke_{uv}^t,
\qquad
V_u=W_V\hat h_u^t.
\]

因此时间改变的是“这一轮如何使用节点信息”，边 Key 仍然只来自 edge-state embedding，不破坏：

\[
\boxed{
\text{Node = information,\qquad Edge = information-flow condition}
}
\]

的核心分工。

### 12.4 Branch Scorer 中的时间编码

当前基线保留 timestep condition：

\[
\boxed{
c_{ik}
=
[
h_i^{t-1},
\bar h_{ik}^{t-1},
\tau_t
]
}
\]

因此 Branch Scorer 输入维度：

\[
\boxed{3d}
\]

NULL Scorer 输入：

\[
[h_i^{t-1},\tau_t]\in\mathbb R^{2d}.
\]

这里的 \(\tau_t\) 仅表示当前去噪阶段，不是边特征。

---

## 13. Branch Representation：读取整条 Branch 的节点信息

本轮 Graph Flow 完成后得到：

\[
H_{t-1}.
\]

对于 Junction \(i\) 的候选 Branch：

\[
B_{ik}
=
[i,v_1,v_2,\ldots,j]
\]

对整条 Branch 上的节点状态做平均，当前 Junction \(i\) 自己不参与 pooling：

\[
\boxed{
\bar h_{ik}^{\,t-1}
=
\frac1{|B_{ik}\setminus\{i\}|}
\sum_{v\in B_{ik}\setminus\{i\}}
h_v^{t-1}
}
\]

这就是：

\[
\boxed{
\text{Branch 当前整体的信息状态}
}
\]

---

## 14. Branch Scorer 与 NULL Scorer

对于 Branch \(B_{ik}\)：

\[
\boxed{
c_{ik}
=
[
h_i^{t-1},
\bar h_{ik}^{\,t-1},
\tau_t
]
}
\]

共享：

\[
MLP_{\mathrm{branch}}
\]

得到：

\[
\boxed{
\ell_{ik}
=
MLP_{\mathrm{branch}}(c_{ik})
}
\]

它回答：

> 根据当前 Junction 已经获得的信息，以及这条 Branch 上整体的信息，这条 Branch 应不应该被选择？

普通 Junction 的 NULL：

\[
\boxed{
\ell_{i,NULL}
=
MLP_{\mathrm{NULL}}
(
[h_i^{t-1},\tau_t]
)
}
\]

然后对每个 Junction 自己的 ragged candidate set 做 grouped softmax：

\[
\boxed{
p_\theta(z_0^i)
=
Softmax
(
\ell_{i,NULL},
\ell_{i1},
\ldots,
\ell_{iK_i}
)
}
\]

Source 无 NULL：

\[
\boxed{
p_\theta(z_0^s)
=
Softmax
(
\ell_{s1},\ldots,\ell_{sK_s}
)
}
\]

---

## 15. Categorical Reverse Posterior 保持不变

Denoiser 预测：

\[
p_\theta
(
z_0^i
\mid
H_{t-1},z_t,t
).
\]

继续使用原来的 categorical posterior：

\[
\boxed{
p_\theta(z_{t-1}^i\mid z_t)
=
\sum_c
q(z_{t-1}^i\mid z_t^i,z_0^i=c)
p_\theta(z_0^i=c\mid\cdots)
}
\]

采样：

\[
\boxed{
z_{t-1}
\sim
p_\theta(z_{t-1}\mid z_t)
}
\]

新的 \(z_{t-1}\) 再生成：

\[
E_{t-1}=\Psi(z_{t-1})
\]

并作用在已经保留下来的：

\[
H_{t-1}
\]

上。

---

## 16. 完整 Reverse Chain

初始化：

\[
\boxed{
H_T
=
TypeEmbedding(V)
}
\]

\[
\boxed{
z_T\sim\pi
}
\]

对于：

\[
t=T,T-1,\ldots,1
\]

依次执行：

### Step 1：Edge-State Construction

\[
\boxed{
E_t=\Psi(z_t)
}
\]

### Step 2：Persistent Graph Flow

\[
\boxed{
H_{t-1}
=
F_\theta(H_t,E_t,\tau_t)
}
\]

其中：

\[
\boxed{
H_{t-1}
\text{ 直接保留到下一步，不重新初始化}
}
\]

### Step 3：Branch Pooling

\[
\boxed{
R_{ik}^{t-1}
=
MeanPool
\left(
\{h_v^{t-1}:v\in B_{ik}\setminus\{i\}\}
\right)
}
\]

### Step 4：Clean-State Prediction

\[
\boxed{
p_\theta(z_0\mid H_{t-1},t)
}
\]

### Step 5：Categorical Posterior

\[
\boxed{
p_\theta(z_{t-1}\mid z_t)
}
\]

### Step 6：Sample Next Edge Decisions

\[
\boxed{
z_{t-1}
\sim
p_\theta(z_{t-1}\mid z_t)
}
\]

然后继续：

\[
\boxed{
(H_{t-1},z_{t-1})
}
\]

---

## 17. 完整网络流程图

```text
┌──────────────────────────────┐
│ Node Type Embedding          │
│ Ordinary / Junction / S / G  │
└──────────────┬───────────────┘
               │
               ▼
              H_T
               │
        z_T ───┤
         │     │
         ▼     │
   Branch selected /
      unselected
         │
         ▼
      Edge State E_T
         │
         └──────────┐
                    ▼
        ┌────────────────────┐
        │ Shared Graph Flow  │
        │ Cell F_theta       │
        │                    │
        │ Q = node receiver  │
        │ K = edge state     │
        │ V = node sender    │
        └─────────┬──────────┘
                  │
                  ▼
                H_{T-1}
                  │
                  │  ← 完整保留到下一步
                  │
           Branch Mean Pool
                  │
                  ▼
            Branch Scorer
                  │
                  ▼
              p_theta(z_0)
                  │
           categorical posterior
                  │
                  ▼
                z_{T-1}
                  │
                  ▼
           Edge State E_{T-1}
                  │
                  └──────────────┐
                                 ▼
                         同一个 F_theta
                                 │
                                 ▼
                              H_{T-2}
                                 │
                                ...
                                 │
                                 ▼
                               H_0,z_0
```

---

## 18. 训练方式必须与 Persistent State 匹配

由于：

\[
H_t
\]

来自前面若干 reverse steps 的累计传播，因此训练不能再简单地：

> 随机抽一个 \(t\)，重新初始化节点，然后只训练这一单步。

那样训练时的 \(H_t\) 与推理时的 \(H_t\) 不一致。

训练至少需要沿一段 reverse chain 展开：

1. 从 GT \(z_0\) 构造 noisy categorical trajectory；
2. 只初始化一次 \(H_T=TypeEmbedding(V)\)；
3. 连续调用同一个 \(F_\theta\)；
4. 每一步得到的 \(H_{t-1}\) 直接传入下一步；
5. 每一步都可预测 \(z_0\) 并计算监督；
6. 后续可加入 model self-rollout，缩小 teacher forcing 与真实采样链的差异。

因此训练逻辑需要从“独立单步训练”转向：

\[
\boxed{
\text{Recurrent reverse-chain training}
}
\]

---

## 19. 新旧网络的本质区别

### 旧网络

```text
图结构
  ↓
预先编码出强静态节点表示
  ↓
每个 t 重新读取同一个图表示
  ↓
z_t 只是额外条件
  ↓
预测 z_0
```

### 新网络

```text
初始只知道节点类型
  ↓
Start / Goal 向外传播信息
  ↓
z_t 决定当前 Branch 的 selected / unselected 状态
  ↓
Edge State 改变本轮信息传播
  ↓
节点状态 H 持续累积并保留
  ↓
H 反过来决定新的 Branch 选择
  ↓
z_{t-1}
  ↓
继续传播
```

核心闭环：

\[
\boxed{
\text{Edge Decisions}
\rightarrow
\text{Information Flow}
\rightarrow
\text{Persistent Node States}
\rightarrow
\text{Edge Decisions}
}
\]

---

## 20. 最终结构定义

整个方法仍然是：

\[
\boxed{
\textbf{Junction Categorical Diffusion}
}
\]

新的 Denoiser 定义为：

\[
\boxed{
\textbf{Edge-State Conditioned Recurrent Graph Flow Denoiser}
}
\]

最核心的状态递推：

\[
\boxed{
E_t=\Psi(z_t)
}
\]

\[
\boxed{
H_{t-1}=F_\theta(H_t,E_t,\tau_t)
}
\]

\[
\boxed{
z_{t-1}
\sim
p_\theta(z_{t-1}\mid z_t,H_{t-1},t)
}
\]

整个 Reverse Diffusion 中有两个共同演化的状态：

\[
\boxed{
\underbrace{z_t}_{\text{Branch Decision Field}}
+
\underbrace{H_t}_{\text{Persistent Graph Information Field}}
}
\]

其中：

- \(z_t\) 控制当前信息流；
- \(H_t\) 记录已经传播到图上的信息；
- \(H_t\) **每一步完整保留并直接传递到下一步**；
- 同一个 Graph Flow Cell \(F_\theta\) 在所有 timestep 上共享参数；
- Start / Goal 是固定信息源；
- Branch prediction 读取 Junction 状态和整条 Branch 的聚合状态；
- categorical posterior 继续完成 \(z_t\rightarrow z_{t-1}\)。

这就是当前最终确定的新网络结构。
