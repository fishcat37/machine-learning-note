# RoPE(**Rotary Position Embedding**)

传统的位置编码存在一个问题，他表示的的位置在计算注意力的时候只能获取全局信息，但我们希望的是在计算注意力的时候能关住二者的相对位置，所以我们需要一种将相对位置引入注意力计算的方法，所以首先我们需要先观察计算注意力的公式：

$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$

可以看到其中最关键的部分在于$QK^\top$，所以我们需要想办法在其中加入相对位置信息，这是我们就会借鉴数学或者其他领域中的重要方法，这里使用的就是信号处理中的一种方式：旋转相乘 → 相位相减

对于$q_p$和$k_q$来计算注意力，他们实际上是两个列向量，$k_q$进行T转置后对于这两个向量来说$q_p^\top k_q$就是逐元素相乘，，所以如果我们对传统正余弦位置编码进行一种旋转，其旋转定义如下：

对于位置为p的token，其(2k,2k+1)这两个维度的位置编码如下：

旋转角：$\theta_{p,k} = p \cdot \theta_k = p \cdot 10000^{-\frac{2k}{d}}$

(2k,2k+1)位置编码后的结果：$\begin{aligned}
\text{RoPE}(x_{2k}, x_{2k+1}, p) = 
\big[\, x_{2k} \cos \theta_{p,k} - x_{2k+1} \sin \theta_{p,k}, \;
x_{2k} \sin \theta_{p,k} + x_{2k+1} \cos \theta_{p,k} \,\big]
\end{aligned}$

实际上就是使$\begin{array}{c} R_p = 
\begin{bmatrix}
\cos \Theta_p & -\sin \Theta_p \\
\sin \Theta_p & \cos \Theta_p
\end{bmatrix} \end{array}$乘上列向量$\begin{bmatrix}x_{2k} \\ x_{2k+1}\end{bmatrix}$ ,就能得到$\begin{bmatrix}x_{2k} \cos \theta_{p,k} - x_{2k+1} \sin \theta_{p,k} \\ x_{2k} \sin \theta_{p,k} + x_{2k+1} \cos \theta_{p,k}\end{bmatrix}$，这一旋转位置编码，为什么这样能表示相对位置呢，正如前面所说，$q_p$和$k_q$会逐元素相乘，所以我们可以将编码后的值表示为$\tilde{x}_p = R_p x$，在计算$QK^\top$的时 候就能得到$(R_pq_p)^\top (R_qk_q)$能变成$q_p^\top R_p^\top R_q k_q$其中$R_p^\top$就等于$\begin{array}{c} R_p = 
\begin{bmatrix}
\cos (-\Theta_p) & -\sin (-\Theta_p) \\
\sin (-\Theta_p) & \cos (-\Theta_p)
\end{bmatrix} \end{array}$，即将角度反过来，然后他再与$R_q$相乘实际上就能得到$R_{p-q}$，这两步在线性代数中是可以很简单的推理出来的，所以在具体实现中，我们实际上只需要先获得最后一个维度，即head_size的d有多大来构造一个shape为[d//2]的$10000^{-\frac{2k}{d}}$，然后获得p后得到[seq_len,d//2]的角度，然后获得$(seq_len, h//2)$的sin和cos值，这个矩阵会缓存下来进行多次使用，因为该位置编码仅与位置相关，与值无关，所以可以复用

## AI实现代码

```py
import torch
import math
from typing import Tuple

def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造 sin/cos 缓存以复用（seq_len, dim//2）。
    dim 必须为偶数（通常是 head_dim）。
    返回 sin, cos，形状均为 (seq_len, dim//2)
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    # angles: (seq_len, half)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.sin(angles), torch.cos(angles)


def apply_rope_to_tensor(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    对张量 x 应用 RoPE。
    支持两类 x 形状：
      - (batch, seq_len, head_dim)
      - (batch, seq_len, n_head, head_dim)
    假设 head_dim is even.
    sin, cos: (seq_len, head_dim//2)
    返回与 x 相同形状的 tensor（已旋转）。
    """
    orig_shape = x.shape
    if x.dim() == 4:
        # (B, S, H, D)
        b, s, nh, d = x.shape
        assert d % 2 == 0
        x = x.view(b, s, nh, d // 2, 2)  # 把最后一维拆为 (..., half, 2)
        # sin/cos 需要扩展到 (1, seq_len, 1, half, 1)
        sin_ = sin[:, None, None, :, None]   # (S,1,1,half,1) after permute later
        cos_ = cos[:, None, None, :, None]
        # x[...,0] is real (even indices), x[...,1] is imag (odd indices)
        x0 = x[..., 0]  # (B,S,nh,half)
        x1 = x[..., 1]
        # apply rotation: [x0*cos - x1*sin, x0*sin + x1*cos]
        x0_rot = x0 * cos_ - x1 * sin_
        x1_rot = x0 * sin_ + x1 * cos_
        x_rot = torch.stack([x0_rot, x1_rot], dim=-1)  # (..., half, 2)
        x_rot = x_rot.view(b, s, nh, d)
        return x_rot.view(orig_shape)
    elif x.dim() == 3:
        # (B, S, D)
        b, s, d = x.shape
        assert d % 2 == 0
        x = x.view(b, s, d // 2, 2)
        sin_ = sin[None, :, None, None]  # (1,S,half,1)
        cos_ = cos[None, :, None, None]
        x0 = x[..., 0]
        x1 = x[..., 1]
        x0_rot = x0 * cos_ - x1 * sin_
        x1_rot = x0 * sin_ + x1 * cos_
        x_rot = torch.stack([x0_rot, x1_rot], dim=-1)
        x_rot = x_rot.view(b, s, d)
        return x_rot.view(orig_shape)
    else:
        raise ValueError("x must be 3D or 4D tensor")


# --- 示例：将 RoPE 应用到 Q,K 并计算 attention（非常常见写法） ---
def example_attention_with_rope(Q, K, V, sin, cos, attn_scale=True):
    """
    Q, K, V shapes:
      - Q,K: (B, S, nh, head_dim)  or (B, S, head_dim) for single-head
      - V: (B, S, nh, head_dim_v)  or (B, S, head_dim_v)
    sin, cos: (S, head_dim//2)
    返回 attention 输出 (B, S, nh, head_dim_v) 或 (B,S,head_dim_v)
    """
    # apply RoPE to Q and K (in-place optional but we avoid it here)
    Qp = apply_rope_to_tensor(Q, sin, cos)
    Kp = apply_rope_to_tensor(K, sin, cos)

    # reshape for batched matmul: (B, nh, S, head_dim)
    if Qp.dim() == 4:
        B, S, NH, D = Qp.shape
        Qp_ = Qp.permute(0, 2, 1, 3).contiguous().view(B * NH, S, D)
        Kp_ = Kp.permute(0, 2, 1, 3).contiguous().view(B * NH, S, D)
        V_  = V.permute(0, 2, 1, 3).contiguous().view(B * NH, S, -1)
    else:
        B, S, D = Qp.shape
        Qp_ = Qp
        Kp_ = Kp
        V_ = V

    # attention scores: (B*NH, S, S)
    scores = torch.matmul(Qp_, Kp_.transpose(-2, -1))
    if attn_scale:
        scores = scores / math.sqrt(D)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, V_)  # (B*NH, S, head_dim_v)

    # reshape back
    if Qp.dim() == 4:
        out = out.view(B, NH, S, -1).permute(0, 2, 1, 3).contiguous()  # (B,S,NH,head_dim_v)

    return out

```

# MRoPE(Mutimodel Rotary Position Embedding)

