# Swin Transformer

## 核心思想

Swin Transformer中使用shifed window来将图划分为多块，在每一块内进行注意力计算，但是这样没有window间的信息交互，所以在每一个layer中都会shift window来进行信息交互，具体来说，他每次会将整个window向右下移动单块的一半，然后具体的图还是原本的位置，这就会导致左上角的几个块会被切分为一些异构的小块，由于整个计算时有4种大小的attention block，会导致运算特别慢，所以我们会将左上角三个小块移动到右下角，如上图，然后在4个block中分别计算attention，然后通过mask要求attention只在物理上相邻的区域间计算attention，需要注意的是，在移位后，3左侧的小块不再与他连续，因为它们都是平移过去的，所以他们的相对位置变反了

## 架构图

具体架构图如下：
![[1772694660396.png]]
其中一开始一个patch为$4*4$，每个stage之后就将相邻的4个patch融合，他们在channel维度上拼接，然后通过layernorm之后使用mlp将通道数减半，也就是说每个stage channel翻倍，同时swin transformer支持不同大小的输入，因为在stage4之后，不管剩下的是$7*7$还是多少，我们都用global average pooling将他只留下channel维度

## 伪代码

### 1. 整体架构

```python
class SwinTransformer:
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7):
        # Patch Embedding: 将图像分割为patches
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        # 4个stage，每个stage包含多个Swin Transformer Block
        self.stages = []
        for i in range(4):
            stage = SwinStage(
                dim=embed_dim * (2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size
            )
            self.stages.append(stage)

            # 除了最后一个stage，都需要patch merging
            if i < 3:
                self.patch_merging = PatchMerging(dim=embed_dim * (2**i))

        # 分类头
        self.head = Linear(embed_dim * 8, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # [B, H/4 * W/4, C]

        for i, stage in enumerate(self.stages):
            x = stage(x)  # Swin Transformer Blocks
            if i < 3:
                x = self.patch_merging(x)  # 降低空间分辨率，增加通道数

        x = global_avg_pool(x)  # [B, C]
        x = self.head(x)  # [B, num_classes]
        return x
```

### 2. Swin Transformer Block

```python
class SwinTransformerBlock:
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # 0表示W-MSA，>0表示SW-MSA

        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0)

    def forward(self, x):
        # x: [B, H*W, C]
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 如果shift_size > 0，进行cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # 将特征图划分为windows
        x_windows = window_partition(shifted_x, self.window_size)
        # x_windows: [B*num_windows, window_size, window_size, C]

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # x_windows: [B*num_windows, window_size*window_size, C]

        # Window Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 将windows合并回特征图
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 如果之前进行了shift，需要reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                          dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # 残差连接
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x
```

### 3. Window Partition 和 Reverse

```python
def window_partition(x, window_size):
    """
    将特征图划分为不重叠的windows
    Args:
        x: [B, H, W, C]
        window_size: int
    Returns:
        windows: [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将windows合并回特征图
    Args:
        windows: [B*num_windows, window_size, window_size, C]
        window_size: int
        H, W: 原始特征图的高度和宽度
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

### 4. Window Attention with Relative Position Bias

```python
class WindowAttention:
    def __init__(self, dim, num_heads, window_size):
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads)
        )

        # 计算每个token对的相对位置索引
        self.relative_position_index = self._get_relative_position_index()

        self.qkv = Linear(dim, dim * 3)
        self.proj = Linear(dim, dim)

    def _get_relative_position_index(self):
        # 为window内的每个位置对计算相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # [2, Mh*Mw, Mh*Mw]

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # [Mh*Mw, Mh*Mw, 2]

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1

        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        return relative_position_index

    def forward(self, x, mask=None):
        # x: [B*num_windows, N, C], N = window_size * window_size
        B_, N, C = x.shape

        # 计算Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: [B_, num_heads, N, head_dim]

        # 计算attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # [N, N, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果有mask（用于SW-MSA），应用mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = softmax(attn, dim=-1)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x
```

### 5. Attention Mask 生成（用于SW-MSA）

```python
def create_mask(H, W, window_size, shift_size):
    """
    为shifted window创建attention mask
    Args:
        H, W: 特征图高度和宽度
        window_size: window大小
        shift_size: shift大小
    Returns:
        attn_mask: [num_windows, window_size*window_size, window_size*window_size]
    """
    img_mask = torch.zeros((1, H, W, 1))

    # 将特征图划分为9个区域（cyclic shift后）
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # 将mask划分为windows
    mask_windows = window_partition(img_mask, window_size)
    # [num_windows, window_size, window_size, 1]
    mask_windows = mask_windows.view(-1, window_size * window_size)

    # 创建attention mask
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # [num_windows, window_size*window_size, window_size*window_size]

    # 同一区域内的mask为0，不同区域间的mask为-100（softmax后接近0）
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask
```

### 6. Patch Merging

```python
class PatchMerging:
    def __init__(self, dim):
        self.dim = dim
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        self.norm = LayerNorm(4 * dim)

    def forward(self, x):
        # x: [B, H*W, C]
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        # 将相邻的2x2 patches合并
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
```

### 7. Swin Stage

```python
class SwinStage:
    def __init__(self, dim, depth, num_heads, window_size):
        self.blocks = []
        for i in range(depth):
            # 交替使用W-MSA和SW-MSA
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

## 关键特性总结

1. **层级化特征表示**：通过patch merging逐步降低空间分辨率，增加通道数
2. **局部注意力**：在固定大小的window内计算attention，复杂度为 $O(M^2 \cdot HW)$，其中M是window大小
3. **Shifted Window**：通过交替使用常规window和shifted window实现跨window信息交互
4. **相对位置偏置**：为attention添加可学习的相对位置偏置，增强位置感知能力
5. **高效的mask机制**：通过cyclic shift和attention mask实现shifted window，避免padding带来的计算浪费