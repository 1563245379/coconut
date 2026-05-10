# Plan: 实现"范数保持随机方向"随机化方法

## Context

根据研究报告（`research/deep-research-report.md`）第 4 项，需要在 Coconut 中引入随机替换 latent 的机制：
- **方案 D：范数保持随机方向** (`norm_preserve`) —— 仅保留原始 latent 的 L2 范数，将其替换为一个随机方向的单位向量 × 同范数
- 数学定义：$\tilde z_j = \|h_j\|_2 \cdot \frac{\epsilon}{\|\epsilon\|_2},\ \epsilon \sim \mathcal N(0, I)$

目标是以最小的代码改动添加两个新配置参数：
1. `latent_randomize_mode` — 随机化方法 (`none` / `norm_preserve`)
2. `latent_randomize_mask` — 对哪些 latent 应用随机化（0/1 列表，如 `[1,0,0,0,0,1]`）

## 实施计划

### 步骤 1：修改 `coconut.py`

**文件**: `coconut.py`

1. 给 `Coconut.__init__` 增加两个参数：
   - `latent_randomize_mode: str = "none"` — 随机化模式
   - `latent_randomize_mask: List[int] = None` — 0/1 列表，默认 None 表示全 0（全不随机化）

2. 在 `__init__` 中初始化这两个属性

3. 在 `forward()` 方法中，**在第 150 行** `h = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]` 之后、`tensor_list[batch_idx][token_idx] = h` 之前，增加随机化逻辑：

```python
# 只在 eval 模式且 randomize_mode 不是 "none" 时应用
if (not self.training) and self.latent_randomize_mode != "none":
    # 判断当前 pass_idx 是否在 mask 中（mask 长度不足时默认 0）
    if pass_idx < len(self.latent_randomize_mask) and self.latent_randomize_mask[pass_idx] == 1:
        if self.latent_randomize_mode == "norm_preserve":
            eps = torch.randn_like(h)
            eps = eps / eps.norm(p=2).clamp_min(1e-6)
            h = h.norm(p=2).clamp_min(1e-6) * eps
```

4. 在 `generate()` 方法中，当调用 `self.forward()` 时，通过 `kwargs` 传递 `scheduled_stage`（目前 `forward()` 未使用此参数，但后续扩展可能需要）

### 步骤 2：修改 `run.py`

**文件**: `run.py`

1. 在配置加载部分（约第 59 行后）添加默认值：

```python
if not hasattr(configs, 'latent_randomize_mode'):
    configs.latent_randomize_mode = "none"
if not hasattr(configs, 'latent_randomize_mask'):
    configs.latent_randomize_mask = None
```

2. 在实例化 Coconut 模型时（约第 169 行），将新参数传入：

```python
if configs.coconut:
    model = Coconut(
        model, latent_id, start_id, end_id, tokenizer.eos_token_id,
        latent_randomize_mode=configs.latent_randomize_mode,
        latent_randomize_mask=configs.latent_randomize_mask,
    )
```

### 步骤 3：更新 YAML 配置示例

**文件**: `args/gsm_coconut_eval.yaml`

添加两个新参数用于评测时的随机化实验：

```yaml
latent_randomize_mode: norm_preserve
latent_randomize_mask: [1, 0, 0, 0, 0, 1]
```

## 关键文件

| 文件 | 改动内容 |
|------|---------|
| `coconut.py` | `__init__` 增加参数、`forward()` 增加随机化逻辑 |
| `run.py` | 配置默认值、Coconut 实例化传参 |
| `args/gsm_coconut_eval.yaml` | 添加配置示例 |

## 验证方案

1. **单元测试** — 在 eval 模式下运行 `gsm_coconut_eval.yaml`，确认随机化生效（输出结果应与基线不同）
2. **Mask 验证** — 设置 `latent_randomize_mask: [1,0,0,0,0,0]`，确认仅第一个 latent 被随机化
3. **模式切换** — 切换 `latent_randomize_mode: none`，确认输出与原始 deterministic Coconut 完全一致
4. **Debug 模式** — 配合 `debug_latent_k: 5` 检查 latent 解码 token 是否因随机化而改变
