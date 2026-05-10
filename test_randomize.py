"""Test latent randomization: norm_preserve mode and mask behavior."""
import torch
import torch.nn as nn
from coconut import Coconut


class FakeCausalLM(nn.Module):
    """Minimal model that Coconut treats as non-GPT2 (uses get_input_embeddings)."""
    def __init__(self, hidden_size=128, vocab_size=1000, n_heads=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._hs = hidden_size
        self._n_heads = n_heads

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None,
                past_key_values=None, output_hidden_states=False, **kw):
        bs, seq_len, hs = inputs_embeds.shape
        logits = self.lm_head(inputs_embeds)
        head_dim = hs // self._n_heads
        pkv = [(torch.randn(bs, self._n_heads, seq_len, head_dim),
                torch.randn(bs, self._n_heads, seq_len, head_dim))
               for _ in range(self._n_heads)]
        class Out:
            pass
        o = Out()
        o.logits = logits
        o.hidden_states = (inputs_embeds,)
        o.past_key_values = pkv
        return o


def _make_coconut(mode="none", mask=None):
    model = FakeCausalLM()
    return Coconut(model, latent_token_id=900, start_latent_id=901,
                   end_latent_id=902, eos_token_id=50256,
                   latent_randomize_mode=mode,
                   latent_randomize_mask=mask)


def _run_forward(coconut, n_latent=2):
    ids = [1] + [900] * n_latent + [2]
    input_ids = torch.tensor([ids])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(len(ids)).unsqueeze(0)
    return coconut.forward(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels, position_ids=position_ids)


def test_norm_preserve_math():
    """Verify formula: ||z_tilde|| == ||h||, direction is random."""
    torch.manual_seed(0)
    h = torch.randn(768)
    original_norm = h.norm(p=2).item()
    eps = torch.randn_like(h)
    eps = eps / eps.norm(p=2).clamp_min(1e-6)
    h_rand = h.norm(p=2).clamp_min(1e-6) * eps
    assert abs(original_norm - h_rand.norm(p=2).item()) < 1e-4
    cos = torch.nn.functional.cosine_similarity(h.unsqueeze(0), h_rand.unsqueeze(0)).item()
    assert abs(cos) < 0.99
    print(f"[PASS] Norm preserved: {original_norm:.6f} ≈ {h_rand.norm(p=2).item():.6f}, cos_sim={cos:.4f}")


def test_norm_preserve_forward_runs():
    coconut = _make_coconut(mode="norm_preserve", mask=[1, 1, 1])
    coconut.eval()
    out = _run_forward(coconut, n_latent=2)
    assert out.inputs_embeds is not None
    assert out.loss is not None
    print("[PASS] norm_preserve forward runs without error")


def test_mode_none_deterministic():
    coconut = _make_coconut(mode="none", mask=[1, 1, 1])
    coconut.eval()
    torch.manual_seed(99); out1 = _run_forward(coconut)
    torch.manual_seed(123); out2 = _run_forward(coconut)
    assert torch.allclose(out1.inputs_embeds, out2.inputs_embeds, atol=1e-6)
    print("[PASS] mode='none' is deterministic")


def test_norm_preserve_changes_output():
    coconut = _make_coconut(mode="norm_preserve", mask=[1, 1])
    coconut.eval()
    torch.manual_seed(42); out1 = _run_forward(coconut)
    torch.manual_seed(999); out2 = _run_forward(coconut)
    assert not torch.allclose(out1.inputs_embeds, out2.inputs_embeds, atol=1e-6)
    print("[PASS] norm_preserve produces different outputs across runs")


def test_training_mode_skips_randomization():
    coconut = _make_coconut(mode="norm_preserve", mask=[1, 1, 1])
    coconut.train()
    torch.manual_seed(99); out1 = _run_forward(coconut)
    torch.manual_seed(123); out2 = _run_forward(coconut)
    assert torch.allclose(out1.inputs_embeds, out2.inputs_embeds, atol=1e-6)
    print("[PASS] Training mode skips randomization")


def test_mask_selective():
    coconut = _make_coconut(mode="norm_preserve", mask=[1, 0])
    coconut.eval()
    torch.manual_seed(42); out1 = _run_forward(coconut)
    torch.manual_seed(999); out2 = _run_forward(coconut)
    assert not torch.allclose(out1.inputs_embeds, out2.inputs_embeds, atol=1e-6)
    print("[PASS] mask=[1,0] runs correctly")


def test_empty_mask():
    coconut = _make_coconut(mode="norm_preserve", mask=[])
    coconut.eval()
    torch.manual_seed(42); out1 = _run_forward(coconut)
    torch.manual_seed(999); out2 = _run_forward(coconut)
    assert torch.allclose(out1.inputs_embeds, out2.inputs_embeds, atol=1e-6)
    print("[PASS] Empty mask → deterministic")


if __name__ == "__main__":
    tests = [
        test_norm_preserve_math,
        test_norm_preserve_forward_runs,
        test_mode_none_deterministic,
        test_norm_preserve_changes_output,
        test_training_mode_skips_randomization,
        test_mask_selective,
        test_empty_mask,
    ]
    results = []
    for t in tests:
        try:
            t()
            results.append((t.__name__, True))
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            results.append((t.__name__, False))

    print(f"\n{'='*50}")
    passed = sum(1 for _, ok in results if ok)
    print(f"Results: {passed}/{len(results)} passed")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
