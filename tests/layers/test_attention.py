import torch

from chroma.layers.attention import AttentionChainPool, MultiHeadAttention, ScaledDotProductAttention


def _legacy_multi_head_attention(module, Q, K, V, mask=None):
    mb_size = Q.size(0)
    sdp = ScaledDotProductAttention()

    q_s = torch.cat([Q @ W for W in module.Wq])
    k_s = torch.cat([K @ W for W in module.Wk])
    v_s = torch.cat([V @ W for W in module.Wv])

    if mask is not None:
        mask = mask.repeat(module.n_head, 1, 1)

    outputs, attns = sdp(q_s, k_s, v_s, mask=mask)
    outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
    outputs = module.dropout(outputs @ module.Wo)
    return outputs, attns


def _legacy_attention_chain_pool(module, h, C):
    chains = C.abs().unique()
    chains = chains[chains > 0]
    num_chains = len(chains.unique())
    output = []
    chain_mask = []
    for chain_id in chains:
        mask = (C.abs() == chain_id).unsqueeze(1)
        output.append(module.attention(module.get_query(h), h, h, mask=mask))
        chain_mask.append(mask.any(dim=-1))
    output = torch.cat(output, dim=1)
    chain_mask = torch.cat(chain_mask, dim=1)
    return output, chain_mask


def test_multi_head_attention_matches_legacy_path():
    torch.manual_seed(7)
    module = MultiHeadAttention(n_head=3, d_k=4, d_v=5, d_model=8, dropout=0.0)

    Q = torch.randn(2, 6, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [[[True, True, False, True]], [[True, False, True, True]]]
    )

    expected_output, expected_attns = _legacy_multi_head_attention(
        module, Q, K, V, mask=mask
    )
    actual_output, actual_attns = module(Q, K, V, mask=mask)

    # Batched einsum projection vs. the legacy per-head loop are only
    # equivalent up to floating-point reduction order, which differs across
    # BLAS backends, so use a looser-than-default tolerance.
    assert torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-5)
    assert torch.allclose(actual_attns, expected_attns, rtol=1e-4, atol=1e-5)


def test_attention_chain_pool_matches_legacy_path():
    torch.manual_seed(11)
    pool = AttentionChainPool(n_head=2, d_model=6)

    h = torch.randn(2, 5, 6)
    C = torch.tensor([[1, 1, 2, 2, 0], [1, 3, 3, 0, 0]])

    expected_output, expected_chain_mask = _legacy_attention_chain_pool(pool, h, C)
    actual_output, actual_chain_mask = pool(h, C)

    assert torch.allclose(actual_output, expected_output)
    assert torch.equal(actual_chain_mask, expected_chain_mask)
    assert actual_output.shape == (2, 3, 6)
    assert torch.equal(
        actual_chain_mask,
        torch.tensor([[True, True, False], [True, False, True]]),
    )
