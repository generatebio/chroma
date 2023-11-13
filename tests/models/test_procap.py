import tempfile

import torch

from chroma.models.procap import ProteinCaption, load_model, save_model


def test_procap():
    model = ProteinCaption(
        lm_id="EleutherAI/gpt-neo-125m",
        gnn_dim_edges=16,
        context_size=8,
        context_per_chain=1,
        gnn_num_neighbors=4,
        gnn_num_layers=1,
    )

    assert sum(p.numel() for p in model.parameters()) == 128839584
    X = torch.randn(1, 8, 4, 3)
    C = torch.ones(X.shape[:2])
    caption = ["test caption"]
    chain_id = torch.tensor([1])
    with torch.no_grad():
        logits = model(X, C, caption, chain_id).logits
        assert logits.shape == torch.Size([1, 11, 50260])
    temp = tempfile.NamedTemporaryFile()
    save_model(model, temp.name)
    del model
    model = load_model(temp.name)
    assert sum(p.numel() for p in model.parameters()) == 128839584
    temp.close()
