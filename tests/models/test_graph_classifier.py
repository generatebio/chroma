from unittest import TestCase

import torch

from chroma.models.graph_classifier import GraphClassifier


class TestGraphClassifier(TestCase):
    def test_graph_classifier(self):
        class_config = {
            "dummy_1": {
                "tokens": ["a", "b", "c", "d"],
                "loss": "bce",
                "level": "chain",
            },
            "dummy_2": {
                "tokens": ["w", "x", "y", "z"],
                "loss": "ce",
                "level": "first_order",
            },
            "dummy_3": {
                "tokens": ["alpha", "beta"],
                "loss": "ce",
                "level": "complex",
            },
        }
        for k, v in class_config.items():
            v["tokenizer"] = {k: i for i, k in enumerate(v["tokens"])}

        model = GraphClassifier(
            dim_nodes=16,
            dim_edges=16,
            edge_mlp_dim=8,
            node_mlp_dim=8,
            class_config=class_config,
        )

        bs = 1
        sl = 8

        X = torch.randn(bs, sl, 4, 3)
        C = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long)

        with torch.no_grad():
            node_h, edge_h = model(X, C)

        self.assertTrue(node_h.size() == torch.Size([bs, sl, 16]))

        grad = model.gradient(X, C, t=0.5, label="dummy_2", value="w")
        self.assertTrue(grad.size() == torch.Size([bs, sl, 4, 3]))

        grad_complex = model.gradient(X, C, t=0.5, label="dummy_3", value="alpha")
        self.assertTrue(grad_complex.size() == torch.Size([bs, sl, 4, 3]))

    def test_graph_classifier_default_class_config_not_shared(self):
        model_a = GraphClassifier(dim_nodes=8, dim_edges=8, edge_mlp_dim=4, node_mlp_dim=4)
        model_b = GraphClassifier(dim_nodes=8, dim_edges=8, edge_mlp_dim=4, node_mlp_dim=4)

        model_a.class_config["dummy"] = {"tokens": []}

        self.assertEqual(model_b.class_config, {})
