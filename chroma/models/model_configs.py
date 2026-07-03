from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


def build_model_config(config_cls, values: Dict[str, Any]):
    field_names = {field.name for field in fields(config_cls)}
    config_values = {key: values[key] for key in field_names if key in values}
    extra_kwargs = {
        key: value for key, value in values.items() if key not in field_names
    }
    return config_cls(**config_values), extra_kwargs


def config_to_kwargs(config, extra_kwargs: Optional[Dict[str, Any]] = None):
    kwargs = asdict(config)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


@dataclass
class GraphDesignConfig:
    dim_nodes: int = 128
    dim_edges: int = 128
    num_neighbors: int = 30
    node_features: Tuple = ((("internal_coords", {"log_lengths": True})),)
    edge_features: Tuple = (
        "distances_2mer",
        "orientations_2mer",
        "distances_chain",
    )
    sequence_embedding: str = "linear"
    sidechain_embedding: str = "chi_rbf"
    sidechains: bool = True
    num_layers: int = 3
    num_layers_encoder: Optional[int] = None
    dropout: float = 0.1
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    edge_update: bool = True
    edge_mlp_layers: int = 1
    edge_mlp_dim: Optional[int] = None
    skip_connect_input: bool = False
    mlp_activation: str = "softplus"
    num_alphabet: int = 20
    num_chi_bins: int = 20
    decoder_num_hidden: int = 512
    label_smoothing: float = 0.1
    separate_packing: bool = True
    graph_criterion: str = "knn"
    graph_random_min_local: int = 20
    graph_attentional: bool = False
    graph_num_attention_heads: int = 4
    predict_S_marginals: bool = False
    predict_S_potts: bool = False
    potts_parameterization: str = "factor"
    potts_num_factors: Optional[int] = None
    potts_symmetric_J: bool = True
    noise_schedule: Optional[str] = None
    noise_covariance_model: str = "brownian"
    noise_complex_scaling: bool = False
    noise_beta_range: Tuple[float, float] = (0.2, 70.0)
    noise_log_snr_range: Tuple[float, float] = (-7.0, 13.5)
    checkpoint_gradients: bool = False


@dataclass
class BackboneEncoderGNNConfig:
    dim_nodes: int = 128
    dim_edges: int = 128
    num_neighbors: int = 30
    node_features: Tuple = ((("internal_coords", {"log_lengths": True})),)
    edge_features: Tuple = (
        "distances_2mer",
        "orientations_2mer",
        "distances_chain",
    )
    num_layers: int = 3
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    edge_update: bool = True
    edge_mlp_layers: int = 1
    edge_mlp_dim: Optional[int] = None
    skip_connect_input: bool = False
    mlp_activation: str = "softplus"
    dropout: float = 0.1
    graph_distance_atom_type: int = -1
    graph_cutoff: Optional[float] = None
    graph_mask_interfaces: bool = False
    graph_criterion: str = "knn"
    graph_random_min_local: int = 20
    checkpoint_gradients: bool = False


@dataclass
class SidechainDecoderGNNConfig:
    dim_nodes: int = 128
    dim_edges: int = 128
    num_neighbors: int = 30
    predict_S: bool = True
    predict_chi: bool = True
    sequence_embedding: str = "linear"
    sidechain_embedding: str = "mixed_chi_X"
    num_layers: int = 3
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    edge_update: bool = True
    edge_mlp_layers: int = 1
    edge_mlp_dim: Optional[int] = None
    skip_connect_input: bool = False
    mlp_activation: str = "softplus"
    dropout: float = 0.1
    num_alphabet: int = 20
    num_chi_bins: int = 20
    decoder_num_hidden: int = 512
    label_smoothing: float = 0.1
    checkpoint_gradients: bool = False


@dataclass
class GraphBackboneConfig:
    dim_nodes: int = 128
    dim_edges: int = 128
    num_neighbors: int = 30
    node_features: Tuple = ((("internal_coords", {"log_lengths": True})),)
    edge_features: Tuple = (
        "distances_2mer",
        "orientations_2mer",
        "distances_chain",
    )
    num_layers: int = 3
    dropout: float = 0.1
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    edge_update: bool = True
    edge_mlp_layers: int = 1
    edge_mlp_dim: Optional[int] = None
    skip_connect_input: bool = False
    mlp_activation: str = "softplus"
    decoder_num_hidden: int = 512
    graph_criterion: str = "knn"
    graph_random_min_local: int = 20
    backbone_update_method: str = "neighbor"
    backbone_update_iterations: int = 1
    backbone_update_num_weights: int = 1
    backbone_update_unconstrained: bool = True
    use_time_features: bool = True
    time_feature_type: str = "t"
    time_log_feature_scaling: float = 0.05
    noise_schedule: str = "log_snr"
    noise_covariance_model: str = "brownian"
    noise_beta_min: float = 0.2
    noise_beta_max: float = 70.0
    noise_log_snr_range: Tuple[float, float] = (-7.0, 13.5)
    noise_complex_scaling: bool = False
    loss_scale: float = 10.0
    loss_scale_ssnr_cutoff: float = 0.99
    loss_function: str = "squared_fape"
    checkpoint_gradients: bool = False
    prediction_type: str = "X0"
    num_graph_cycles: int = 1


@dataclass
class GraphClassifierConfig:
    dim_nodes: int = 128
    dim_edges: int = 128
    num_neighbors: int = 30
    node_features: Tuple = ((("internal_coords", {"log_lengths": True})),)
    edge_features: Tuple = (
        "random_fourier_2mer",
        "orientations_2mer",
        "distances_chain",
    )
    num_layers: int = 3
    dropout: float = 0.1
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    edge_update: bool = True
    edge_mlp_layers: int = 1
    edge_mlp_dim: Optional[int] = None
    skip_connect_input: bool = False
    mlp_activation: str = "softplus"
    graph_criterion: str = "knn"
    graph_random_min_local: int = 20
    use_time_features: bool = True
    noise_schedule: str = "log_snr"
    noise_beta_min: float = 0.2
    noise_beta_max: float = 70.0
    checkpoint_gradients: bool = False
    class_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    out_mlp_layers: int = 2
    noise_covariance_model: str = "globular"
    noise_log_snr_range: Tuple[float, float] = (-7.0, 13.5)
    time_feature_type: str = "t"
    time_log_feature_scaling: float = 0.05
    fourier_scale: float = 16.0
    zero_grad_fix: bool = False


@dataclass
class ProteinCaptionConfig:
    lm_id: str = "EleutherAI/gpt-neo-125m"
    gnn_dim_edges: int = 128
    context_size: int = 16
    context_per_chain: int = 1
    gnn_num_neighbors: int = 30
    gnn_num_layers: int = 3
    only_encode_caption_chain: bool = False
    gnn_embed_ratio: int = 1
    graph_criterion: str = "knn"
    node_mlp_layers: int = 1
    node_mlp_dim: Optional[int] = None
    noise_schedule: str = "log_snr"
    covariance_model: str = "globular"
    noise_complex_scaling: bool = False
    noiseless: bool = False
    normalize_context_embeddings: bool = False
    standardize_context_embeddings: bool = False
    time_feature_type: str = "t"
    time_log_feature_scaling: float = 0.05
    use_transformer: bool = False
    classifier_checkpoint: Optional[str] = None
    direct_gnn: bool = False
    classifier_kwargs: Optional[dict] = None
