from .seg_rnn   import Model as _SegRNN,   SegRNNConfigs
from .seg_mamba import Model as _SegMamba, SegMambaConfigs


def get_configs(model_name: str):
    """Return model-specific configs instance."""
    key = model_name.lower()
    if key == "segrnn":   return SegRNNConfigs()
    if key == "segmamba": return SegMambaConfigs()
    raise ValueError(f"Unknown model '{model_name}'. Available: segrnn, segmamba")


def build_model(name: str, configs):
    """Build model from modelzoo by name."""
    key = name.lower()

    if key == "segrnn":
        return _SegRNN(configs)

    if key == "segmamba":
        return _SegMamba(
            H=configs.H,
            L=configs.L,
            enc_in=configs.enc_in,
            num_layer=configs.num_layer,
            dropout=configs.dropout,
            seg_len=configs.seg_len,
            channel_id=configs.channel_id,
            revin=configs.revin,
            D_STATE=configs.d_state,
            DCONV=configs.d_conv,
            E_FACT=configs.e_factor,
        )

    raise ValueError(f"Unknown model '{name}'. Available: segrnn, segmamba")
