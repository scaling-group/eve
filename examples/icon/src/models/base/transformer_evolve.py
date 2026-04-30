from src.models.base.transformer import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderEvolve(TransformerEncoder):
    """No-op encoder subclass reserved for Evolve mutations."""

    pass


class TransformerEncoderLayerEvolve(TransformerEncoderLayer):
    """No-op encoder-layer subclass reserved for Evolve mutations."""

    pass
