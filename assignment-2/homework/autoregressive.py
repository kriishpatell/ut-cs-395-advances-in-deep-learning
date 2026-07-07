import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.start_token = torch.nn.Parameter(torch.zeros(d_latent))
        layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=4)
        self.output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        emb = self.embedding(x.view(B, h * w))
        seq = torch.cat([self.start_token[None, None].expand(B, 1, -1), emb[:, :-1]], dim=1)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)
        logits = self.output(self.transformer(seq, mask=mask, is_causal=True))
        return logits.view(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        L = h * w
        seq = torch.zeros(B, 0, dtype=torch.long, device=device)
        cur = self.start_token[None, None].expand(B, 1, -1).to(device)
        for _ in range(L):
            mask = torch.nn.Transformer.generate_square_subsequent_mask(cur.size(1)).to(device)
            logits = self.output(self.transformer(cur, mask=mask, is_causal=True)[:, -1])
            nxt = torch.distributions.Categorical(logits=logits).sample()
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            cur = torch.cat([cur, self.embedding(nxt)[:, None]], dim=1)
        return seq.view(B, h, w)