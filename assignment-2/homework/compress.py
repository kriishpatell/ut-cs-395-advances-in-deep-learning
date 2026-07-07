from pathlib import Path
from typing import cast
from fractions import Fraction
import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive
 
    @staticmethod
    def _quantize(probs: torch.Tensor, bits: int = 16) -> torch.Tensor:
        total = 1 << bits
        freqs = (probs * total).floor().long().clamp(min=1)
        freqs[int(freqs.argmax())] += total - int(freqs.sum())
        return freqs
 
    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        device = next(self.autoregressive.parameters()).device
        idx = self.tokenizer.encode_index(x[None].to(device))[0]
        h, w = idx.shape
        seq = idx.flatten().tolist()
        with torch.inference_mode():
            logits, _ = self.autoregressive(idx[None])
        all_probs = torch.softmax(logits[0], dim=-1).view(h * w, -1).cpu()
 
        low, high = Fraction(0), Fraction(1)
        for i, tok in enumerate(seq):
            cum = self._quantize(all_probs[i]).cumsum(0)
            lo = 0 if tok == 0 else int(cum[tok - 1])
            hi, tot = int(cum[tok]), int(cum[-1])
            rng = high - low
            high = low + rng * hi / tot
            low = low + rng * lo / tot
 
        k = 24 * len(seq) + 32
        p = -(-(low.numerator << k) // low.denominator)  
        nbytes = (k + 7) // 8
        p <<= nbytes * 8 - k
        return p.to_bytes(nbytes, "big")
 
    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        device = next(self.autoregressive.parameters()).device
        ps = self.tokenizer.patch_size
        h, w = 100 // ps, 150 // ps
        val = Fraction(int.from_bytes(x, "big"), 1 << (8 * len(x)))
 
        low, high = Fraction(0), Fraction(1)
        grid = torch.zeros(1, h, w, dtype=torch.long, device=device)
        flat = grid.view(-1)
        for i in range(h * w):
            with torch.inference_mode():
                logits, _ = self.autoregressive(grid)
            probs = torch.softmax(logits[0].view(h * w, -1)[i], dim=-1).cpu()
            cum = self._quantize(probs).cumsum(0)
            tot = int(cum[-1])
            rng = high - low
            target = int((val - low) / rng * tot)
            tok = int(torch.searchsorted(cum, target, right=True))
            lo = 0 if tok == 0 else int(cum[tok - 1])
            hi = int(cum[tok])
            low, high = low + rng * lo / tot, low + rng * hi / tot
            flat[i] = tok
        return self.tokenizer.decode_index(grid)[0].cpu()


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
