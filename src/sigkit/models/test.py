# noqa
import click
import numpy as np
import torch

from sigkit.models.Module import SigKitClassifier
from sigkit.models.utils import get_class_index
from sigkit.modem.fsk import FSK
from sigkit.modem.psk import PSK
from sigkit.transforms.awgn import ApplyAWGN
from sigkit.transforms.utils import InferenceTransform

SAMPLE_RATE = 1024
SYMBOL_RATE = 32


@click.command()
@click.option("-n", "--n_signals", default=32)
def main(n_signals):
    ckpt_path = "data/checkpoints/best.ckpt"
    model = SigKitClassifier.load_from_checkpoint(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    modems = []

    for modulator in [PSK, FSK]:
        for n_components in [2, 4, 8, 16]:
            modems.append(modulator(SAMPLE_RATE, SYMBOL_RATE, n_components))

    for modem in modems:
        num_symbols = 4096 // modem.sps
        bitstreams = [
            np.random.randint(
                0, 2, size=(num_symbols * modem.bits_per_symbol,), dtype=np.uint8
            )
            for _ in range(n_signals)
        ]
        tensors = [modem.modulate(bits).to_tensor() for bits in bitstreams]

        transform = ApplyAWGN((-20, 30))
        tensors = [transform(sig) for sig in tensors]

        x = torch.stack([InferenceTransform(sig) for sig in tensors]).to(device)
        with torch.no_grad():
            preds = model(x)
            predicted = torch.argmax(preds, dim=1)
            acc = (predicted == get_class_index(modem.__label__())).float().mean()

        print(f"{modem.__label__()}: {acc*100:2f}%")


if __name__ == "__main__":
    main()
