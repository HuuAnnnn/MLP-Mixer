import argparse


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_args():
    parser = argparse.ArgumentParser(prog="MLP Mixer training")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The batch size of dataset"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./root",
        help="The destination path to save the data",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="32x32",
        help="The size of the input image",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="The size of train ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training on GPU/CPU",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="The initial learning rate"
    )
    parser.add_argument("--channels", type=int, default=3, help="The image's channels")
    parser.add_argument("--depth", type=int, default=4, help="MLP mixer depth")
    parser.add_argument("--hidden_dim", type=int, default=512, help="The hidden dim")
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="The size of each patch",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--token_dim", type=int, default=256, help="MLP dimension DC")
    parser.add_argument(
        "--channel_dim",
        type=int,
        default=2048,
        help="MLP dimension DS",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="The dropout rate",
    )

    return parser.parse_args()