import torch
from dataset import load_data
from utils import load_args, EarlyStopper
from mlp_mixer.mixer import MLPMixer
from torcheval.metrics import MulticlassAccuracy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

args = load_args()
print(f"Using device {args.device} -> {torch.cuda.get_device_name(args.device)}")


(
    classes,
    train_loader,
    val_loader,
    test_loader,
) = load_data(
    args.save_path,
    args.batch_size,
    args.train_ratio,
)

num_classes = len(classes)
image_size = tuple(int(num) for num in args.image_size.split("x"))
model = MLPMixer(
    args.channels,
    image_size,
    args.patch_size,
    args.hidden_dim,
    args.depth,
    num_classes,
    args.token_dim,
    args.channel_dim,
    args.dropout_rate,
)

model = model.to(args.device)
num_params = sum(layer.numel() for layer in model.parameters())
print(f"Number of parameters: {num_params:,}")


optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()
early_stopping = EarlyStopper(patience=20)

history = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "val_acc": []}

for epoch in range(1, args.epochs + 1):
    epoch_pbar = tqdm(range(0, len(train_loader)), desc=f"Epoch {epoch}/{args.epochs}")
    epoch_train_loss = 0
    epoch_val_loss = 0
    train_acc = MulticlassAccuracy()
    val_acc = MulticlassAccuracy()

    for sample in train_loader:
        images, actual = sample
        images = images.to(args.device)
        actual = actual.to(args.device)

        pred = model(images.float())
        optimizer.zero_grad()
        _, pred_labels = torch.max(pred, dim=1)
        train_acc.update(actual, pred_labels)
        loss = criterion(pred, actual)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_pbar.update(1)
    epoch_train_loss /= len(train_loader)

    with torch.no_grad():
        for sample in val_loader:
            images, actual = sample
            images = images.to(args.device)
            actual = actual.to(args.device)
            pred = model(images.float())
            _, pred_labels = torch.max(pred, dim=1)
            val_acc.update(actual, pred_labels)
            loss = criterion(pred, actual)
            epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)

    history["epoch"].append(epoch)
    history["val_loss"].append(epoch_val_loss)
    history["loss"].append(epoch_train_loss)
    history["val_acc"].append(val_acc.compute().item())
    history["acc"].append(train_acc.compute().item())

    epoch_pbar.set_postfix(
        {
            "loss": f"{history['loss'][-1]:.4f}",
            "val_loss": f"{history['val_loss'][-1]:.4f}",
            "acc": f"{history['acc'][-1]:.4f}",
            "val_acc": f"{history['val_acc'][-1]:.4f}",
        }
    )

    if early_stopping(epoch_val_loss):
        print(f"The model stop at {epoch}")
        break

    epoch_pbar.close()


history_df = pd.DataFrame(history)

plt.plot(history_df["epoch"], history_df["loss"])
plt.plot(history_df["epoch"], history_df["val_loss"])
plt.legend(["loss", "val_loss"])
plt.savefig("TrainLoss.png", dpi=300, format="png")


plt.plot(history_df["epoch"], history_df["acc"])
plt.plot(history_df["epoch"], history_df["val_acc"])
plt.legend(["acc", "val_acc"])

plt.savefig("TrainAcc.png", dpi=300, format="png")

acc = MulticlassAccuracy()

for sample in test_loader:
    images, actual = sample
    images = images.to(args.device)
    actual = actual.to(args.device)
    pred = model(images.float())
    _, pred_labels = torch.max(pred, dim=1)
    acc.update(actual, pred_labels)

print(f"Test accuracy: {acc.compute().item():.4f}")
