import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from rich.progress import Progress, MofNCompleteColumn

from model import Rater
from generate_dataset import generate_dataset, batch_dataset
from stringdsl import stringdsl
from stringprops import llProps
from bustle import propertySignature, propertySignatureSize

It = ("str",)
Ot = "str"
dsl = stringdsl
dataset = batch_dataset(generate_dataset(), llProps)


def initialModel(key):
    print("initial model called for", key)
    (It, Ot, Vt) = key
    return Rater(
        propertySignatureSize(It, Ot, llProps)
        + propertySignatureSize((Vt,), Ot, llProps)
    )


import shutil
import subprocess
import datetime


def saveModel(Ms):
    git_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y-%m-%dT%H:%M")

    key = timestamp + "-" + git_id

    fn_key = "models/rater_%s.pt"
    fn = fn_key % key
    torch.save(Ms, fn)
    shutil.copy(fn, fn_key % "latest")


Ms = {}
optimizers = {}
loss = BCELoss()

print()
for epoch in range(100):
    print(f"Epoch {epoch + 1}")
    Ts = {}
    for key in dataset:
        M = Ms.get(key)
        optimizer = optimizers.get(key)
        if M is None:
            M = initialModel(key)
            Ms[key] = M
            optimizer = torch.optim.Adam(M.parameters(), lr=0.001)
            optimizers[key] = optimizer
        train_losses = Ts.get(key)
        if train_losses is None:
            train_losses = []
            Ts[key] = train_losses
        M.train()

        with Progress(
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
        ) as progress:
            for i, (x, y) in enumerate(
                progress.track(
                    DataLoader(dataset[key], batch_size=64, shuffle=True),
                    description=f"Model {key[-1].ljust(4)}",
                )
            ):
                optimizer.zero_grad()

                outputs = M(x)
                loss_v = loss(outputs, y)
                (It, Ot, Vt) = key
                if i == 0:
                    print("loss for ", Vt, ": ", loss_v.item(), sep="")
                loss_v.backward()
                optimizer.step()
                train_losses.append(loss_v.item())
    saveModel(Ms)
    print()
