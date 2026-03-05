import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# ---------- 1) Your prediction function ----------
def GetPrediction(model, dataloader, device):
    model = model.to(device)
    model.eval()

    n = len(dataloader.dataset)
    preds = np.empty(n, dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)

    k = 0
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(device, dtype=torch.float32)
            target = target.to(device)

            logits = model(inputs)  # (B, num_classes)
            batch_preds = torch.argmax(logits, dim=1)

            bsz = inputs.size(0)
            preds[k:k+bsz] = batch_preds.cpu().numpy()
            labels[k:k+bsz] = target.cpu().numpy()
            k += bsz

    return preds, labels

def GetProbsAndLabels(model, dataloader, device):
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            logits = model(inputs)                 # (B, 10)
            probs  = F.softmax(logits, dim=1)      # (B, 10)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_probs, axis=0)            # (N, 10)
    y_test  = np.concatenate(all_labels, axis=0).astype(int)  # (N,)
    return y_test, y_score

def LoadTestData(npz_path, batch_size=64):
    data = np.load(npz_path)
    X_test = torch.from_numpy(data["X_test"]).float()  # (N, 1, 20, 10)
    y_test = torch.from_numpy(data["y_test"]).long()   # (N,)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def LoadLiParTestData(npz_path, batch_size=256, num_workers=4, pin_memory=True):
    data = np.load(npz_path)

    X_img_test = torch.from_numpy(data["X_img_test"]).float()  # (N,3,9,9)
    X_seq_test = torch.from_numpy(data["X_seq_test"]).float()  # (N,27,9)
    y_test     = torch.from_numpy(data["y_test"]).long()       # (N,)

    test_dataset = TensorDataset(X_img_test, X_seq_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    return test_loader

def LiParGetProbsAndLabels(model, dataloader, device):
    """
    LiPar dataloader must yield: (x_img, x_seq, labels)
      x_img: (B,3,9,9)
      x_seq: (B,27,9)
      labels: (B,)
    returns:
      y_test:  (N,) int
      y_score: (N, num_classes) float (softmax probs)
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_img, x_seq, labels in dataloader:
            x_img = x_img.to(device, non_blocking=True).float()
            x_seq = x_seq.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            logits = model(x_img, x_seq)          # (B, num_classes)
            probs  = F.softmax(logits, dim=1)     # (B, num_classes)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_probs, axis=0)              # (N, num_classes)
    y_test  = np.concatenate(all_labels, axis=0).astype(int) # (N,)
    return y_test, y_score

def LiParGetPrediction(model, dataloader, device):
    """
    LiPar dataloader must yield: (x_img, x_seq, target)
      x_img: (B,3,9,9)
      x_seq: (B,27,9)
      target: (B,)
    returns:
      preds:  (N,) int64
      labels: (N,) int64
    """
    model = model.to(device)
    model.eval()

    n = len(dataloader.dataset)
    preds  = np.empty(n, dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)

    k = 0
    with torch.no_grad():
        for x_img, x_seq, target in dataloader:
            x_img  = x_img.to(device, dtype=torch.float32, non_blocking=True)
            x_seq  = x_seq.to(device, dtype=torch.float32, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(x_img, x_seq)              # (B, num_classes)
            batch_preds = torch.argmax(logits, dim=1) # (B,)

            bsz = target.size(0)
            preds[k:k+bsz]  = batch_preds.cpu().numpy()
            labels[k:k+bsz] = target.cpu().numpy()
            k += bsz

    return preds, labels