import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def evaluate_classifier(model, dataloader, device):
    """Оцінює NLP модель класифікації за комплексом метрик."""
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_score(all_targets, all_preds),
        "AUC": roc_auc_score(all_targets, all_probs),
        "Precision": precision_score(all_targets, all_preds, zero_division=0),
        "Recall": recall_score(all_targets, all_preds, zero_division=0),
        "F1-Score": f1_score(all_targets, all_preds, zero_division=0)
    }
    return metrics
