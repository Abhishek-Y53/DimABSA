import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from scipy.stats import pearsonr
import torch.nn as nn



class DimASRModel(nn.Module):
    def __init__(self):
        super(DimASRModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regressor(cls_output)

        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, labels)
            return loss, predictions

        return predictions


#  Data loading


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_aspect_level_samples(raw_data):
    samples = []

    for item in raw_data:
        sample_id = item["ID"]
        text = item["Text"]

        if "Quadruplet" in item:
            for q in item["Quadruplet"]:
                v, a = q["VA"].split("#")
                samples.append({
                    "id": sample_id,
                    "text": text,
                    "aspect": q["Aspect"],
                    "valence": float(v),
                    "arousal": float(a)
                })

        elif "Triplet" in item:
            for t in item["Triplet"]:
                v, a = t["VA"].split("#")
                samples.append({
                    "id": sample_id,
                    "text": text,
                    "aspect": t["Aspect"],
                    "valence": float(v),
                    "arousal": float(a)
                })

        elif "Aspect_VA" in item:
            for av in item["Aspect_VA"]:
                v, a = av["VA"].split("#")
                samples.append({
                    "id": sample_id,
                    "text": text,
                    "aspect": av["Aspect"],
                    "valence": float(v),
                    "arousal": float(a)
                })

        else:
            raise ValueError(f"Unknown or label-less format in item {sample_id}")

    return samples


#  Dataset & DataLoader

def build_model_input(text, aspect):
    if aspect == "NULL":
        aspect = "overall"
    return text.strip() + " [SEP] " + aspect.strip()


class DimASRDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model_input = build_model_input(sample["text"], sample["aspect"])

        encoding = self.tokenizer(
            model_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                [sample["valence"], sample["arousal"]],
                dtype=torch.float
            )
        }


#  Evaluation
def evaluate_cross_domain(model, dataloader, device):
    """
    Runs inference and computes:
      - RMSE_VA  (combined valence + arousal, as per SemEval DimASR formula)
      - RMSE_V   (valence only)
      - RMSE_A   (arousal only)
      - Pearson r for valence
      - Pearson r for arousal
    """
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.clamp(preds, min=1.0, max=9.0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)   # (N, 2)
    all_labels = np.vstack(all_labels)  # (N, 2)

    Vp, Ap = all_preds[:, 0],  all_preds[:, 1]
    Vg, Ag = all_labels[:, 0], all_labels[:, 1]

    N = len(Vp)

    # SemEval combined RMSE_VA
    rmse_va = np.sqrt(np.sum((Vp - Vg)**2 + (Ap - Ag)**2) / N)

    # Individual RMSE
    rmse_v  = np.sqrt(np.mean((Vp - Vg)**2))
    rmse_a  = np.sqrt(np.mean((Ap - Ag)**2))

    # Pearson correlations
    pearson_v, _ = pearsonr(Vp, Vg)
    pearson_a, _ = pearsonr(Ap, Ag)

    return {
        "RMSE_VA":   rmse_va,
        "RMSE_V":    rmse_v,
        "RMSE_A":    rmse_a,
        "Pearson_V": pearson_v,
        "Pearson_A": pearson_a,
        "N":         N
    }


# Main


if __name__ == "__main__":

    MODEL_PATH       = "best_model_restraunt.pt"
    RESTAURANT_TRAIN = "eng_laptop_train_alltasks.jsonl"
    BATCH_SIZE       = 16
    MAX_LENGTH       = 128


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load tokenizer & model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = DimASRModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model weights from '{MODEL_PATH}'")

    # Load restaurant data
    raw_data = read_jsonl(RESTAURANT_TRAIN)
    samples  = extract_aspect_level_samples(raw_data)
    print(f"Loaded {len(samples)} aspect-level samples from '{RESTAURANT_TRAIN}'")

    # Build dataset & loader
    dataset = DimASRDataset(samples, tokenizer, max_length=MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate
    metrics = evaluate_cross_domain(model, loader, device)

    print("\n" + "="*45)
    print("  Cross-Domain Evaluation: Restaurant Train")
    print("="*45)
    print(f"  Instances  : {metrics['N']}")
    print(f"  RMSE_VA    : {metrics['RMSE_VA']:.4f}  ← primary SemEval metric")
    print(f"  RMSE_V     : {metrics['RMSE_V']:.4f}")
    print(f"  RMSE_A     : {metrics['RMSE_A']:.4f}")
    print(f"  Pearson_V  : {metrics['Pearson_V']:.4f}")
    print(f"  Pearson_A  : {metrics['Pearson_A']:.4f}")
    print("="*45)


"""
    TRAIN DATA of restraunt on laptop model
    Instances: 3659
    RMSE_VA: 1.3515  ← primary
    SemEval
    metric
    RMSE_V: 1.0385
    RMSE_A: 0.8649
    Pearson_V: 0.8312
    Pearson_A: 0.6550
"""

"""
TRAIN DATA of laptop on restraunt model
 Instances  : 5773
  RMSE_VA    : 1.2115  ← primary SemEval metric
  RMSE_V     : 0.9175
  RMSE_A     : 0.7911
  Pearson_V  : 0.8681
  Pearson_A  : 0.6657
"""
