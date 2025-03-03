import torch
from torch.utils.data import Dataset


# 6. Custom Dataset and DataLoader
class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        if item['structures'] is not None:
            target = torch.tensor(item['structures'][0], dtype=torch.float32)
            return features, target, item['id']
        else:
            return features, None, item['id']


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    max_length = features[0].shape[0]
    feature_dim = features[0].shape[1]
    padded_features = []
    padded_targets = []
    for i, feature in enumerate(features):
        length = feature.shape[0]
        padding = torch.zeros(
            (max_length - length, feature_dim), dtype=torch.float32)
        padded_feature = torch.cat([feature, padding], dim=0)
        padded_features.append(padded_feature)
        if targets[i] is not None:
            target_padding = torch.zeros(
                (max_length - length, 3), dtype=torch.float32)
            padded_target = torch.cat([targets[i], target_padding], dim=0)
            padded_targets.append(padded_target)
    features_tensor = torch.stack(padded_features)
    if all(target is not None for target in targets):
        targets_tensor = torch.stack(padded_targets)
        return features_tensor, targets_tensor, ids, [f.shape[0] for f in features]
    else:
        return features_tensor, None, ids, [f.shape[0] for f in features]
