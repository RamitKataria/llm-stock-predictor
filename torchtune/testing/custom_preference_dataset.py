from torch.utils.data import Dataset

class CustomPreferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'news': row['News'],
            'date': row['Date'],
            'sentiment': row['Sentiment'],
            'industry': row['Industry'],
            'positive_changes': row['positive_changes'],
            'negative_changes': row['negative_changes']
        }
