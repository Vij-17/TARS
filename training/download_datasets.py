"""Download and prepare datasets for TARS training"""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

def download_squad():
    """Download SQuAD dataset for reading comprehension"""
    print("\nDownloading SQuAD dataset...")
    squad = load_dataset("squad")
    return squad

def download_ropes():
    """Download ROPES dataset for reasoning over paragraphs"""
    print("\nDownloading ROPES dataset...")
    ropes = load_dataset("ropes")
    return ropes

def download_synthetic():
    """Download Synthetic Reasoning dataset"""
    print("\nDownloading Synthetic Reasoning dataset...")
    synthetic = load_dataset("tasksource/synthetic-reasoning")
    return synthetic

class ReasoningDataset(Dataset):
    """Combined reasoning dataset"""
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = sum(len(d['train']) for d in datasets.values())
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # TODO: Implement proper dataset combination
        pass

def main():
    print("="*60)
    print("Downloading Training Datasets")
    print("="*60)
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download all datasets
    datasets = {}
    try:
        datasets['squad'] = download_squad()
        print("✅ SQuAD dataset downloaded")
    except Exception as e:
        print(f"⚠️ Error downloading SQuAD: {e}")
    
    try:
        datasets['ropes'] = download_ropes()
        print("✅ ROPES dataset downloaded")
    except Exception as e:
        print(f"⚠️ Error downloading ROPES: {e}")
    
    try:
        datasets['synthetic'] = download_synthetic()
        print("✅ Synthetic Reasoning dataset downloaded")
    except Exception as e:
        print(f"⚠️ Error downloading Synthetic Reasoning: {e}")
    
    if not datasets:
        print("\n❌ No datasets could be downloaded!")
        return
    
    print("\nDataset statistics:")
    for name, dataset in datasets.items():
        print(f"\n{name.upper()}:")
        for split in dataset:
            print(f"  {split}: {len(dataset[split])} examples")
    
    # Save dataset info
    with open(os.path.join(data_dir, "dataset_info.txt"), "w") as f:
        f.write("TARS Training Datasets\n")
        f.write("="*60 + "\n\n")
        for name, dataset in datasets.items():
            f.write(f"{name.upper()}:\n")
            for split in dataset:
                f.write(f"  {split}: {len(dataset[split])} examples\n")
            f.write("\n")
    
    print("\n✅ All available datasets downloaded successfully!")
    print(f"Dataset info saved to: {os.path.join(data_dir, 'dataset_info.txt')}")

if __name__ == "__main__":
    main()