import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.preprocess import load_battery_sequence
from model.model import DeepSCBattery
from performance import evaluate
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/00002.csv", type=str)
    parser.add_argument("--window-size", default=30, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    args = parser.parse_args()

    X, scaler = load_battery_sequence(args.data_path, args.window_size)
    dataset = TensorDataset(torch.tensor(X))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cpu")
    model = DeepSCBattery().to(device)
    train(model, train_loader, num_epochs=args.epochs, lr=args.lr)
    evaluate(model, X, scaler)


if __name__ == '__main__':
    main()
