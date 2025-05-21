import argparse

import torch
from preprocess import load_battery_sequence
from torch.utils.data import DataLoader, TensorDataset

from model import DeepSCBattery
from performance import evaluate
from train import train


def main():
    # 하이퍼 파라미터 설정 및 데이터 경로 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/data.csv', type=str)
    parser.add_argument('--meta-path', default='data/metadata.csv', type=str)
    parser.add_argument('--window-size', default=50, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
    args = parser.parse_args()

    # 데이터 로딩 및 전처리
    X, Y, fscaler, tscaler = load_battery_sequence(args.data_path, args.meta_path, args.window_size)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 모델 초기화 및 학습
    model = DeepSCBattery().cuda()
    train(model, train_loader, num_epochs=args.epochs, lr=args.lr)

    # 평가 및 시각화
    evaluate(model, X, Y, fscaler, tscaler)

if __name__ == '__main__':
    main()
