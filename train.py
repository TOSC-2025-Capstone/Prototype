import torch
import torch.nn as nn

from config import DEVICE


def train(model, train_loader, num_epochs=10, lr=0.001):
    """
    모델 학습 루프: reconstruction + 상태 예측 loss를 함께 최적화
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_recon = nn.MSELoss()
    loss_fn_target = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for xb in train_loader:
            xb = xb[0].to(DEVICE)  # TensorDataset은 튜플 반환
            optimizer.zero_grad()
            recon, target = model(xb)
            # decoder의 reconstruction 출력만 MSE 계산, target은 상태 예측용이므로 무시 가능
            recon_loss = loss_fn_recon(recon, xb)
            loss = recon_loss  # 상태 예측 loss는 생략 또는 나중에 추가  # placeholder Y
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
