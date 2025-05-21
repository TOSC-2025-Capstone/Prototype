# model.py
import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    """
    시계열 입력을 받아 의미 벡터(latent vector)로 압축하는 LSTM 기반 인코더
    """
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class SemanticDecoder(nn.Module):
    """
    의미 벡터를 받아 시계열 복원 + 상태 지표 예측을 수행하는 디코더
    """
    def __init__(self, latent_dim=32, hidden_dim=64, seq_len=50, output_dim=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.recon_head = nn.Linear(hidden_dim, output_dim)
        self.reg_head = nn.Linear(hidden_dim, 3)

    def forward(self, z):
        h = self.fc(z).unsqueeze(1).repeat(1, 50, 1)
        out, _ = self.lstm(h)
        return self.recon_head(out), self.reg_head(out[:, -1])

class DeepSCBattery(nn.Module):
    """
    전체 모델: 인코더 + 디코더
    """
    def __init__(self):
        super().__init__()
        self.encoder = SemanticEncoder()
        self.decoder = SemanticDecoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)