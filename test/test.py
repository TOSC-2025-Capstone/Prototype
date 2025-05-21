import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Task-Oriented Semantic Communication Training")
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')
parser.add_argument('--input_dim', type=int, default=6, help='Input feature dimension')
parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension for encoder')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension in task predictor')
parser.add_argument('--task_output_dim', type=int, default=3, help='Number of task prediction targets')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight for reconstruction loss in total loss')
args = parser.parse_args()

# -----------------------------
# Device 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 하이퍼파라미터 할당
# -----------------------------
batch_size = args.batch_size
seq_len = args.seq_len
input_dim = args.input_dim
latent_dim = args.latent_dim
hidden_dim = args.hidden_dim
task_output_dim = args.task_output_dim
learning_rate = args.lr
alpha = args.alpha
epochs = args.epochs

# -----------------------------
# 모델 정의
# -----------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (batch_size, latent_dim)

class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim, output_dim, batch_first=True)

    def forward(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_repeated)
        return out  # (batch_size, seq_len, output_dim)

class TaskPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # (batch_size, output_dim)

# -----------------------------
# 모델 초기화
# -----------------------------
encoder = SemanticEncoder(input_dim, latent_dim).to(device)
decoder = SemanticDecoder(latent_dim, input_dim, seq_len).to(device)
predictor = TaskPredictor(input_dim, hidden_dim, task_output_dim).to(device)

# -----------------------------
# 손실함수 & 옵티마이저
# -----------------------------
recon_loss_fn = nn.MSELoss()
task_loss_fn = nn.MSELoss()

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(predictor.parameters()), lr=learning_rate)

# -----------------------------
# 학습 루프
# -----------------------------
for epoch in range(1, epochs + 1):
    # 더미 배터리 시계열 입력 & 목표값
    x_raw = torch.randn(batch_size, seq_len, input_dim).to(device)
    y_task = torch.randn(batch_size, task_output_dim).to(device)

    # Forward
    z = encoder(x_raw)
    x_recon = decoder(z)
    task_pred = predictor(x_recon)

    recon_loss = recon_loss_fn(x_recon, x_raw)
    task_loss = task_loss_fn(task_pred, y_task)
    total_loss = alpha * recon_loss + (1 - alpha) * task_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Logging
    print(f"Epoch {epoch:02d}/{epochs} | Recon Loss: {recon_loss.item():.4f} | Task Loss: {task_loss.item():.4f} | Total Loss: {total_loss.item():.4f}")
