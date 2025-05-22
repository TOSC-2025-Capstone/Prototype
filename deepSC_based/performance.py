import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from config import DEVICE
from utils import visualize_reconstruction


def evaluate(model, X_test, feature_scaler):
    """
    모델 성능 평가: 복원 정확도 측정 및 시각화 수행
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test).to(DEVICE)
        recon, target = model(X_tensor)
        recon = recon.cpu().numpy()

    recon_mse = mean_squared_error(X_test.reshape(-1), recon.reshape(-1))
    print("Reconstruction MSE:", recon_mse)

    visualize_reconstruction(X_test, recon, feature_scaler)
