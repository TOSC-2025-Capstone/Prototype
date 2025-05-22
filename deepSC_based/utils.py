import matplotlib.pyplot as plt


# ---------- 시각화 함수 ----------
def visualize_reconstruction(X, X_recon, scaler, idx=0):
    """
    재구성 결과 시각화: 입력 시계열과 복원 시계열 비교
    """
    original = scaler.inverse_transform(X[idx])
    recon = scaler.inverse_transform(X_recon[idx])

    plt.figure(figsize=(12, 8))
    for i in range(original.shape[1]):
        plt.subplot(original.shape[1], 1, i+1)
        plt.plot(original[:, i], label='Original')
        plt.plot(recon[:, i], label='Reconstructed', linestyle='--')
        plt.legend()
    plt.tight_layout()
    plt.show()
