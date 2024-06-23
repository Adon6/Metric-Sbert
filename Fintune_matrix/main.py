import torch
from model import SiameseNetworkMUL, SiameseNetworkADD
from trainer import NLIModelTrainer

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 选择模型: SiameseNetworkMUL 或 SiameseNetworkADD
    num_classes = 3  # 由于NLI任务包含三个标签：contradiction, entailment, neutral
    model = SiameseNetworkADD(num_classes=num_classes, smodel='all-MiniLM-L6-v2', normalized=True, device = device).to(device)

    # 初始化训练器
    trainer = NLIModelTrainer(model=model, device=device, save_path='models/Jun/')

    # 开始训练
    trainer.train(epochs=20, batch_size=32, lr=0.001)

    # 测试模型
    trainer.test(batch_size=1)

if __name__ == "__main__":
    main()
