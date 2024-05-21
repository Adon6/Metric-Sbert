import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import util, losses
from tqdm import tqdm
import logging
import os
import gzip
import csv
import datetime

def print_grad_fn(grad_fn, level=0):
    if grad_fn is None:
        return
    print(' ' * level * 2 + str(grad_fn))
    if hasattr(grad_fn, 'next_functions'):
        for func in grad_fn.next_functions:
            if func[0] is not None:
                print_grad_fn(func[0], level + 1)
    if hasattr(grad_fn, 'saved_tensors'):
        for t in grad_fn.saved_tensors:
            if hasattr(t, 'grad_fn') and t.grad_fn is not None:
                print_grad_fn(t.grad_fn, level + 1)

# dataset allnli
class NLIDataset(Dataset):
    def __init__(self, datasplit = "train", device = "cuda" ):
        self.device = device 
        self.datasplit = datasplit 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if dataset exists. If not, download and extract it
        nli_dataset_path = "data/AllNLI.tsv.gz"
        if not os.path.exists(nli_dataset_path):
            util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)
        
        count = 0
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == self.datasplit:
                    train_samples.append((row["sentence1"], row["sentence2"],label2int[row["label"]]))
                    count+=1
        print(f"In total {count} items for traininf")
        self.dataset = train_samples

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        premise, hypothesis, label  = self.dataset[idx]
        return premise, hypothesis, label

class NLIModelTrainer():
    def __init__(self, model, device = "cuda", save_path='models/') -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO
        )
        self.device = device
        self.model_type = model.model_name  
        self.base_path = save_path  

        # makedir for saving
        os.makedirs(self.base_path, exist_ok=True)

        # 生成初始保存路径，此时不包含时间戳和训练轮次
        self.save_path = self.generate_save_path("initial")

        logging.info(f"Initializing model on device {self.device} and saving to {self.save_path}")

        # Initialize model
        self.siamese_model = model.to(self.device)

    def generate_save_path(self, affix):
        # filename generation with time stamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.model_type}_{current_time}_"+ affix+".pth"
        return os.path.join(self.base_path, filename)
    


    def train(self, epochs=3, batch_size=32, lr=0.001):
        logging.info("Starting training process")
        data = NLIDataset("train", self.device)
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.siamese_model.parameters(), lr=lr)
        minloss = 3
        for epoch in range(epochs):
            total_loss = 0
            for premise, hypothesis, labels in self.dataloader:
                optimizer.zero_grad()
                labels = labels.to(self.device)
                outputs = self.siamese_model(premise, hypothesis) 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(self.dataloader)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")
            
            if average_loss < minloss:
                minloss = average_loss
                self.save_path = self.generate_save_path(str(epoch+1))
                self.save_model()   
                logging.info(f"Model saved to {self.save_path}")
        
        self.save_path = self.generate_save_path("final")  
        self.save_model()   
        logging.info(f"Model saved to {self.save_path}")

    def testx(self, batch_size=1):
        logging.info("Starting testing process")
        data = NLIDataset("test", self.device)
        self.test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        self.siamese_model.eval()

        for premise, hypothesis, label in self.test_dataloader:
            output = self.siamese_model(premise, hypothesis)
            predicted_label = torch.argmax(output)
            logging.info(f"Predicted: {predicted_label.item()}, Actual: {label.item()}")

    def test(self, batch_size=1):
        logging.info("Starting testing process")
        data = NLIDataset("test", self.device)
        self.test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        self.siamese_model.eval()

        all_predictions = []
        all_labels = []

        for premise, hypothesis, label in self.test_dataloader:
            output = self.siamese_model(premise, hypothesis)
            predicted_label = torch.argmax(output, dim=1)

            all_predictions.extend(predicted_label.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")

    def testa(self, batch_size=1):
        logging.info("Starting testing process")
        data = NLIDataset("test", self.device)
        self.test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        self.siamese_model.eval()
        metrics = MetricWrapper(self.device)

        for premise, hypothesis, label in self.test_dataloader:
            output = self.siamese_model(premise, hypothesis)
            predicted_label = torch.argmax(output, dim=1)

            metrics.update(predicted_label, label)

        results = metrics.compute()

        logging.info(results["accuracy"])
        logging.info(results["precision"])
        logging.info(results["recall"])
        logging.info(results["f1"])

    def plot(self):
        pass  # Placeholder for potential future plot implementations

    def save_model(self):
        model_dict = self.siamese_model.state_dict()
        # exclude Embedding model for saving storage
        model_dict_filtered = {k: v for k, v in model_dict.items() if not k.startswith('E.sbert')}

        metadata = {
            'model_type': self.siamese_model.model_name,
            'embedding_model': self.siamese_model.embedding_model,
            'normalized': self.siamese_model.normalized
        }
        model_dict['metadata'] = metadata

        torch.save(model_dict_filtered, self.save_path)

class MetricWrapper:
    def __init__(self, device):
        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes=3).to(device)
        self.precision = torchmetrics.Precision(task = "multiclass", num_classes=3).to(device)
        self.recall = torchmetrics.Recall(task = "multiclass",num_classes=3).to(device)
        self.f1 = torchmetrics.F1Score(task = "multiclass",num_classes=3).to(device)

    def update(self, preds, targets):
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self):
        metrics = {
            'accuracy': self.accuracy.compute().mean(),
            'precision': self.precision.compute().mean(),
            'recall': self.recall.compute().mean(),
            'f1': self.f1.compute().mean()
        }
        return metrics

def NLIModelTester():
    def __init__(self):
        pass
    def load_model_with_metadata(model_path):
        # 加载整个文件内容
        checkpoint = torch.load(model_path)

        # 提取元数据
        metadata = checkpoint.pop('metadata', None)

        # 实例化模型，这里需要知道具体的模型类型
        if metadata and metadata['model_type'] == 'SiameseNetworkMUL':
            model = SiameseNetworkMUL(num_classes=3, smodel=metadata['embedding_model'], normalized=metadata['normalized'])
            model.load_state_dict(checkpoint)  # 加载模型参数
            model.eval()  # 如果是用于推断，设置为评估模式
            return model, metadata
        else:
            raise ValueError("Model type is not supported or metadata is missing")

        # 使用示例
        model_path = 'your_model_path.pth'
        model, loaded_metadata = load_model_with_metadata(model_path)
        print("Loaded model metadata:", loaded_metadata)
