import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import AbstractDataset
from utils import combine_logs
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item
import pandas as pd

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = self.dataset.fetch_train_example if split == 'train' else self.dataset.fetch_val_example

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def compute_accuracy(y_true, logits):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == y_true).sum().item()
    return correct / y_true.size(0)

def train(config):
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']

    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()

    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))

    step = 0
    metrics_log = []

    try:
        for x, y in tqdm(train_dataloader):
            loss, logs = model.get_loss(x.to(device), y.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()

            train_accuracy = compute_accuracy(y.to(device), logs['logits'])

            if (step + 1) % train_cfg['eval_every'] == 0:
                if wandb_cfg['use_wandb']:
                    model_weights_file = f"final-S5-model_weights_step_{step + 1}.pth"
                    torch.save(model.state_dict(), model_weights_file)
                    artifact = wandb.Artifact(f"final-S5-model-weights-step-{step + 1}", type="model")
                    artifact.add_file(model_weights_file)
                    wandb.log_artifact(artifact)

                model.eval()
                with torch.no_grad():
                    all_val_logs = []
                    val_accuracies = []
                    val_losses = []
                    for i, (val_x, val_y) in enumerate(val_dataloader):
                        if i >= train_cfg['eval_batches']:
                            break
                        val_loss, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                        all_val_logs.append(val_logs)
                        val_losses.append(val_logs['loss'])
                        val_accuracies.append(val_logs['accuracy'])

                total_val_loss = sum(l * c for l, c in val_losses)
                total_val_count = sum(c for _, c in val_losses)
                val_loss_avg = total_val_loss / total_val_count

                total_val_acc = sum(a * c for a, c in val_accuracies)
                total_acc_count = sum(c for _, c in val_accuracies)
                val_accuracy = total_val_acc / total_acc_count

                out_log = {
                    'step': step + 1,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss_avg,
                    'lr': float(lr_schedule.get_last_lr()[0])
                }

                print(out_log)
                if wandb_cfg['use_wandb']:
                    wandb.log(out_log)

                metrics_log.append(out_log)
                model.train()

            step += 1
            if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving metrics...")

    finally:
        df = pd.DataFrame(metrics_log)
        csv_filename = "S5-training_metrics.csv"
        df.to_csv(csv_filename, index=False)

        if wandb_cfg['use_wandb']:
            metrics_artifact = wandb.Artifact("training-metrics-log", type="metrics")
            metrics_artifact.add_file(csv_filename)
            wandb.log_artifact(metrics_artifact)

@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
