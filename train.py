from utils import get_dataloader, Meter
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import time


class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss. i.e. bce loss + dice loss / others 
        optimizer: optimizer for weights updating. i.e. Adam 
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases. i.e. DataLoader / dataloader 
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics. -> return the jaccard coeff / dice loss 
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """

    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 data_type: list,
                 batch_size: int,
                 num_epochs: int,
                 path_to_log: str,
                 model_name: str,
                 img_depth: int,
                 img_width: int,
                 display_plot: bool = True,
                 ):
        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=2, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.path_to_log = path_to_log
        self.model_name = model_name
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs
        self.best_epoch = 0
        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset, data_type=data_type,
                phase=phase,
                batch_size=batch_size,
                img_depth=img_depth,
                img_width=img_width,
                num_workers=4
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")

        # calculating the list of losses for both train & validation phases
        self.losses = {phase: [] for phase in self.phases}

        # calculating the dice scores for both train & validation phases
        self.dice_scores = {phase: [] for phase in self.phases}

        # calculating the jaccard scores for both train & validation phases
        self.jaccard_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)

        # making images predictions symmetric using logits
        logits = self.net(images)

        # calculating the loss bce loss / dice loss / jaccard loss / combined loss
        # as defined calcluating the mean square error loss
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        # with open(f'{self.path_to_log}/train_log({self.model_name}).txt', 'a') as f:
        #     f.write(
        #         f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}\n")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in tqdm(enumerate(dataloader), total=total_batches, desc=f'epoch {epoch}({phase})'):
            images, targets = data_batch['image'], data_batch['mask']
            # BCEDiceLoss & raw prediction( logits ) are calculated
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                # Backpropagating the losses generated to train the Unet
                loss.backward()

                # if a certain no. is reached then all the gradient accuwlated will be given to the optiizer & it gets trained
                # after giving, gradient gets reset to 0
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            # with open(f'{self.path_to_log}/train_log({self.model_name}).txt','a') as f:
            #     f.write(f"[{itr}/{total_batches}] running loss of epoch {epoch} is : {running_loss}\n")
            # meter.update stores running_loss for each iteration in one epoch in a list to visualize in graph
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                         )

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss

    def run(self):
        for epoch in tqdm(range(self.num_epochs), desc=self.model_name):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                print(f"BCEDiceLoss for epoch {epoch} is : ", val_loss)
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(),
                           f"models/{self.model_name}/best-{self.model_name}.pth")
            self.best_epoch = epoch
            self._save_log()
        self._save_train_history()

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            plt.suptitle(self.model_name)
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(
                f'models/{self.model_name}/plots/train_scores({self.model_name}).jpg')
            # plt.show()

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"models/{self.model_name}/latest-{self.model_name}.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key+log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                     ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(f"{self.path_to_log}/train_log({self.model_name}).csv", index=False)

    def _save_log(self):
        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key+log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                     ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(f"{self.path_to_log}/train_log({self.model_name}).csv", index=False)
