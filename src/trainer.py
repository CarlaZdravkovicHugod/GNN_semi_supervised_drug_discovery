from functools import partial

import numpy as np
import torch
from tqdm import tqdm

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        unsupervised_weight: float = 0.0,
        use_mean_teacher: bool = True,
        ema_decay: float = 0.999,
        rampup_epochs: int = 80,
    ):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else device
        self.models = models

        # Mean Teacher: create EMA teacher models
        self.use_mean_teacher = use_mean_teacher
        self.ema_decay = ema_decay
        if self.use_mean_teacher:
            self.teacher_models = [self._create_ema_model(model) for model in self.models]
        else:
            self.teacher_models = None

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        # Optional unlabeled dataloader (QM9 datamodule provides unsupervised_train_dataloader)
        if hasattr(datamodule, "unsupervised_train_dataloader"):
            try:
                self.unlabeled_dataloader = datamodule.unsupervised_train_dataloader()
            except Exception:
                self.unlabeled_dataloader = None
        else:
            self.unlabeled_dataloader = None

        # Unsupervised training hyperparams
        self.unsupervised_weight = unsupervised_weight
        self.rampup_epochs = rampup_epochs
        self.current_epoch = 0  # track for rampup

        # Logging
        self.logger = logger

    def _create_ema_model(self, model):
        """Create a copy of the model for EMA teacher."""
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def _update_ema_model(self, student_model, teacher_model, alpha):
        """Update teacher model parameters using EMA from student model."""
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

    def _get_current_unsupervised_weight(self, epoch):
        """Ramp up unsupervised weight during first rampup_epochs."""
        if self.rampup_epochs == 0:
            return self.unsupervised_weight
        # Linear ramp-up
        rampup_value = min(1.0, epoch / self.rampup_epochs)
        return self.unsupervised_weight * rampup_value

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device) # test Apple M2
                
                # Ensemble prediction (use teacher if available, otherwise student)
                if self.use_mean_teacher and self.teacher_models is not None:
                    preds = [teacher(x) for teacher in self.teacher_models]
                else:
                    preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.current_epoch = epoch
            current_unsup_weight = self._get_current_unsupervised_weight(epoch)
            
            for model in self.models:
                model.train()
            # Set teacher models to eval mode
            if self.use_mean_teacher and self.teacher_models is not None:
                for teacher in self.teacher_models:
                    teacher.eval()
                    
            supervised_losses_logged = []
            unsupervised_losses_logged = []
            # prepare unlabeled iterator if available
            if self.unlabeled_dataloader is not None and current_unsup_weight > 0.0:
                unlabeled_iter = iter(self.unlabeled_dataloader)
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss

                # Optional unsupervised pseudo-labeling / consistency loss with Mean Teacher
                if (
                    self.unlabeled_dataloader is not None
                    and current_unsup_weight > 0.0
                ):
                    try:
                        xu, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_dataloader)
                        xu, _ = next(unlabeled_iter)
                    xu = xu.to(self.device)

                    # Compute ensemble pseudo-labels from teacher (or student ensemble if no teacher)
                    with torch.no_grad():
                        if self.use_mean_teacher and self.teacher_models is not None:
                            preds_u = [teacher(xu) for teacher in self.teacher_models]
                        else:
                            preds_u = [model(xu) for model in self.models]
                        avg_preds_u = torch.stack(preds_u).mean(0)

                    # Consistency: make each student model predict close to the teacher ensemble average
                    unsupervised_losses = [
                        self.supervised_criterion(model(xu), avg_preds_u) for model in self.models
                    ]
                    unsupervised_loss = sum(unsupervised_losses)
                    unsupervised_losses_logged.append(unsupervised_loss.detach().item() / len(self.models))  # type: ignore

                    loss = loss + current_unsup_weight * unsupervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
                
                # Update teacher models using EMA
                if self.use_mean_teacher and self.teacher_models is not None:
                    for student, teacher in zip(self.models, self.teacher_models):
                        self._update_ema_model(student, teacher, self.ema_decay)
                        
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            if len(unsupervised_losses_logged) > 0:
                unsupervised_losses_logged = np.mean(unsupervised_losses_logged)
            else:
                unsupervised_losses_logged = 0.0

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "unsupervised_loss": unsupervised_losses_logged,
                "epochs": epoch,
            }


            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)


# Notes: 
# Slowly reduce the learning rate. 
# This is because our features and labels are all in different magnitudes. 
# Our weights need to move far to get into the right order of magnitude and 
# then need to fine-tune a little. Thus, we start at high learning rate and decrease.

# TODO: Run on test data as well, ie test on testdata. 
# TODO: add unsupervised loss component as well. Fx mean teacher
# TODO: dataset uyamæ qm9 choose predicting variable