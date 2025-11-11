from functools import partial
import logging
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
        rampup_scheduler=None,
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
            logging.info('Mean Teacher not used')

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        
        # Rampup scheduler for unsupervised weight
        # Create a dummy optimizer with a single parameter to use with the scheduler
        self.max_unsupervised_weight = unsupervised_weight
        self.current_unsup_weight_param = torch.nn.Parameter(torch.tensor(0.0))
        self.rampup_optimizer = torch.optim.SGD([self.current_unsup_weight_param], lr=0.1)
        if rampup_scheduler is not None:
            self.rampup_scheduler = rampup_scheduler(optimizer=self.rampup_optimizer)
        else:
            # Default: linear ramp-up over 80 epochs
            logging.info('Using default linear ramp-up scheduler for unsupervised weight over 80 epochs.')
            self.rampup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.rampup_optimizer, 
                start_factor=0.0, 
                end_factor=1.0, 
                total_iters=80
            )

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
            logging.info('No unlabeled dataloader found; unsupervised training disabled.')

        # Track current epoch
        self.current_epoch = 0

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

    def _get_current_unsupervised_weight(self):
        """Get current unsupervised weight based on scheduler."""
        # Get the learning rate from the rampup optimizer (which is being scheduled)
        current_lr = self.rampup_optimizer.param_groups[0]['lr']
        # Scale it to the max unsupervised weight and cap at max value
        current_weight = min(current_lr, self.max_unsupervised_weight)
        return current_weight

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
                    logging.info('No teacher models available; using student predictions in validation.')
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def test(self):
        """Evaluate the model on the test dataset after training is complete."""
        for model in self.models:
            model.eval()

        test_losses = []
        
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction (use teacher if available, otherwise student)
                if self.use_mean_teacher and self.teacher_models is not None:
                    preds = [teacher(x) for teacher in self.teacher_models]
                else:
                    preds = [model(x) for model in self.models]
                    logging.info('No teacher models available; using student predictions in test.')
                avg_preds = torch.stack(preds).mean(0)
                
                test_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())
        test_loss = np.mean(test_losses)
        return {"test_MSE": test_loss}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.current_epoch = epoch
            current_unsup_weight = self._get_current_unsupervised_weight()
            
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
            self.rampup_scheduler.step()  # Step the rampup scheduler
            supervised_losses_logged = np.mean(supervised_losses_logged)
            if len(unsupervised_losses_logged) > 0:
                unsupervised_losses_logged = np.mean(unsupervised_losses_logged)
            else:
                unsupervised_losses_logged = 0.0

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "unsupervised_loss": unsupervised_losses_logged,
                "unsupervised_weight": current_unsup_weight,
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
