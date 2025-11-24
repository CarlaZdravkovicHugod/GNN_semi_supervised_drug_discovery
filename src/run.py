from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf

import logger
from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    # Print a random sample from the dataloader
    print("\n" + "="*60)
    print("INSPECTING FIRST SAMPLE FROM TRAIN DATALOADER")
    print("="*60)
    
    train_loader = dm.train_dataloader(shuffle=False)
    batch, targets = next(iter(train_loader))
    
    # shape info
    print(f"First sample node features size: {batch[0].x.size()}")
    print(f"First sample node features (x):\n{batch[0].x}")
    print(f"\nFirst sample target: {targets[0].item():.6f}")
    print("="*60 + "\n")

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    trainer.train(**cfg.trainer.train)
    
    # Evaluate on test set after training is complete
    test_metrics = trainer.test()
    print(f"\nFinal Test Results: {test_metrics}")
    logger.log_dict(test_metrics)


if __name__ == "__main__":
    main()
