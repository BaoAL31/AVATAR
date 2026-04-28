import logging
import os
import warnings

import utils.hydra_traceback_compat  # noqa: F401  # before hydra (Py3.10 + hydra-core 1.1.x)

# Backup if torchvision/other deps still emit these after fixes below.
warnings.filterwarnings(
    "ignore",
    message=".*antialias parameter of all the resizing transforms.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*number of training batches.*smaller than the logging interval.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*monitoring_step.*floating point.*",
    category=UserWarning,
)

from utils.hf_env import apply_hub_download_ui_env, ensure_hf_env

ensure_hf_env()  # set HF_HUB_* cache/timeouts before huggingface_hub.constants is imported

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from data.data_module_ft import DataModule
from learner_ft import SSLLearner
from utils.hf_paths import register_hf_hydra_resolvers
from utils.utils import average_checkpoints

register_hf_hydra_resolvers()

# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False
# __spec__ = None


# Default finetuning recipe (Base Plus + LoRA). Override with --config-name (e.g. config_ft_lrs2_lora_base_high_lrs3).
@hydra.main(config_path="conf", config_name="config_ft_lrs2_lora_baseplus_high_lrs3vox2")
def main(cfg):
    apply_hub_download_ui_env(cfg)
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
    cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]

    cfg.gpus = torch.cuda.device_count()
    print('num gpus:', cfg.gpus)

    wandb_logger = None
    if cfg.log_wandb:
        wandb_logger = instantiate(cfg.logger)
    
    torch.set_float32_matmul_precision(precision=cfg.matmul_precision)

    data_module = DataModule(cfg)
    learner = SSLLearner(cfg)

    ckpt_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        dirpath=os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name) if cfg.checkpoint.dirpath else None,
        save_last=True,
        filename=f'{{epoch}}',
        save_top_k=cfg.checkpoint.save_top_k,
    )
    callbacks = [ckpt_callback]
    if cfg.log_wandb:
        callbacks.append(LearningRateMonitor(logging_interval=cfg.logging.logging_interval))
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    if cfg.test:
        trainer.test(learner, datamodule=data_module)
    else:
        if not cfg.test_avg:
            trainer.fit(learner, data_module, ckpt_path=cfg.ckpt_path)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            ckpt_dir = os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name)
            # range(max_epochs - avg_ckpts, max_epochs) is wrong when max_epochs < avg_ckpts (e.g. 1 vs 10 → epoch=-9).
            start_e = max(0, trainer.max_epochs - cfg.model.avg_ckpts)
            last = [
                os.path.join(ckpt_dir, f"epoch={n}.ckpt") for n in range(start_e, trainer.max_epochs)
            ]
            last = [p for p in last if os.path.isfile(p)]
            if not last:
                fallback = os.path.join(ckpt_dir, "last.ckpt")
                if os.path.isfile(fallback):
                    last = [fallback]
            if not last:
                print(
                    f"Skipping checkpoint average: no epoch=*.ckpt under {ckpt_dir} "
                    f"(expected epochs {start_e}..{trainer.max_epochs - 1})."
                )
            else:
                avg = average_checkpoints(last)

                model_path = os.path.join(ckpt_dir, f"model_avg_{cfg.model.avg_ckpts}.pth")
                torch.save(avg, model_path)

                # compute WER
                cfg.gpus = cfg.trainer.devices = cfg.trainer.num_nodes = 1
                cfg.model.pretrained_model_path = model_path
                cfg.model.transfer_only_encoder = False
                data_module = DataModule(cfg)
                learner = SSLLearner(cfg)
                trainer = Trainer(**cfg.trainer, logger=wandb_logger)
                trainer.test(learner, datamodule=data_module)

    
if __name__ == "__main__":
    main()
