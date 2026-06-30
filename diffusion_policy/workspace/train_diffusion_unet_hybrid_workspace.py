if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    @staticmethod
    def _is_colab_runtime():
        return 'google.colab' in sys.modules

    def _get_resume_checkpoint_path(self, cfg):
        resume_path = OmegaConf.select(cfg, 'checkpoint.resume_path')
        if resume_path is not None and str(resume_path).strip() != '':
            return pathlib.Path(os.path.expanduser(str(resume_path)))
        return self.get_checkpoint_path()

    def _should_sync_to_drive(self, cfg):
        if not self._is_colab_runtime():
            return False
        drive_sync_enabled = OmegaConf.select(cfg, 'checkpoint.drive_sync_enabled')
        if drive_sync_enabled is False:
            return False
        drive_sync_every = int(OmegaConf.select(cfg, 'checkpoint.drive_sync_every') or 0)
        if drive_sync_every <= 0:
            return False
        return (self.epoch % drive_sync_every) == 0

    def _sync_latest_checkpoint_to_drive(self, cfg, latest_ckpt_path):
        drive_sync_dir = OmegaConf.select(cfg, 'checkpoint.drive_sync_dir')
        if drive_sync_dir is None or str(drive_sync_dir).strip() == '':
            drive_sync_dir = '/content/drive/MyDrive/diffusion_checkpoints/'

        source_path = pathlib.Path(latest_ckpt_path)
        if not source_path.is_file():
            return

        run_name = pathlib.Path(self.output_dir).name
        target_dir = pathlib.Path(os.path.expanduser(str(drive_sync_dir))).joinpath(run_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, target_dir.joinpath(source_path.name))
        print(f"Synced checkpoint to Google Drive: {target_dir}")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        # torch, numpy, random의 seed 고정 -> 같은 sequence의 난수 생성, 디버깅이 용이함
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # policy model 인스턴스 생성
        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)   # cfg의 policy: 아래 긁어옴

        # use_ema가 True이면 원본 모델 self.model을 copy함; ema는 최근 data에 지수적으로 큰 가중치로 평균을 매김, 추가적인 도구
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        resumed = False

        # resume training
        if cfg.training.resume:
            resume_ckpt_path = self._get_resume_checkpoint_path(cfg)
            if resume_ckpt_path.is_file():
                print(f"Resuming from checkpoint {resume_ckpt_path}")
                self.load_checkpoint(path=resume_ckpt_path)
                resumed = True
            else:
                print(f"Resume requested but checkpoint not found: {resume_ckpt_path}")

        # 데이터셋 다루는 부분
        # configure dataset 
        dataset: BaseImageDataset  
        dataset = hydra.utils.instantiate(cfg.task.dataset)   # config에서 task: dataset : 아래의 _targe_ Class가 호출되고 그 아래 파라미터들이 같이 들어감
        assert isinstance(dataset, BaseImageDataset)
        # training용 data를 load하는 인스턴스; torch.utils.data 찾아보기
        train_dataloader = DataLoader(dataset, **cfg.dataloader)   # config에서 dataloader: 아래 변수들 가져옴
        # data 정규화
        normalizer = dataset.get_normalizer() # obs, action의 scale, offset params 들어있음

        # Real에서는 val 안씀
        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # validation용 data를 load하는 인스턴스
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
      
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            start_epoch = 0
            if resumed:
                start_epoch = int(self.epoch) + 1
                if start_epoch >= cfg.training.num_epochs:
                    print(
                        f"Checkpoint epoch ({self.epoch}) already reached/exceeded num_epochs ({cfg.training.num_epochs}). "
                        "Nothing to train.")
                    return

            for local_epoch_idx in range(start_epoch, cfg.training.num_epochs):
                self.epoch = local_epoch_idx
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema; self.model의 가중치를 지수 이동 평균으로 update
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 0:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model.compute_loss(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    should_drive_sync = self._should_sync_to_drive(cfg)
                    latest_ckpt_path = None

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        # use_thread=False to ensure checkpoint is written before continuing
                        latest_ckpt_path = self.save_checkpoint(
                            use_thread=False)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    if should_drive_sync:
                        try:
                            if latest_ckpt_path is None:
                                latest_ckpt_path = self.get_checkpoint_path()
                            self._sync_latest_checkpoint_to_drive(cfg, latest_ckpt_path)
                        except Exception as sync_error:
                            print(f"Google Drive sync failed: {sync_error}")

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
