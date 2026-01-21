import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from viewpoint_diverse_training_diet.trainer.trainer import Trainer
from viewpoint_diverse_training_diet.utils.distributed import barrier, is_main_process


class DistributedTrainer(Trainer):
    """Trainer variant with distributed training safeguards."""

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        *,
        config,
        device,
        data_loader,
        distributed_cfg,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        self.distributed_cfg = distributed_cfg or {}
        self.is_distributed = self.distributed_cfg.get('enabled', False)
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config=config,
            device=device,
            data_loader=data_loader,
            valid_data_loader=valid_data_loader,
            lr_scheduler=lr_scheduler,
            len_epoch=len_epoch,
        )
        self.is_main_process = is_main_process()
        if not self.is_main_process and self.writer.enabled:
            self.writer.enabled = False

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.is_distributed:
                barrier()

            log = {'epoch': epoch}
            log.update(result)

            if self.is_main_process:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (
                        self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    if self.is_main_process:
                        self.logger.warning(
                            "Warning: Metric '%s' is not found. Model performance monitoring is disabled.",
                            self.mnt_metric,
                        )
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    if self.is_main_process:
                        self.logger.info(
                            "Validation performance didn't improve for %s epochs. Training stops.",
                            self.early_stop,
                        )
                    break

            if self.is_main_process and epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        if self.is_main_process and self.writer.enabled:
            self.writer.finish()

    def _train_epoch(self, epoch):
        self.model.train()

        if (
            self.is_distributed
            and hasattr(self.data_loader, 'sampler')
            and isinstance(self.data_loader.sampler, DistributedSampler)
        ):
            self.data_loader.sampler.set_epoch(epoch)

        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if self.is_main_process and batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                    )
                )
                if self.writer.enabled:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        if self.is_distributed:
            self.train_metrics.synchronize_between_processes()

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                if self.is_main_process and self.writer.enabled:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if self.writer.enabled and self.is_main_process:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        if self.is_distributed:
            self.valid_metrics.synchronize_between_processes()
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best=False):
        if not self.is_main_process:
            return
        super()._save_checkpoint(epoch, save_best=save_best)
