import wandb


class WandbWriter:
    """Thin wrapper around Weights & Biases logging."""

    def __init__(self, log_dir, logger, cfg_wandb):
        cfg = cfg_wandb or {}
        self.logger = logger
        self.run = None
        self.enabled = cfg.get('enabled', True)
        self.step = 0
        self.mode = ''

        if not self.enabled:
            return

        init_kwargs = {
            'project': cfg.get('project', 'viewpoint-diverse-training'),
            'name': cfg.get('name'),
            'config': cfg.get('config', {}),
            'dir': str(log_dir),
        }

        try:
            self.run = wandb.init(**init_kwargs)
            self.logger.info(
                "Wandb initialized. Project: %s, Run: %s",
                self.run.project,
                self.run.name,
            )
        except Exception as exc:  # pragma: no cover - wandb runtime failure
            self.logger.warning("Failed to initialize wandb: %s", exc)
            self.enabled = False
            self.run = None

    def set_step(self, step, mode='train'):
        """Set the current step for logging"""
        self.mode = mode
        self.step = step

    def log(self, key, value):
        """Log a scalar value"""
        if self.enabled:
            wandb.log({f'{self.mode}/{key}': value}, step=self.step)

    def add_scalar(self, tag, value):
        """Log a scalar value (TensorBoard-like API)"""
        if self.enabled:
            wandb.log({f'{self.mode}/{tag}': value}, step=self.step)

    def add_image(self, tag, image):
        """Log an image (expects a torch tensor or numpy array)"""
        if self.enabled:
            wandb.log({f'{self.mode}/{tag}': wandb.Image(image)}, step=self.step)

    def add_histogram(self, tag, values, bins='auto'):
        """Log a histogram of values"""
        if self.enabled:
            wandb.log({f'{self.mode}/{tag}': wandb.Histogram(values.cpu().detach().numpy())}, step=self.step)

    def finish(self):
        """Finish the wandb run"""
        if self.enabled:
            wandb.finish()
