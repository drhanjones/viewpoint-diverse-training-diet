import wandb


class WandbWriter:
    """
    Wandb writer class for logging metrics, images, and histograms
    """
    def __init__(self, log_dir, logger, cfg_wandb):
        self.logger = logger
        self.enabled = cfg_wandb
        self.step = 0
        self.mode = ''
        
        if self.enabled:
            wandb.init(
                project=cfg_wandb.get('project', 'viewpoint-diverse-training'),
                name=cfg_wandb.get('name', None),
                config=cfg_wandb.get('config', {}),
                dir=str(log_dir)
            )
            self.logger.info(f"Wandb initialized. Project: {wandb.run.project}, Run: {wandb.run.name}")

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
