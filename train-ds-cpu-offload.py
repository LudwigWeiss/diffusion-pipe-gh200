import argparse
import os
from datetime import datetime, timezone
import shutil
import json
import multiprocess as mp

import toml
import deepspeed
from deepspeed import comm as dist
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import dataset as dataset_util
from utils.common import is_main_process, DTYPE_MAP
from utils.config import set_config_defaults, get_most_recent_run_dir, print_model_info
from utils.training import evaluate, get_optimizer
from utils.patches import apply_patches
from models.hunyuan_video_wrapper import HunyuanVideoWrapper

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to TOML configuration file.")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher",
)
parser.add_argument(
    "--resume_from_checkpoint",
    action="store_true",
    default=None,
    help="resume training from the most recent checkpoint",
)
parser.add_argument(
    "--regenerate_cache",
    action="store_true",
    default=None,
    help="Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.",
)
parser.add_argument(
    "--cache_only",
    action="store_true",
    default=None,
    help="Cache model inputs then exit.",
)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


if __name__ == "__main__":
    apply_patches()

    # needed for broadcasting Queue in dataset.py
    mp.current_process().authkey = b"afsaskgfdjh4"

    # Load and process configuration
    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config["model"]["dtype"]

    # Handle command line overrides
    resume_from_checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint is not None else config.get("resume_from_checkpoint", False)
    regenerate_cache = args.regenerate_cache if args.regenerate_cache is not None else config.get("regenerate_cache", False)

    # Initialize distributed training
    deepspeed.init_distributed()
    torch.cuda.set_device(dist.get_rank())

    # Create and initialize model
    wrapped_model = HunyuanVideoWrapper(config)
    layers = wrapped_model.to_layers()
    
    # Setup pipeline configuration
    additional_pipeline_module_kwargs = {}
    if config['activation_checkpointing']:
        checkpoint_func = deepspeed.checkpointing.checkpoint
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': wrapped_model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })
    
    # Create pipeline model
    pipeline_model = deepspeed.pipe.PipelineModule(
        layers=layers,
        num_stages=config['pipeline_stages'],
        partition_method=config.get('partition_method', 'parameters'),
        **additional_pipeline_module_kwargs
    )
    
    # Get trainable parameters and initialize DeepSpeed
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
        config=config["ds_config"]
    )

    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1/warmup_steps,
            total_iters=warmup_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[warmup_steps]
        )
    model_engine.lr_scheduler = lr_scheduler

    # Setup datasets
    with open(config["dataset"]) as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model.name)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache)
    dataset_manager.register(train_data)

    # Setup evaluation datasets
    eval_data_map = {}
    for i, eval_dataset in enumerate(config["eval_datasets"]):
        if type(eval_dataset) == str:
            name = f"eval{i}"
            config_path = eval_dataset
        else:
            name = eval_dataset["name"]
            config_path = eval_dataset["config"]
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model.name)
        dataset_manager.register(eval_data_map[name])

    dataset_manager.cache()
    if args.cache_only:
        quit()

    # Initialize adapter if configured
    if adapter_config := config.get("adapter", None):
        peft_config = model.configure_adapter(adapter_config)
        if init_from_existing := adapter_config.get("init_from_existing", None):
            model.load_adapter_weights(init_from_existing)
    else:
        peft_config = None

    # Setup output directory
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config["output_dir"], datetime.now(timezone.utc).strftime("%Y%m%d_%H-%M-%S"))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
    dist.barrier()
    run_dir = get_most_recent_run_dir(config["output_dir"])

    # Initialize data parallel configuration
    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
        )

    # Setup training state
    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine.gradient_accumulation_steps(), model)
    step = 1

    # Resume from checkpoint if requested
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states="force_constant_lr" not in config,
        )
        dist.barrier()
        assert load_path is not None
        train_dataloader.load_state_dict(client_state["custom_loader"])
        step = client_state["step"] + 1
        del client_state
        if is_main_process():
            print(f"Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}")

    # Handle forced constant learning rate
    if "force_constant_lr" in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = config["force_constant_lr"]

    # Setup evaluation
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config["epochs"]
    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(
            eval_data,
            config["eval_gradient_accumulation_steps"],
            model,
            num_dataloader_workers=0,
        )
        for name, eval_data in eval_data_map.items()
    }

    # Initialize training
    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, peft_config, run_dir, model, train_dataloader, model_engine, pipeline_model)

    if config["eval_before_first_step"] and not resume_from_checkpoint:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, config["eval_gradient_accumulation_steps"])

    # Set communication data type
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    # Training loop
    epoch_loss = 0
    num_steps = 0

    while True:
        try:
            # Reset activation shapes for pipeline stages
            model_engine.reset_activation_shape()
            
            # Forward and backward pass
            loss = model_engine.train_batch().item()
            
            # Update metrics
            epoch_loss += loss
            num_steps += 1
            train_dataloader.sync_epoch()
            
            # Handle epoch completion
            new_epoch = saver.process_epoch(epoch, step)
            finished_epoch = new_epoch != epoch
            
            # Log progress
            if is_main_process() and step % config['logging_steps'] == 0:
                tb_writer.add_scalar('train/loss', loss, step)
            
            # Handle evaluation
            if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or \
               (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
                evaluate(
                    model_engine,
                    eval_dataloaders,
                    tb_writer,
                    step,
                    config['eval_gradient_accumulation_steps']
                )
            
            # Handle epoch completion
            if finished_epoch:
                if is_main_process():
                    tb_writer.add_scalar('train/epoch_loss', epoch_loss/num_steps, epoch)
                epoch_loss = 0
                num_steps = 0
                epoch = new_epoch
                if epoch is None:
                    break
            
            # Process step and increment
            saver.process_step(step)
            step += 1
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    if is_main_process():
        print("TRAINING COMPLETE!")
