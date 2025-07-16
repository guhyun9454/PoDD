""" Main worker function """

import os
import gc
import time
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from pathlib import Path
from pytorch_lightning import seed_everything

from src.PoCO import PoCO
from src.PoDD import PoDD
from src.PoDDL import PoDDL
from src.PoDD_utils import combine_images_with_fade, get_crops_from_poster
from src.util import Summary, AverageMeter, ProgressMeter, accuracy, accuracy_ind
from src.data_utils import get_dataset, get_transform, init_gaussian, ImageIntervention, project

# ------------------------------------------------------------
# Utility: Safe forward pass with automatic cuDNN fallback
# ------------------------------------------------------------
# Some cuDNN kernels fail to find a valid algorithm for large batch / resolution
# combinations. In that case we temporarily disable cuDNN and rerun the forward
# using PyTorch's native convolution implementation, which is slower but
# prevents the training run from crashing.


def safe_forward(model, inputs):  # noqa: D401
    """Forward pass that falls back to ATen convolutions on cuDNN failure.

    Parameters
    ----------
    model : torch.nn.Module or torch.nn.parallel.DataParallel
        The network to be executed.
    inputs : torch.Tensor
        Input mini-batch.

    Returns
    -------
    tuple
        (output, embedding) as returned by the network.
    """
    try:
        return model(inputs)
    except RuntimeError as err:
        # Detect the specific cuDNN algorithm selection failure message.
        msg = str(err).lower()
        if "no valid convolution algorithms" in msg or "cudnn" in msg:
            prev_enabled = cudnn.enabled
            cudnn.enabled = False
            try:
                return model(inputs)
            finally:
                cudnn.enabled = prev_enabled
        # Re-raise if it is a different error.
        raise


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='best_checkpoint.pth'):
    """
    Save checkpoint with comprehensive training state
    
    Args:
        state: Dictionary containing all training state
        is_best: Whether this is the best model so far
        filename: Filename for regular checkpoint
        best_filename: Filename for best checkpoint
    """
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_checkpoint_path = checkpoint_dir / best_filename
        torch.save(state, best_checkpoint_path)
        print(f"Best checkpoint saved to {best_checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scaler=None):
    """
    Load checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Dictionary containing loaded state
    """
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if hasattr(model, 'module'):
        # Handle DataParallel model
        model.module.data = checkpoint['model_data'].cuda().requires_grad_(True)
        if 'model_label' in checkpoint:
            model.module.label = checkpoint['model_label'].cuda().requires_grad_(True)
            
        # Load PoDD model's additional state
        if 'podd_model_state' in checkpoint and hasattr(model.module, 'load_checkpoint_state'):
            model.module.load_checkpoint_state(checkpoint['podd_model_state'])
    else:
        model.data = checkpoint['model_data'].cuda().requires_grad_(True)
        if 'model_label' in checkpoint:
            model.label = checkpoint['model_label'].cuda().requires_grad_(True)
            
        # Load PoDD model's additional state
        if 'podd_model_state' in checkpoint and hasattr(model, 'load_checkpoint_state'):
            model.load_checkpoint_state(checkpoint['podd_model_state'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state for mixed precision
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Checkpoint loaded successfully. Resuming from epoch {checkpoint['epoch']}")
    return checkpoint


def create_checkpoint_state(model, optimizer, scaler, epoch, best_acc1, best_loss1, 
                          best_loss_ind, distill_steps, grad_acc, best_rec, args):
    """
    Create checkpoint state dictionary
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scaler: GradScaler for mixed precision
        epoch: Current epoch
        best_acc1: Best accuracy achieved
        best_loss1: Best loss achieved
        best_loss_ind: Epoch index of best loss
        distill_steps: Current distillation steps
        grad_acc: Gradient accumulation history
        best_rec: Best results record
        args: Training arguments
    
    Returns:
        Dictionary containing all checkpoint state
    """
    state = {
        'epoch': epoch,
        'model_data': model.module.data.clone().cpu().detach(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc1': best_acc1,
        'best_loss1': best_loss1,
        'best_loss_ind': best_loss_ind,
        'distill_steps': distill_steps,
        'grad_acc': grad_acc,
        'best_rec': best_rec,
        'args': args,
        'curriculum': model.module.curriculum,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    
    # Add PoDD model's additional state
    if hasattr(model.module, 'get_checkpoint_state'):
        state['podd_model_state'] = model.module.get_checkpoint_state()
    
    # Add label if training labels
    if hasattr(model.module, 'label') and model.module.train_y:
        state['model_label'] = model.module.label.clone().cpu().detach()
    
    # Add scaler state for mixed precision
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    
    # Add CUDA random state if available
    if torch.cuda.is_available():
        state['cuda_random_state'] = torch.cuda.get_rng_state()
    
    return state


def resume_from_checkpoint(checkpoint_path, model, optimizer, scaler=None):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Tuple of (checkpoint_state, start_epoch) or (None, 0) if no checkpoint
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None, 0
    
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scaler)
    if checkpoint is None:
        return None, 0
    
    # Restore random states for reproducibility
    if 'random_state' in checkpoint:
        random.setstate(checkpoint['random_state'])
    if 'numpy_random_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_random_state'])
    if 'torch_random_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_random_state'])
    if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
    
    # Restore model curriculum
    if hasattr(model, 'module'):
        model.module.curriculum = checkpoint.get('curriculum', model.module.curriculum)
    
    return checkpoint, checkpoint['epoch']


curriculum_type = {}
tmp = list(range(20, -5, -5)) * 20
tmp.sort()
curriculum_type[0] = tmp
tmp.sort(reverse=True)
curriculum_type[1] = tmp

epoch_list = [300, 600, 1000, 2000]


def main_worker(args):
    """ Main worker function """
    global best_acc1, best_loss1

    # seed all the things:
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.

    Path('checkpoints').mkdir(parents=True, exist_ok=True)

    best_acc1 = 0
    best_loss1 = 1000

    cudnn.benchmark = True
    cudnn.deterministic = True
    
    # For ImageNet subset, use the root path directly instead of appending dataset name
    if args.dataset.startswith('imagenet-subset'):
        args.data_root = args.root
    else:
        args.data_root = os.path.join(args.root, args.dataset)
        
    print("Dataset: %s" % args.dataset)
    print("Dataset Path: %s" % args.data_root)
    print(args)

    # 0. Preprocess datasets
    print('==> Preparing data..')
    # Pass resolution for ImageNet subset
    resolution = getattr(args, 'res', None)
    transform_train, transform_test = get_transform(args.dataset, resolution=resolution)

    print(transform_train, transform_test)
    train1, train2, testset, num_classes, shape, process_config = get_dataset(args.dataset,
                                                                              args.data_root,
                                                                              transform_train,
                                                                              transform_test,
                                                                              zca=args.zca,
                                                                              resolution=resolution)

    zca_inverse = None
    if args.zca and process_config is not None:
        zca_inverse = process_config[0]

    print('Dataset: number of classes: {}'.format(num_classes))
    args.num_classes = num_classes

    print('Training set size: {}'.format(len(train1)))

    train_sampler = None
    val_sampler = None

    train_loader1 = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_loader2 = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    channel, image_size_y, image_size_x = shape

    print('Image size: channel {}, height {}, width {}'.format(channel, image_size_y, image_size_x))

    class_order = PoCO.optimize_poster_class_order(
        (args.poster_class_num_y, args.poster_class_num_x), args.dataset, 'cuda:0')

    args.class_order = class_order

    cropping_function = lambda data, indexes_subset: \
        get_crops_from_poster(data, image_size_x, image_size_y,
                              args.patch_num_x, args.patch_num_y, indexes_subset=indexes_subset)

    if args.load_poster_run_name == '':
        class_areas = init_gaussian(num_classes, 1, int(channel * args.class_area_width * args.class_area_height))
        class_areas = project(class_areas)
        class_areas = class_areas.reshape(num_classes, channel, args.class_area_width, args.class_area_height)

        distilled_data = combine_images_with_fade(class_areas, args.poster_width, args.poster_height,
                                                  args.poster_class_num_x, args.poster_class_num_y).unsqueeze(0)

        if not args.train_y:
            y_init = PoDDL.get_poster_labels(class_order, image_size_x, image_size_y,
                                             args.class_area_width, args.class_area_height,
                                             args.poster_width, args.poster_height,
                                             args.poster_class_num_x, args.poster_class_num_y,
                                             args.patch_num_x, args.patch_num_y)
        else:
            y_init = PoDDL.init_label_array(distilled_data.shape, class_order, args.comp_ipc)

    else:
        distilled_data = torch.load(f'checkpoints/{args.load_poster_run_name}_poster.pt')
        y_init = torch.load(f'checkpoints/{args.load_poster_run_name}_label.pt')

    label_cropping_function = None
    if args.train_y:
        label_shrink_factor_x = 1 / (distilled_data.shape[3] / y_init.shape[3])
        label_shrink_factor_y = 1 / (distilled_data.shape[2] / y_init.shape[2])
        label_cropping_function = lambda labels, indexes_subset: \
            PoDDL.get_labels_from_array(labels, label_shrink_factor_x, label_shrink_factor_y,
                                        image_size_x, image_size_y, args.patch_num_x, args.patch_num_y,
                                        indexes_subset=indexes_subset)

    distilled_data = distilled_data.detach().to('cuda:0').requires_grad_(True)

    syn_intervention, real_intervention, interv_prob = set_up_interventions(args)
    print('Synthetic images, not_single {}, keys {}'.format(syn_intervention.not_single, syn_intervention.keys))

    # 1. Initialize Distilled Dataset Module
    print('==> Building model..')
    print('Initialized distilled data with size, x: {}, y:{}'.format(distilled_data.shape, y_init.shape))

    model = PoDD(distilled_data, y_init, cropping_function, args.arch, args.window, args.inner_lr,
                 args.num_train_eval, label_cropping_function=label_cropping_function,
                 total_patch_num=args.patch_num_x * args.patch_num_y,
                 distill_batch_size=args.distill_batch_size, train_y=args.train_y, train_lr=args.train_lr,
                 channel=shape[0], num_classes=num_classes, im_size=(shape[1], shape[2]),
                 inner_optim=args.inner_optim, cctype=args.cctype, syn_intervention=syn_intervention,
                 real_intervention=real_intervention, decay=args.decay)

    print(model.net)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    else:
        if not args.train_y:
            model.label = model.label.cuda()
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss().to(device)
    model.module.dd_type = args.ddtype
    continue_training = False
    start_test_epoch = 0 if continue_training else 1

    print('Check the length of the training dataset {}'.format(len(train_loader1.dataset)))
    if args.train_y:
        if args.outer_optim == 'Adam':
            optimizer = optim.Adam([{'params': model.module.data},
                                    {'params': model.module.label,
                                     'lr': args.lr / args.label_lr_scale}],
                                   lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.wd)

        else:
            raise NotImplementedError()
    else:
        optimizer = optim.Adam([model.module.data],
                               lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.wd)

    best_rec = {}
    grad_acc = []
    best_loss_ind = 0

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.use_mixed_precision else None

    distill_steps = 0
    if args.ddtype == 'curriculum' and args.cctype != 2:
        model.module.curriculum = [args.totwindow - args.window, args.minwindow, 0, 0][args.cctype]

    # Resume from checkpoint if provided
    resume_epoch = 0
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint_state, resume_epoch = resume_from_checkpoint(args.resume, model, optimizer, scaler)
        if checkpoint_state is not None:
            best_acc1 = checkpoint_state.get('best_acc1', best_acc1)
            best_loss1 = checkpoint_state.get('best_loss1', best_loss1)
            best_loss_ind = checkpoint_state.get('best_loss_ind', best_loss_ind)
            distill_steps = checkpoint_state.get('distill_steps', distill_steps)
            grad_acc = checkpoint_state.get('grad_acc', grad_acc)
            best_rec = checkpoint_state.get('best_rec', best_rec)
            args.start_epoch = resume_epoch + 1
            print(f"Resumed from epoch {resume_epoch}, best_acc1: {best_acc1:.4f}, best_loss1: {best_loss1:.4f}")

    # Generate checkpoint filename
    if args.checkpoint_name:
        checkpoint_prefix = args.checkpoint_name
    else:
        checkpoint_prefix = f"{args.dataset}_{args.arch}_{args.name}"

    if model.module.data.get_device() == 0 and args.wandb:
        wandb.init(
            entity="TGwithIU",
            project=f"PoDD",
            name=args.name,
            config=vars(args))

    for epoch in range(args.start_epoch, args.epochs):
        # initialize the EMA (only if not resuming from checkpoint)
        if epoch == 0 and not args.resume:
            model.module.ema_init(args.clip_coef)
        elif epoch == args.start_epoch and args.resume and not hasattr(model.module, 'shadow'):
            # Initialize EMA if resuming but EMA state was not saved
            model.module.ema_init(args.clip_coef)

        if args.train_y:
            print(
                f"[DEBUG] Max={float(optimizer.param_groups[1]['params'][0].max().cpu())} Min={float(optimizer.param_groups[1]['params'][0].min().cpu())}")

        grad_tmp, losses_avg, distill_steps = train(train_loader1, None, model, criterion,
                                                    optimizer, epoch, device, distill_steps, args, scaler)
        grad_acc.append(grad_tmp)
        
        # Prevent memory accumulation by keeping only recent gradient history
        if len(grad_acc) > 50:  # Keep only last 50 epochs of gradient data
            grad_acc = grad_acc[-50:]
        
        # Additional memory cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Memory monitoring (optional, can be removed after debugging)
        if torch.cuda.is_available() and epoch % 10 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3      # Convert to GB
            print(f"[Memory] Epoch {epoch}: Allocated {memory_allocated:.2f}GB, Cached {memory_cached:.2f}GB")
        
        print('The current update step is {}'.format(distill_steps))

        # Save checkpoint periodically
        if model.module.data.get_device() == 0 and (epoch % args.save_freq == 0 or args.save_all_checkpoints):
            checkpoint_state = create_checkpoint_state(
                model, optimizer, scaler, epoch, best_acc1, best_loss1,
                best_loss_ind, distill_steps, grad_acc, best_rec, args
            )
            
            if args.save_all_checkpoints:
                checkpoint_filename = f"{checkpoint_prefix}_epoch_{epoch}.pth"
            else:
                checkpoint_filename = f"{checkpoint_prefix}_latest.pth"
            
            save_checkpoint(checkpoint_state, False, checkpoint_filename)

        # evaluate on validation set
        if epoch > 400 * int(5 / args.update_steps):
            args.test_freq = 10 * int(5 / args.update_steps)

        if (epoch - args.start_epoch + start_test_epoch) % args.test_freq == 0:
            if model.module.data.get_device() == 0:
                print('The current seed is {}'.format(torch.seed()))
            if model.module.data.get_device() == 0:
                print('The current lr is: {}'.format(model.module.lr))
            if model.module.data.get_device() == 0:
                print('Testing Results:')

            test_acc, test_loss, scores = test([test_loader, train_loader1, train_loader2], model, criterion, args)
            if model.module.data.get_device() == 0:
                print(test_acc)

            tmp_index = test_acc[2].index(max(test_acc[2]))

            if model.module.data.get_device() == 0 and args.wandb:
                wandb.log({"loss": test_loss,
                           "epoch": int(epoch * args.update_steps / 5), 'distill_steps': distill_steps,
                           "grad_norm": grad_tmp[-1],
                           "train_acc": test_acc[2][-1], "train_acc_full": test_acc[1][-1], "test_acc": test_acc[0][-1],
                           "curr": model.module.curriculum})

            if model.module.data.get_device() == 0 and args.wandb:
                image_log_dict = {}
                
                # Use context manager for memory efficient data copying
                with torch.no_grad():
                    curr_distilled_data = model.module.data.clone().cpu().detach()

                    # inverse the zca one patch at a time, then combine the patches to a poster (for visualization)
                    if zca_inverse is not None:
                        patches = get_crops_from_poster(curr_distilled_data, image_size_x, image_size_y,
                                                        args.patch_num_x, args.patch_num_y)
                        patches_shape = patches.shape
                        patches = patches.reshape(args.patch_num_x,
                                                  args.patch_num_y,
                                                  *patches_shape[1:]).permute(1, 0, 3, 4, 2).reshape(-1, *patches_shape[1:])

                        patches = \
                            np.ascontiguousarray(patches, dtype=np.float32).reshape(patches_shape[0], -1).astype('float32')
                        patches = patches.dot(zca_inverse)
                        patches = torch.Tensor(patches.reshape(patches_shape).astype('float32'))
                        patches = patches.reshape(patches.shape[0], -1, 3)[:, :, [1, 2, 0]].permute(0, 2, 1).reshape(
                            patches_shape)
                        inverse_distilled = combine_images_with_fade(patches, args.poster_width, args.poster_height,
                                                                     args.patch_num_x, args.patch_num_y)[[2, 0, 1], :, :]
                        clip_val = 4
                        mean, std = inverse_distilled.mean(), inverse_distilled.std()
                        inverse_distilled = np.clip(inverse_distilled, a_min=mean - clip_val * std,
                                                    a_max=mean + clip_val * std)
                        image_log_dict['inverse_zca_poster'] = wandb.Image(inverse_distilled)
                        
                        # Clean up intermediate variables
                        del patches, patches_shape, inverse_distilled

                    image_log_dict['distilled_poster'] = wandb.Image(
                        curr_distilled_data.squeeze().numpy().transpose(1, 2, 0))
                    
                    # Clean up data copy
                    del curr_distilled_data
                    
                wandb.log(image_log_dict)
                
                # Clean up after logging
                del image_log_dict
                torch.cuda.empty_cache()
                gc.collect()

            # remember best acc@1 and save checkpoint
            is_best = test_acc[2][tmp_index] > best_acc1
            if is_best:
                best_acc1 = max(test_acc[2][tmp_index], best_acc1)
                if model.module.data.get_device() == 0:
                    best_rec['acc'] = test_acc[2][tmp_index]
                    best_rec['test'] = test_acc[0]
                    best_rec['train'] = test_acc[2]
                    best_rec['ind'] = tmp_index
                    best_rec['epoch'] = epoch + 1
                    best_rec['data'] = model.module.data.clone().cpu().detach().numpy()
                    if args.train_y:
                        best_rec['label'] = model.module.label.data.cpu().clone().numpy()

                    # Save best accuracy checkpoint
                    checkpoint_state = create_checkpoint_state(
                        model, optimizer, scaler, epoch, best_acc1, best_loss1,
                        best_loss_ind, distill_steps, grad_acc, best_rec, args
                    )
                    best_acc_filename = f"{checkpoint_prefix}_best_acc.pth"
                    save_checkpoint(checkpoint_state, True, filename=best_acc_filename)

            is_best_loss = test_loss < best_loss1
            if is_best_loss:
                best_loss1 = test_loss
                best_loss_ind = epoch

                if model.module.data.get_device() == 0:
                    # Save best loss checkpoint
                    checkpoint_state = create_checkpoint_state(
                        model, optimizer, scaler, epoch, best_acc1, best_loss1,
                        best_loss_ind, distill_steps, grad_acc, best_rec, args
                    )
                    best_loss_filename = f"{checkpoint_prefix}_best_loss.pth"
                    save_checkpoint(checkpoint_state, True, filename=best_loss_filename)

                    # Also save individual poster files for backward compatibility
                    file_name = wandb.run.id if args.wandb else 'PoDD_run'
                    torch.save(model.module.data.clone().cpu().detach(), f'checkpoints/{file_name}_poster.pt')
                    if args.train_y:
                        torch.save(model.module.label.clone().cpu().detach(), f'checkpoints/{file_name}_label.pt')

            else:
                if epoch >= best_loss_ind + 200:
                    best_loss_ind = epoch
                    if args.ddtype == 'curriculum':
                        if args.cctype == 0:
                            if model.module.curriculum == args.minwindow:
                                break

                        elif args.cctype == 1:
                            if model.module.curriculum == args.totwindow - args.window:
                                break

                            model.module.curriculum += args.window
                            model.module.curriculum = min(args.totwindow - args.window, model.module.curriculum)

            print('train loss {}, epoch {}, best loss {}, best_epoch {}'.format(test_loss, epoch,
                                                                                best_loss1, best_loss_ind))


def train(train_loader1, train_loader2, model, criterion, optimizer, epoch, device, distill_steps, args, scaler):
    print('Check the length of the training dataset {}'.format(len(train_loader1.dataset)))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(len(train_loader1),
                             [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model.module.net.train()

    end = time.time()

    grad_acc = []
    if model.module.cctype == 2:
        shared_curriculum = torch.tensor(random.randint(args.minwindow, args.totwindow - args.window)).to(device)

        model.module.curriculum = shared_curriculum.item()

    if model.module.cctype == 3:
        model.module.curriculum = 0
        model.module.window = random.randint(args.window, args.totwindow)

    print('GPU_{}_using curriculum {} with window {}'.format(args.rank, model.module.curriculum, model.module.window))

    for train1 in enumerate(tqdm(train_loader1)):
        data_time.update(time.time() - end)

        i, (inputs, targets) = train1
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        '''
        # Debug: Check target values
        if i == 0:  # Only print for first batch
            print(f"Debug: Batch {i} - targets shape: {targets.shape}, values: {targets}")
            print(f"Debug: Min target: {targets.min()}, Max target: {targets.max()}")
            if targets.min() < 0 or targets.max() >= model.module.num_classes:
                print(f"ERROR: Target values out of range! Expected [0, {model.module.num_classes})")
        '''
        
        # Mixed precision forward pass
        if args.use_mixed_precision:
            with autocast():
                output, _ = safe_forward(model, inputs)
                '''
                # Debug: Check output shape
                if i == 0:  # Only print for first batch
                    print(f"Debug: Output shape: {output.shape}, Expected: (batch_size, {model.module.num_classes})")
                '''    
                loss = criterion(output, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accumulation_steps
        else:
            output, _ = safe_forward(model, inputs)
            '''
            # Debug: Check output shape
            if i == 0:  # Only print for first batch
                print(f"Debug: Output shape: {output.shape}, Expected: (batch_size, {model.module.num_classes})")
            '''    
            loss = criterion(output, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / args.grad_accumulation_steps

        # measure accuracy and record loss
        acc = accuracy(output, targets)
        losses.update(loss.item() * args.grad_accumulation_steps, inputs.size(0))
        top1.update(acc, inputs.size(0))

        # Mixed precision backward pass
        if args.use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only perform optimization step every grad_accumulation_steps
        if (i + 1) % args.grad_accumulation_steps == 0:
            # More efficient memory clearing
            torch.cuda.empty_cache()

            # Calculate gradient norm without keeping references
            with torch.no_grad():
                # Scale gradients back to original scale for norm calculation
                grad_tensor = optimizer.param_groups[0]['params'][0].grad
                if grad_tensor is not None:
                    # Unscale gradients before calculating norm
                    if args.use_mixed_precision:
                        scaler.unscale_(optimizer)
                    grad_norm = calculate_grad_norm(torch.norm(grad_tensor, dim=1).detach())
                else:
                    grad_norm = 0.0
            
            grad_acc.append(grad_norm)
            
            # obtain the ema norm and perform gradient clipping
            with torch.no_grad():
                grad_tensor = optimizer.param_groups[0]['params'][0].grad
                if grad_tensor is not None:
                    clip_coef = model.module.ema_update(
                        (torch.norm(grad_tensor, dim=1) ** 2).sum().item() ** 0.5)
                else:
                    clip_coef = 1.0

            torch.nn.utils.clip_grad_norm_(model.module.data, max_norm=clip_coef * 2)

            # Mixed precision optimizer step
            if args.use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            model.module.net.zero_grad()

        if args.train_y:
            with torch.no_grad():
                model.module.label.data = torch.clip(model.module.label.data, min=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        distill_steps += 1

        # Periodic memory cleanup
        if i % 10 == 0:  # Every 10 iterations
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 6) % args.print_freq == 0 and model.module.data.get_device() == 0:
            progress.display(i + 6)

    # Clear accumulated gradients after each epoch
    grad_acc = grad_acc[-100:]  # Keep only last 100 gradient norms
    
    # Handle remaining gradients if any
    if len(train_loader1) % args.grad_accumulation_steps != 0:
        # More efficient memory clearing
        torch.cuda.empty_cache()

        # Calculate gradient norm without keeping references
        with torch.no_grad():
            grad_tensor = optimizer.param_groups[0]['params'][0].grad
            if grad_tensor is not None:
                # Unscale gradients before calculating norm
                if args.use_mixed_precision:
                    scaler.unscale_(optimizer)
                grad_norm = calculate_grad_norm(torch.norm(grad_tensor, dim=1).detach())
            else:
                grad_norm = 0.0
        
        grad_acc.append(grad_norm)
        
        # obtain the ema norm and perform gradient clipping
        with torch.no_grad():
            grad_tensor = optimizer.param_groups[0]['params'][0].grad
            if grad_tensor is not None:
                clip_coef = model.module.ema_update(
                    (torch.norm(grad_tensor, dim=1) ** 2).sum().item() ** 0.5)
            else:
                clip_coef = 1.0

        torch.nn.utils.clip_grad_norm_(model.module.data, max_norm=clip_coef * 2)

        # Mixed precision optimizer step
        if args.use_mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()
        model.module.net.zero_grad()
    
    return grad_acc, losses.avg, distill_steps


# use pair_aug with train will apply a deterministic augmentation for all the data
def set_up_interventions(args):
    syn_intervention = ImageIntervention(
        'syn_aug',
        args.syn_strategy,
        phase='test',
        not_single=args.comp_aug
    )
    real_intervention = ImageIntervention(
        'real_aug',
        args.real_strategy,
        phase='test',
        not_single=args.comp_aug_real
    )
    # This is a customizable prob \in [0, 1]
    intervention_prob = 1.0

    return syn_intervention, real_intervention, intervention_prob


def calculate_grad_norm(grad_norm):
    return grad_norm[grad_norm > 1e-5].mean().item()


def one_gpu_test_2(val_loader, model, args):
    def run_validate(loader, base_progress=0):
        acc_ind = []
        with torch.no_grad():
            for images, target in loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    target = target.cuda()

                output, _ = model.module.test(images)
                if len(target.shape) == 2:
                    target = target.max(1)[1]
                acc_ind.append(accuracy_ind(output, target))
        return torch.cat(acc_ind, 0)

    acc_ind = run_validate(val_loader)
    return acc_ind.to(torch.int)


def test(data_loaders, model, criterion, args):
    if args.dataset == 'tiny-imagenet-200':
        epoch_list = [100, 300, 600, 1000, 2000]
    else:
        epoch_list = [300, 600, 1000, 2000]
    acc = []
    for i in range(len(data_loaders)):
        acc.append([0] * (len(epoch_list)))
    loss = 0

    for train_ind in range(args.num_train_eval):
        model.module.init_train(0, init=True)
        start_epoch = 0
        for train_time in range(len(epoch_list)):
            model.train()
            model.module.net.train()
            model.module.init_train(epoch_list[train_time] - start_epoch)
            for loader_i in range(len(data_loaders)):
                tmp_acc, tmp_loss = default_test(data_loaders[loader_i], model, criterion, args)
                acc[loader_i][train_time] += tmp_acc
            start_epoch = epoch_list[train_time]
        loss += tmp_loss
        if train_ind == 0:
            acc_ind = one_gpu_test_2(data_loaders[2], model, args)
        else:
            acc_ind += one_gpu_test_2(data_loaders[2], model, args)

    acc_ind = args.num_train_eval - acc_ind

    for loader_i in range(len(data_loaders)):
        acc[loader_i] = [acc_id / args.num_train_eval for acc_id in acc[loader_i]]
        if model.module.data.get_device() == 0:
            for train_time in range(len(epoch_list)):
                if model.module.data.get_device() == 0:
                    print('Training for {} epoch: {}'.format(epoch_list[train_time], acc[loader_i][train_time]))
    return acc, tmp_loss / args.num_train_eval, acc_ind


def default_test(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for images, target in loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    target = target.cuda()

                output, _ = model.module.test(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                if len(target.shape) == 2:
                    target = target.max(1)[1]

                acc = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    run_validate(val_loader)

    if model.module.data.get_device() == 0:
        progress.display_summary()

    return top1.avg, losses.avg
