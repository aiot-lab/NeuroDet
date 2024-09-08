import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import datetime

from utils.model import lssModel
from utils.dataset import lssDataset
from utils.loss import customLoss
from args import get_args 

def generate_checkpoint_name(args):
    """Generate a descriptive checkpoint name based on the training arguments, including a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Format: YYYYMMDD-HHMMSS
    return (
        f"{timestamp}_"
        f"model_layers_{args.num_layers}_hidden_{args.hidden_dim}_order_{args.order}_"
        f"dtmin_{args.dt_min}_dtmax_{args.dt_max}_channels_{args.channels}_dropout_{args.dropout}_"
        f"lr_{args.learning_rate}_batch_{args.batch_size}_steps_{args.total_steps}_"
        f"optimizer_{args.optimizer}_decay_{args.weight_decay}_step_{args.step_size}_gamma_{args.gamma}_losstype_{args.loss_type}"
    )

def visualize_result(spectrum, pointcloud_pred, pointcloud_gt, writer, step):
    idx = np.random.randint(0, spectrum.shape[1])
    spectrum_np = spectrum[:, idx].squeeze().cpu().view(87, 128).numpy()
    pointcloud_pred_np = pointcloud_pred[:, idx].squeeze().detach().cpu().view(87, 128).numpy()  # Detach before converting to NumPy
    pointcloud_gt_np = pointcloud_gt[:, idx].squeeze().cpu().view(87, 128).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
    axes[0].set_title('Spectrum')

    axes[1].imshow(pointcloud_pred_np, cmap='viridis', aspect='auto')
    axes[1].set_title('Predicted Pointcloud')
    
    axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
    axes[2].set_title('Ground Truth Pointcloud')

    writer.add_figure('Visualization', fig, global_step=step)
    plt.close(fig)

def train():
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_name = generate_checkpoint_name(args)
    checkpoint_dir = os.path.join(args.save_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(args.log_dir, checkpoint_name)
    writer = SummaryWriter(log_dir=log_dir)

    scalar = torch.cuda.amp.GradScaler()

    # Create model
    model = lssModel(
        num_layers=args.num_layers,
        d=args.hidden_dim,
        order=args.order,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        channels=args.channels,
        dropout=args.dropout
    ).to(device)
    
    # Load dataset
    train_dataset = lssDataset(phase='train', dataset_paths=args.dataset_paths, calibration_paths=args.calibration_paths)
    val_dataset = lssDataset(phase='test', dataset_paths=args.dataset_paths, calibration_paths=args.calibration_paths)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset._collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=val_dataset._collate_fn)

    # Loss and optimizer
    criterion = customLoss(loss_type=args.loss_type).to(device)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_loss = float('inf')
    current_step = 0

    while current_step < args.total_steps:
        # Within your train loop
        for phase, loader in [('train', train_loader), ('test', val_loader)]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                accumulated_test_loss = 0.0
                num_test_steps = 0

            for batch in loader:
                spectrum = batch['spectrum'].to(device)
                pointcloud_gt = batch['pointcloud'].to(device)

                if phase == 'train':
                    with torch.cuda.amp.autocast():
                        pointcloud_pred = model(spectrum)
                        loss = criterion(pointcloud_pred, pointcloud_gt)

                    optimizer.zero_grad()
                    scalar.scale(loss).backward()
                    scalar.step(optimizer)
                    scalar.update()

                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Learning Rate', current_lr, current_step)

                    scheduler.step()

                    logger.info(f"Step {current_step}/{args.total_steps}, Loss: {loss.item():.6f}")
                    writer.add_scalar(f'Loss/{phase}', loss.item(), current_step)

                    if current_step % args.visualization_stride == 0:
                        visualize_result(spectrum, pointcloud_pred, pointcloud_gt, writer, current_step)

                    current_step += 1

                    if current_step >= args.total_steps:
                        break
                else:
                    with torch.no_grad():
                        pointcloud_pred = model(spectrum)
                        loss = criterion(pointcloud_pred, pointcloud_gt)
                        accumulated_test_loss += loss.item()
                        num_test_steps += 1

            if phase == 'test':
                average_test_loss = accumulated_test_loss / num_test_steps
                writer.add_scalar('Loss/test', average_test_loss, current_step)

                if average_test_loss < best_loss:
                    best_loss = average_test_loss
                    save_path = os.path.join(checkpoint_dir, checkpoint_name + ".pt")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Best model saved with test loss: {best_loss:.6f}")


            scheduler.step()

if __name__ == "__main__":
    train()