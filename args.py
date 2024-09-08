import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model Training Parameters')
    # Dataset hyperparameters
    parser.add_argument('--dataset_paths', type=str, nargs='+', help='List of dataset paths')
    parser.add_argument('--calibration_paths', type=str, nargs='+', help='List of calibration paths')
    
    # Model hyperparameters
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--hidden_dim', type=int, default=142, help='Hidden dimension (d)')
    parser.add_argument('--order', type=int, default=87, help='State space order (N)')
    parser.add_argument('--dt_min', type=float, default=1e-3, help='Minimum discretization step size')
    parser.add_argument('--dt_max', type=float, default=1e-1, help='Maximum discretization step size')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels (M)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--total_steps', type=int, default=28000, help='Total number of training steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--step_size', type=int, default=300, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--loss_type', type=str, default='bce', help='Loss function type')

    # Other configurations
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use, e.g., "0,1"')
    parser.add_argument('--visualization_stride', type=int, default=100, help='Steps between visualizations')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')

    args = parser.parse_args()
    return args
