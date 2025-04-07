"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim
import torch.nn.functional as F

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.datasets.road_transforms import EgoTrackProcessor

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner", 
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    # Set OpenMP environment variable
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    # Setup device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Initialize model 
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load data with appropriate transform based on model type
    transform_pipeline = "default" if model_name == "cnn_planner" else "state_only"

    try:
        train_data = load_data(
            "drive_data/drive_data/train",
            transform_pipeline=transform_pipeline,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_data = load_data(
            "drive_data/drive_data/val", 
            transform_pipeline=transform_pipeline,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    global_step = 0
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epoch):
            metrics = {"train_loss": [], "val_loss": [], "train_longitudinal_loss": [], 
                      "val_longitudinal_loss": [], "train_lateral_loss": [], "val_lateral_loss": []}

            # Training
            model.train()
            for batch_idx, batch in enumerate(train_data):
                try:
                    # Move data to device
                    if model_name == "cnn_planner":
                        image = batch["image"].to(device)
                        target_waypoints = batch["waypoints"].to(device)
                        waypoints_mask = batch["waypoints_mask"].to(device)
                        
                        # Forward pass with image
                        pred_waypoints = model(image=image)
                    else:
                        track_left = batch["track_left"].to(device)
                        track_right = batch["track_right"].to(device)
                        target_waypoints = batch["waypoints"].to(device)
                        waypoints_mask = batch["waypoints_mask"].to(device)
                        
                        # Forward pass with track points
                        pred_waypoints = model(track_left=track_left, track_right=track_right)

                    # Extract longitudinal and lateral components
                    pred_longitudinal = pred_waypoints[waypoints_mask][:, 0]
                    target_longitudinal = target_waypoints[waypoints_mask][:, 0]
                    pred_lateral = pred_waypoints[waypoints_mask][:, 1]
                    target_lateral = target_waypoints[waypoints_mask][:, 1]

                    # Calculate losses
                    longitudinal_loss = F.l1_loss(pred_longitudinal, target_longitudinal)
                    lateral_loss = F.l1_loss(pred_lateral, target_lateral)
                    loss = longitudinal_loss + lateral_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    metrics["train_loss"].append(loss.item())
                    metrics["train_longitudinal_loss"].append(longitudinal_loss.item())
                    metrics["train_lateral_loss"].append(lateral_loss.item())
                    global_step += 1

                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue

            # Validation
            model.eval()
            with torch.inference_mode():
                for batch_idx, batch in enumerate(val_data):
                    try:
                        if model_name == "cnn_planner":
                            image = batch["image"].to(device)
                            target_waypoints = batch["waypoints"].to(device)
                            waypoints_mask = batch["waypoints_mask"].to(device)
                            pred_waypoints = model(image=image)
                        else:
                            track_left = batch["track_left"].to(device)
                            track_right = batch["track_right"].to(device) 
                            target_waypoints = batch["waypoints"].to(device)
                            waypoints_mask = batch["waypoints_mask"].to(device)
                            pred_waypoints = model(track_left=track_left, track_right=track_right)

                        # Calculate validation metrics
                        pred_longitudinal = pred_waypoints[waypoints_mask][:, 0]
                        target_longitudinal = target_waypoints[waypoints_mask][:, 0]
                        pred_lateral = pred_waypoints[waypoints_mask][:, 1]
                        target_lateral = target_waypoints[waypoints_mask][:, 1]

                        longitudinal_loss = F.l1_loss(pred_longitudinal, target_longitudinal)
                        lateral_loss = F.l1_loss(pred_lateral, target_lateral)
                        loss = longitudinal_loss + lateral_loss

                        metrics["val_loss"].append(loss.item())
                        metrics["val_longitudinal_loss"].append(longitudinal_loss.item())
                        metrics["val_lateral_loss"].append(lateral_loss.item())
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue

            # Log metrics if we have any
            if metrics["train_loss"] and metrics["val_loss"]:
                epoch_train_loss = np.mean(metrics["train_loss"])
                epoch_val_loss = np.mean(metrics["val_loss"])
                epoch_train_longitudinal_loss = np.mean(metrics["train_longitudinal_loss"])
                epoch_val_longitudinal_loss = np.mean(metrics["val_longitudinal_loss"])
                epoch_train_lateral_loss = np.mean(metrics["train_lateral_loss"])
                epoch_val_lateral_loss = np.mean(metrics["val_lateral_loss"])
                
                logger.add_scalar("train_loss", epoch_train_loss, epoch)
                logger.add_scalar("val_loss", epoch_val_loss, epoch)
                logger.add_scalar("train_longitudinal_loss", epoch_train_longitudinal_loss, epoch)
                logger.add_scalar("val_longitudinal_loss", epoch_val_longitudinal_loss, epoch)
                logger.add_scalar("train_lateral_loss", epoch_train_lateral_loss, epoch)
                logger.add_scalar("val_lateral_loss", epoch_val_lateral_loss, epoch)

                # Save best model
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")

                if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                    print(
                        f"\033[38;5;214mEpoch {epoch + 1:2d}/{num_epoch:2d}: "
                        f"train_loss={epoch_train_loss:.4f} "
                        f"val_loss={epoch_val_loss:.4f} "
                        f"train_longitudinal_loss={epoch_train_longitudinal_loss:.4f} "
                        f"val_longitudinal_loss={epoch_val_longitudinal_loss:.4f} "
                        f"train_lateral_loss={epoch_train_lateral_loss:.4f} "
                        f"val_lateral_loss={epoch_val_lateral_loss:.4f}\033[0m"
                    )

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Save final model
        try:
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
            print(f"Model saved to {log_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")
    return model, best_val_loss

if __name__ == "__main__":
    # best mlp args: lr = 3e-5, batch size = 64, num_epoch = 40
    # best transformer args: lr = 3e-5, batch size = 32, num_epoch = 50
    # best cnn args: lr = 1e-4, batch size = 32, num_epoch = 20

    # Model-specific hyperparameter configurations
    MODEL_CONFIGS = {
        "mlp_planner": {
            "lr": 3e-5,
            "batch_size": 64,
            "num_epoch": 40
        },
        # We'll try some different configurations for the transformer planner and save the best one
        "transformer_planner": [
            {
                "lr": 3e-5,
                "batch_size": 64,
                "num_epoch": 40,
                "name": "baseline"
            },
            {
                "lr": 1e-4,
                "batch_size": 32,
                "num_epoch": 50,
                "name": "fast_learn"
            },
            {
                "lr": 1e-5,
                "batch_size": 128,
                "num_epoch": 60,
                "name": "slow_learn"
            },
            {
                "lr": 5e-5,
                "batch_size": 96,
                "num_epoch": 45,
                "name": "balanced"
            },
            {
                "lr": 1e-4,
                "batch_size": 64,
                "num_epoch": 40,
                "name": "default"
            },
            {
                "lr": 3e-5,
                "batch_size": 32,
                "num_epoch": 50,
                "name": "fast_train"
            }
        ],
        "cnn_planner": {
            "lr": 1e-4,
            "batch_size": 64,
            "num_epoch": 50
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_name]
    
    if isinstance(config, list):
        # Try multiple configs
        best_lateral_error = float('inf')
        best_config = None
        best_model_state = None

        for cfg in config:
            print(f"\nTrying {args.model_name} config: {cfg['name']}")
            print(f"lr={cfg['lr']}, batch_size={cfg['batch_size']}, epochs={cfg['num_epoch']}")
            
            # Update args with current config
            args.lr = cfg['lr']
            args.batch_size = cfg['batch_size']
            args.num_epoch = cfg['num_epoch']
            
            # Train model
            model, val_lateral_error = train(**vars(args))
            
            if val_lateral_error < best_lateral_error:
                best_lateral_error = val_lateral_error
                best_config = cfg
                best_model_state = model.state_dict()
                
                # Save best model so far
                torch.save(best_model_state, 
                         Path(args.exp_dir) / f"{args.model_name}_best_lateral.th")
                
        print(f"\nBest {args.model_name} config: {best_config['name']}")
        print(f"Best validation lateral error: {best_lateral_error:.4f}")
        
    else:
        # Use single config
        args.lr = config["lr"]
        args.batch_size = config["batch_size"] 
        args.num_epoch = config["num_epoch"]
        train(**vars(args))