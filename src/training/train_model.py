import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from src.utils.helpers import plot_losses, mape_loss, save_losses, max_error, r2_score, weighted_MSE_loss
from torch.utils.tensorboard import SummaryWriter
from src.data.utilities import denorm_values
from src.constants import EIGEN_ENERGY, QUALITY_FACTOR, THRESHOLD_GAIN


def train_model(model, train_dataset, validate_dataset, evaluate_dataset, args):
    """
    Train the model on given datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Selected device: {device}')
    model = model.to(device)

    writer = SummaryWriter()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience)
    criterion = torch.nn.MSELoss()

    train_mse_losses = []
    train_mape_losses = []
    train_max_errors = []
    train_r2_scores = []

    validate_mse_losses = []
    validate_mape_losses = []
    validate_max_errors = []
    validate_r2_scores = []

    logging.info("Train started")
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        start_time = time.time()
        model.train()
        train_loss, train_mape, train_max_error, train_r2 = process_epoch(train_dataloader, model, criterion, optimizer, args.model_type, device, training=True)
        train_mse_losses.append(train_loss)
        train_mape_losses.append(train_mape)
        train_max_errors.append(train_max_error)
        train_r2_scores.append(train_r2)

        model.eval()
        with torch.no_grad():
            validate_loss, validate_mape, validate_max_error, validate_r2 = process_epoch(validate_dataloader, model, criterion, optimizer, args.model_type, device, training=False)
        validate_mse_losses.append(validate_loss)
        validate_mape_losses.append(validate_mape)
        validate_max_errors.append(validate_max_error)
        validate_r2_scores.append(validate_r2)

        writer.add_scalars('Loss/MSE', {'Train': train_loss, 'Validation': validate_loss}, epoch)
        writer.add_scalars('Loss/MAPE', {'Train': train_mape, 'Validation': validate_mape}, epoch)
        writer.add_scalars('Loss/Max Error', {'Train': train_max_error, 'Validation': validate_max_error}, epoch)
        writer.add_scalars('Loss/R2 Score', {'Train': train_r2, 'Validation': validate_r2}, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        duration = time.time() - start_time

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        scheduler.step(validate_loss)

        logging.info(f'Epoch: {epoch + 1}, Epoch time: {duration:.3f}, Train MSE: {train_loss:.16f}, Train MAPE: {train_mape:.3f}, Validation MAPE: {validate_mape:.3f}, Train R2: {train_r2: .3f}, Current LR: {current_lr}')

        #if validate_loss < best_loss:
        #    best_loss = validate_loss
        #    torch.save(model.state_dict(), args.weight_path)

        if validate_mape < best_loss:
            best_loss = validate_mape
            torch.save(model.state_dict(), args.weight_path)


    save_losses(args.weight_path, train_mse_losses, train_mape_losses, train_max_errors, train_r2_scores,
                validate_mse_losses, validate_mape_losses, validate_max_errors, validate_r2_scores)

    writer.close()

    logging.info("Train complete")

    logging.info("Evaluate started")
    model.eval()
    with torch.no_grad():
        evaluate_mse_loss, evaluate_mape, evaluate_max_error, evaluate_r2 = process_epoch(evaluate_dataloader, model, criterion, optimizer, args.model_type, device, training=False)

    logging.info(f"Test results - MSE: {evaluate_mse_loss:.16f}, MAPE: {evaluate_mape:.3f}, Max Error: {evaluate_max_error}, R2 Score: {evaluate_r2}")

    logging.info("Evaluate complete")

    plot_losses(train_mse_losses, validate_mse_losses, "MSE Losses")
    plot_losses(train_mape_losses, validate_mape_losses, "MAPE Losses")
    plot_losses(train_max_errors, validate_max_errors, "Max Errors")
    plot_losses(train_r2_scores, validate_r2_scores, "R2 Scores")


def process_epoch(dataloader, model, criterion, optimizer, model_type, device, training=True):
    total_loss = 0
    total_mape_loss = 0
    total_max_error = 0
    total_r2 = 0

    for batch, data in enumerate(dataloader):
        if training:
            optimizer.zero_grad()
        input_tokens, targets = data['input_tokens'].to(device), data['output_tokens'].to(device)
        output = model(input_tokens)
        if model_type == QUALITY_FACTOR:
            loss = weighted_MSE_loss(output, targets)
        else:
            loss = criterion(output, targets)
        denorm_output, denorm_target = denorm_values(output, targets, model_type)
        mape = mape_loss(denorm_output, denorm_target)
        max_err = max_error(denorm_output, denorm_target)
        r2 = r2_score(denorm_output, denorm_target)

        #r2_error = 1 - r2   # for r2 minimization

        if training:
            #r2_error.backward()  # for r2 minimization
            mape.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mape_loss += mape.item()
        total_max_error = max(max_err, total_max_error)
        total_r2 += r2.item()

    avg_loss = total_loss / (batch+1)
    avg_mape = total_mape_loss / (batch+1)
    avg_r2 = total_r2 / (batch+1)
    return avg_loss, avg_mape, total_max_error, avg_r2
