import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from sklearn.model_selection import KFold
from src.utils.helpers import plot_losses, mape_loss, save_losses, max_error, r2_score
from torch.utils.tensorboard import SummaryWriter


def train_model_k_fold(model, train_dataset, validate_dataset, evaluate_dataset, args, k=5):
    """
    Train the model on given datasets using k-fold cross-validation.
    """

    combined_inputs = train_dataset.inputs + validate_dataset.inputs
    combined_outputs = train_dataset.outputs + validate_dataset.outputs

    combined_data = [{"input_tokens": combined_inputs[i], "output_tokens": combined_outputs[i]} for i in
                     range(len(combined_inputs))]

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    writer = SummaryWriter()

    train_mse_losses = []
    train_mape_losses = []
    train_max_errors = []
    train_r2_scores = []

    validate_mse_losses = []
    validate_mape_losses = []
    validate_max_errors = []
    validate_r2_scores = []

    logging.info(f"Training with {k}-Fold Cross-Validation")
    best_loss = float('inf')
    fold_num = 1
    for train_indices, validate_indices in kfold.split(combined_data):
        train_subset = [combined_data[i] for i in train_indices]
        validate_subset = [combined_data[i] for i in validate_indices]

        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_subset, batch_size=args.batch_size, shuffle=False)
        evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_factor,
                                                               patience=args.scheduler_patience)
        criterion = torch.nn.MSELoss()

        logging.info(f"Training Fold {fold_num}")
        for epoch in range(args.num_epochs):
            start_time = time.time()
            model.train()
            train_loss, train_mape, train_max_error, train_r2 = process_epoch(train_dataloader, model, criterion,
                                                                              optimizer, training=True)
            train_mse_losses.append(train_loss)
            train_mape_losses.append(train_mape)
            train_max_errors.append(train_max_error)
            train_r2_scores.append(train_r2)

            model.eval()
            with torch.no_grad():
                validate_loss, validate_mape, validate_max_error, validate_r2 = process_epoch(validate_dataloader,
                                                                                              model, criterion,
                                                                                              optimizer, training=False)
            validate_mse_losses.append(validate_loss)
            validate_mape_losses.append(validate_mape)
            validate_max_errors.append(validate_max_error)
            validate_r2_scores.append(validate_r2)

            writer.add_scalars(f'KLoss/MSE', {'Train': train_loss, 'Validation': validate_loss}, epoch)
            writer.add_scalars(f'KLoss/MAPE', {'Train': train_mape, 'Validation': validate_mape}, epoch)
            writer.add_scalars(f'KLoss/Max Error', {'Train': train_max_error, 'Validation': validate_max_error}, epoch)
            writer.add_scalars(f'KLoss/R2 Score', {'Train': train_r2, 'Validation': validate_r2}, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'KLearning Rate', current_lr, epoch)

            duration = time.time() - start_time

            for name, param in model.named_parameters():
                writer.add_histogram(f"{name}", param, epoch)

            scheduler.step(validate_loss)

            logging.info(
                f'Epoch: {epoch + 1}, Fold: {fold_num}, Epoch time: {duration:.3f}, Train MSE: {train_loss:.16f}, Train MAPE: {train_mape:.3f}, Validation MAPE: {validate_mape:.3f}, Current LR: {current_lr}')

            if validate_loss < best_loss:
                best_loss = validate_loss
                torch.save(model.state_dict(), args.weight_path)

        fold_num += 1

    save_losses(args.weight_path, train_mse_losses, train_mape_losses, train_max_errors, train_r2_scores,
                validate_mse_losses, validate_mape_losses, validate_max_errors, validate_r2_scores)

    writer.close()

    logging.info("Train complete")

    logging.info("Evaluate started")
    model.eval()
    with torch.no_grad():
        evaluate_mse_loss, evaluate_mape, evaluate_max_error, evaluate_r2 = process_epoch(evaluate_dataloader, model,
                                                                                          criterion, optimizer,
                                                                                          training=False)

    logging.info(
        f"Test results - MSE: {evaluate_mse_loss:.16f}, MAPE: {evaluate_mape:.3f}, Max Error: {evaluate_max_error}, R2 Score: {evaluate_r2}")

    logging.info("Evaluate complete")

    plot_losses(train_mse_losses, validate_mse_losses, "MSE Losses")
    plot_losses(train_mape_losses, validate_mape_losses, "MAPE Losses")
    plot_losses(train_max_errors, validate_max_errors, "Max Errors")
    plot_losses(train_r2_scores, validate_r2_scores, "R2 Scores")


def process_epoch(dataloader, model, criterion, optimizer, training=True):
    total_loss = 0
    total_mape_loss = 0
    total_max_error = 0
    total_r2 = 0
    count = 0

    for batch, data in enumerate(dataloader):
        if training:
            optimizer.zero_grad()
        input_tokens, targets = data['input_tokens'], data['output_tokens']
        output = model(input_tokens)
        loss = criterion(output, targets)
        mape = mape_loss(output, targets)
        max_err = max_error(output, targets)
        r2 = r2_score(output, targets)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mape_loss += mape.item()
        total_max_error = max(max_err, total_max_error)
        total_r2 += r2
        count += 1

    avg_loss = total_loss / count
    avg_mape = total_mape_loss / count
    avg_r2 = total_r2 / count
    return avg_loss, avg_mape, total_max_error, avg_r2
