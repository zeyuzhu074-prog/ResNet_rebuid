import os 
import torch 
import matplotlib.pyplot as plt

# =========================
# evaluate
# =========================

def evaluate(model, dataloader, device):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total

    return acc


# =========================
# train
# =========================

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs
):

    train_errors = []

    test_errors = []

    best_acc = 0

    os.makedirs('./outputs/checkpoints', exist_ok=True)

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        correct = 0

        total = 0

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        # metrics
        epoch_loss = running_loss / total

        epoch_acc = 100. * correct / total

        test_acc = evaluate(model, test_loader, device)

        train_error = 100 - epoch_acc

        test_error = 100 - test_acc

        train_errors.append(train_error)

        test_errors.append(test_error)

        # save best
        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                './outputs/checkpoints/best_model.pth'
            )
            print(f'>>> New Best Model! Acc={best_acc:.2f}%')
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f'Epoch [{epoch+1}/{epochs}] '
            f'LR: {current_lr:.6f} '
            f'BestAcc: {best_acc:.2f}% '
            f'Loss: {epoch_loss:.4f} '
            f'Train Error: {train_error:.2f}% '
            f'Test Error: {test_error:.2f}%'
        )

    return {
        'train_error': train_errors,
        'test_error': test_errors
    }


