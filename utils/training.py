from torch.autograd import Variable
import torch
from scipy.stats import linregress


def stopping_criterion(val_losses, n_stop):
    epoch = len(val_losses)
    if epoch >= n_stop:
        min_idx = epoch - n_stop
        m = linregress(range(n_stop), val_losses[min_idx:])[0]
        return m > 0
    return False


def evaluate(model, val_loader, loss_fn, use_gpu):
    
    model.eval()
    avg_loss = 0.0

    with torch.no_grad():
        
        for n, batch in enumerate(val_loader):
            x, y, w = batch
            y = y.view(-1)
            w = w.view(-1)
            if use_gpu:
                x, y, w = Variable(x.cuda()), Variable(y.cuda()), Variable(w.cuda())
        
            # Forward pass
            logits = model(x)
            loss = loss_fn(logits, y, w)
            avg_loss += loss.item()
        
            del x, y, w, loss
            torch.cuda.empty_cache()
        
    avg_loss /= len(val_loader)
    return avg_loss


def train_iteration(model, optim, train_loader, loss_fn, use_gpu):
    
    model.train()
    for n, batch in enumerate(train_loader):
        x, y, w = batch
        y = y.view(-1)
        w = w.view(-1)
        if use_gpu:
            x, y, w = Variable(x.cuda()), Variable(y.cuda()), Variable(w.cuda())
    
        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y, w)

        del x, y, w
    
        # Backward pass and optimize
        optim.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        optim.step()
        del loss
        torch.cuda.empty_cache()


def print_losses(epoch, train, val):
    print("{:>8} | {:13} | {:9}".format(epoch, round(train, 2), round(val, 2)))


def train(epochs, model, optim, train_loader, val_loader, loss_fn, use_gpu, model_name='model', n_stop=20):
    
    print("\n--------------------------------------------------------------------")
    print("TRAINING MODEL...", "\n\n   Epoch | Training Loss | Val. Loss")
    
    best_loss = float('Inf')

    # Initial evaluation
    train_losses = [evaluate(model, train_loader, loss_fn, use_gpu)]
    val_losses = [evaluate(model, val_loader, loss_fn, use_gpu)]
    print_losses(0, train_losses[-1], val_losses[-1])

    for epoch in range(1, epochs+1):
        
        train_iteration(model, optim, train_loader, loss_fn, use_gpu)
        train_losses.append(evaluate(model, train_loader, loss_fn, use_gpu))
        val_losses.append(evaluate(model, val_loader, loss_fn, use_gpu))
        print_losses(epoch, train_losses[-1], val_losses[-1])

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.cpu(), 'models/'+model_name+'.pt')
            if use_gpu:
                model.cuda()

        if stopping_criterion(val_losses, n_stop):
            break

    return train_losses, val_losses