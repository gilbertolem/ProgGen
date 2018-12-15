from torch.autograd import Variable
import torch


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
    
    avg_loss = 0.0

    for n, batch in enumerate(train_loader):
        x, y, w = batch
        y = y.view(-1)
        w = w.view(-1)
        if use_gpu:
            x, y, w = Variable(x.cuda()), Variable(y.cuda()), Variable(w.cuda())
    
        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y, w)
        avg_loss += loss.item()
    
        del x, y, w
    
        # Backward pass and optimize
        optim.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        optim.step()
        del loss
        torch.cuda.empty_cache()
        
    avg_loss /= len(train_loader)
    return avg_loss
        

def train(epochs, model, optim, train_loader, val_loader, loss_fn, use_gpu, model_name = 'model'):
    
    model.train()
    train_losses = []
    val_losses = []
    print("\n--------------------------------------------------------------------")
    print("TRAINING MODEL...", "\n\n   Epoch | Training Loss | Val. Loss")
    
    best_loss = float('Inf')
    
    for epoch in range(epochs+1):
        
        train_loss = train_iteration(model, optim, train_loader, loss_fn, use_gpu)
        val_loss = evaluate(model, val_loader, loss_fn, use_gpu)
        
        # Print to terminal
        print("{:>8} | {:13} | {:9}".format(epoch, round(train_loss,2), round(val_loss,2) ) )
        
        train_losses.append( train_loss )
        val_losses.append( val_loss )
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.cpu(), 'models/'+model_name+'.pt')

    return train_losses, val_losses