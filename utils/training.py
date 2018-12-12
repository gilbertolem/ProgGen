from torch.autograd import Variable
import torch

def evaluate(model, optim, val_loader, loss_fn, use_gpu):
    
    model.eval()
    avg_loss = 0.0
    N = len(val_loader)
    
    with torch.no_grad():
        
        for n, batch in enumerate(val_loader):
            X, Y = batch
            Y_target = Y.view(-1)
            if use_gpu:
                X, Y_target = Variable(X.cuda()), Variable(Y_target.cuda())
        
            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, Y_target)
            avg_loss += loss.item()
        
            del X, Y, Y_target, loss
            torch.cuda.empty_cache()
        
    avg_loss /= N
    return avg_loss

def train_iteration(model, optim, train_loader, loss_fn, use_gpu):
    
    avg_loss = 0.0
    N = len(train_loader)
    
    for n, batch in enumerate(train_loader):
        X, Y = batch
        Y_target = Y.view(-1)
        if use_gpu:
            X, Y_target = Variable(X.cuda()), Variable(Y_target.cuda())
    
        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, Y_target)
        avg_loss += loss.item()
    
        del X, Y, Y_target
    
        # Backward pass and optimize
        optim.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        optim.step()
        del loss
        torch.cuda.empty_cache()
        
    avg_loss /= N
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
        val_loss = evaluate(model, optim, val_loader, loss_fn, use_gpu)
        
        # Print to terminal
        print("{:>8} | {:13} | {:9}".format(round(float(epoch),0), round(train_loss,2), round(val_loss,2) ) )
        
        train_losses.append( train_loss )
        val_losses.append( val_loss )
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.cpu(), 'models/'+model_name+'.pt')

    return train_losses, val_losses