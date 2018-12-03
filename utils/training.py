from torch.autograd import Variable
import torch

def train(epochs, model, optim, loader, loss_fn, use_gpu):
    
    # X, Y_target = loader
    model.train()
    train_loss = []
    print("\n--------------------------------------------------------------------")
    print("TRAINING MODEL...", "\n\n   Epoch |", "Training Loss |")
    N = len(loader)
    for epoch in range(epochs+1):
        
        avg_loss = 0.0
        for n, batch in enumerate(loader):
            X, Y = batch
            Y_target = Y.view(-1)
            if use_gpu:
                X, Y_target = Variable(X.cuda()), Variable(Y_target.cuda())
        
            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, Y_target)
            avg_loss += loss.item()
        
            del X, Y, Y_target
        
            # Print to terminal
            print("{:>8} | {:13} |".format(round(float(epoch)+n/N,2), round(loss.item(),6) ) )
        
            # Backward pass and optimize
            optim.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            optim.step()
            del loss
        
        avg_loss /= N
        train_loss.append( avg_loss )
        
        
        # logits = model(X)
        # 
        # loss = loss_fn(logits, Y_target)
        # 
        # train_loss.append( loss.item() )
        # 
        # # Print to terminal
        # print("{:>5} | {:13} |".format(epoch, round(loss.item(),6) ) )
        # 
        # # Backward pass and optimize
        # optim.zero_grad()
        # loss.backward()
        # optim.step()
        # 
    return train_loss