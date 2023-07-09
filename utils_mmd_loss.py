import classes


'''
Train, Test, Attack, and MMD-Synced Training Functions for MMD Loss Function experiments as seen in _________

Conducts training of Client and Shadow Model in parallel:
    - Uses MMD Loss between smash layer gradients of client and shadow to match the shadow
        model learning behavior to that of the client
    - Note: We use the smash layer gradient computed from the last batch of data of the client model and loader
        This is not as efficient as computing the smash layer at different batches, but is most representative
            of what an attacker may have in a real scenario
        Further testing should be done on how different batch sizes may influence our results

'''

device = 'mps'

#used for sync_train function to avoid training all epochs at once
def train_one_round(model, loader, optimizer, loss_fn, count):
  '''
  Helper function for train_sync function:
    - Conducts one round of training for client model using client private dataset
    - Saves smash layer computed by first half of SplitNN (true client)

  Args:
    model (nn.Module)
    loader (DataLoader)
    optimizer (optim.Optimizer)
    loss_fn (nn.Loss)
    count (int: # of current training epoch)

  Returns:
    - client_smash_layer (torch.tensor)
  '''
  running_loss = 0
  smash_output = None
  for images, labels in loader:
    images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    optimizer.zero_grad() # Training pass
    
    smash_output = model.first_part(images)
    output = model.second_part(smash_output)

    loss = loss_fn(output, labels)
  
    loss.backward() #This is where the model learns by backpropagating
    optimizer.step() #And optimizes its weights here

    running_loss += loss.item()
  else:
    print("Epoch {} - Training loss: {}".format(count + 1, running_loss/len(loader)))
  return smash_output

def train_tune_shadow(model, loader, sh_optim, loss_fn, cl_smash, count, tuner):
  '''
  Helper function for train_sync function:
    - Conducts one round of training for the shadow model using shadow dataset (of similar distribution to private client one)
    - Uses MMD Loss function to compute loss between shadow and client smash layers
    - tunes shadow model learning behavior to match that of the client model

  Args:
    model (nn.Module)
    loader (DataLoader)
    sh_optim (optim.Optimizer)
    loss_fn (nn.Loss)
    cl_smash (torch.Tensor)
    count (int: # of current training epoch)
    tuner (nn.Module(MMD_Loss))
  '''
  running_loss = 0
  for images, labels in loader:
    images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    sh_optim.zero_grad() # Training pass
    
    sh_smash = model.first_part(images)
    output = model.second_part(sh_smash)

    tuning_loss = tuner(sh_smash, cl_smash.clone().detach())
    shadow_loss = loss_fn(output, labels)
    total_loss = shadow_loss + tuning_loss

    total_loss.backward() #This is where the model learns by backpropagating
    sh_optim.step() #And optimizes its weights here

    running_loss += total_loss.item()
  else:
    print("Epoch {} - Training loss: {}".format(count + 1, running_loss/len(loader)))


def sync_train(epochs, client, shadow, cl_optim, sh_optim, cl_loss, sh_loss, cl_load, sh_load):
  '''
  Trains client and shadow model in parallel

  Args:
    epochs (int)
    client (nn.Module)
    shadow (nn.Module)
    cl_optim (optim.Optimizer)
    sh_optim (optim.Optimizer)
    cl_loss (nn.Loss)
    sh_loss (nn.Loss)
    cl_load (DataLoader)
    sh_load (DataLoader)
  '''
  count = 0
  tuner = classes.MMDLoss().to(device)

  for count in range(epochs):
    cl_smash = train_one_round(client, cl_load['train'], cl_optim, cl_loss, count)
    train_tune_shadow(shadow, sh_load['train'], sh_optim, sh_loss, cl_smash, count, tuner)