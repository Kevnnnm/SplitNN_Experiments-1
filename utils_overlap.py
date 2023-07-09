import torch
from time import time
import classes
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse

'''
Train, Test, Attack, and MMD-Synced Training Functions for Overlapping Data experiments as seen in _________

Look to make the current label tensor process more efficient, probably need to make edits in datas file

'''

device = 'mps'


def overlap_train(num_epochs, model, loader, loss_fn, optimizer):
    """
    Trains model on dataset, loss function, and optimizer

    Args:
      num_epochs (int)
      model (nn.Module)
      loader (DataLoader)
      loss_fn (nn.Module)
      optimizer (optim.Optimizer)
    """

    time0 = time()
    for e in range(num_epochs):
        running_loss = 0
        for images, labels in loader:
          images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
          image_id_labels = [] #does loader reshuffle with each epoch? if not then we don't need to repopulate this list every epoch
          optimizer.zero_grad() # Training pass
          # print('images, ', images)
          output = model(images)
          for i in range(len(labels)):
            image_id_labels.append(labels[i][1])
          image_id_labels = torch.IntTensor(image_id_labels).to(device)
          loss = loss_fn(output, image_id_labels)
        
          loss.backward() #This is where the model learns by backpropagating
          optimizer.step() #And optimizes its weights here

          running_loss += loss.item()
        else:
          print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(loader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


def overlap_test(model, loader):
  '''
  Tests the model on a dataset (usually a test split) and prints accuracy

  Args:
    model (nn.Module)
    loader (DataLoader)
  '''
  correct_count, all_count = 0, 0
  for j, (images,labels) in enumerate(loader):
    for i in range(len(labels)):
      img, labels = images[i].view(1, 784).to('mps'), labels.to('mps')
      with torch.no_grad():
          logps = model(img)
      
      ps = torch.exp(logps)
      probab = list(ps.cpu().numpy()[0])
      pred_label = probab.index(max(probab))
      true_label = labels.cpu().numpy()[i][1]
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy =", (correct_count/all_count))


def overlap_attack(attack, model, loaders):

  '''
    Trains attack to reconstruct model training data, but uses non-client model data for training
                                                        (for slightly more realistic appraoch)
    Args:
        epochs (int)
        attack (nn.Module)
        model (nn.Module)
        optimizer (optim.Optimizer)
        attack_loader (DataLoader)
        test_loader (DataLoader)

    '''
  total_mse = 0
  for i, (data, targets) in enumerate(loaders):
    data, targets = data.to('mps'), targets.to('mps')
    data = data.reshape(data.shape[0], -1)
    target_outputs = model.first_part(data)
    recreated_data = attack(target_outputs)

    data_np = data.cpu().numpy()
    recreated_data_np = recreated_data.cpu().detach().numpy()

    total_mse += mse(data_np, recreated_data_np)
    
    if i < 3:
      # Display the original data
      plt.imshow(data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Original Data")
      plt.show()

      # Display the reconstructed data
      plt.imshow(recreated_data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Reconstructed Data")
      plt.show()
  print(f"AVG MSE: {total_mse / len(loaders)}")




def overlap_one_round(model, loader, optimizer, loss_fn, count):

  running_loss = 0
  for images, labels in loader:
    images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    image_id_labels = []
    if len(images) != 64: continue
    
    optimizer.zero_grad() # Training pass
    
    smash_output = model.first_part(images)
    output = model.second_part(smash_output)

    for i in range(len(labels)):
      image_id_labels.append(labels[i][1])
    image_id_labels = torch.IntTensor(image_id_labels).to(device)

    loss = loss_fn(output, image_id_labels)
  
    loss.backward() #This is where the model learns by backpropagating
    optimizer.step() #And optimizes its weights here

    running_loss += loss.item()
  else:
    print("Epoch {} - Client Training loss: {}".format(count + 1, running_loss/len(loader)))
  return smash_output

def overlap_tune_shadow(model, loader, sh_optim, loss_fn, cl_smash, count, tuner):
  running_loss = 0
  countt = 0
  for images, labels in loader:
    if len(images) != 64: continue
    image_id_labels = []
    images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    sh_optim.zero_grad() # Training pass
    
    sh_smash = model.first_part(images)
    output = model.second_part(sh_smash)
    for i in range(len(labels)):
      image_id_labels.append(labels[i][1])
    image_id_labels = torch.IntTensor(image_id_labels).to(device)

    tuning_loss = tuner(sh_smash, cl_smash.clone().detach())
    shadow_loss = loss_fn(output, image_id_labels)
    total_loss = shadow_loss + tuning_loss
    try:
      total_loss.backward() #This is where the model learns by backpropagating
    except:
      countt += 1
      print('fails:', countt)
      continue
    sh_optim.step() #And optimizes its weights here    
    running_loss += total_loss.item()
  else:
    print("Epoch {} - Tuned Training loss: {}".format(count + 1, running_loss/len(loader)))

def overlap_sync(epochs, client, shadow, cl_optim, sh_optim, cl_loss, sh_loss, cl_load, sh_load):
  tuner = classes.MMDLoss().to(device)
  for count in range(epochs):
    cl_smash = overlap_one_round(client, cl_load, cl_optim, cl_loss, count)
    overlap_tune_shadow(shadow, sh_load, sh_optim, sh_loss, cl_smash, count, tuner)