import torch
from time import time
import classes
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse

'''
Train, Test, Attack functions for basic SplitNN experiments as seen in ________
'''


#set device to whatever you please
device = 'mps'

def train(num_epochs, model, loader, loss_fn, optimizer):
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
          optimizer.zero_grad() # Training pass
          output = model(images)
          loss = loss_fn(output, labels)
        
          loss.backward() #This is where the model learns by backpropagating
          optimizer.step() #And optimizes its weights here

          running_loss += loss.item()
        else:
          print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(loader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


def test(model, loader):
  '''
  Tests the model on a dataset (usually a test split) and prints accuracy

  Args:
    model (nn.Module)
    loader (DataLoader)
  '''
  correct_count, all_count = 0, 0
  for images,labels in loader:
    for i in range(len(labels)):
      img, labels = images[i].view(1, 784).to('mps'), labels.to('mps')
      with torch.no_grad():
          logps = model(img)
      
      ps = torch.exp(logps)
      probab = list(ps.cpu().numpy()[0])
      pred_label = probab.index(max(probab))
      true_label = labels.cpu().numpy()[i]
      if true_label == pred_label:
        correct_count += 1
      all_count += 1

  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy =", (correct_count/all_count))


def train_attack(epochs, attack, model, optimizer, attack_loader, test_loader):
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
    for e in range(epochs):
        running_loss = 0
        for data, targets in attack_loader:
            data, targets = data.to('mps'), targets.to('mps')
            data = data.reshape(data.shape[0], -1)
            optimizer.zero_grad()

            target_outputs = model.first_part(data) #get outputs from first half of SplitNN

            attack_outputs = attack(target_outputs) #recreate data

            loss = ((data - attack_outputs)**2).mean() # We want attack outputs to resemble the original data

            loss.backward() # Update the attack model
            optimizer.step()
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(attack_loader)))
    total_mse = 0
    attack.eval()

    '''
    Compute MSE between original training data and reconstructed data
    Displays first 3 reconstructions and original images
    '''
    for i, (data, targets) in enumerate(test_loader):
        data, targets = data.to('mps'), targets.to('mps')
        #print(data.shape)
        data = data.reshape(data.shape[0], -1)
        target_outputs = model.first_part(data)
        recreated_data = attack(target_outputs)

        data_np = data.cpu().numpy()
        recreated_data_np = recreated_data.cpu().detach().numpy()

        total_mse += mse(data_np, recreated_data_np)
        
        if i < 3:

            # print(data_np.shape)
            # print(recreated_data_np.shape)

            # Display the original data
            plt.imshow(data_np[-1].reshape(28, 28), cmap='gray')
            plt.title("Original Data")
            plt.show()

            # Display the reconstructed data
            plt.imshow(recreated_data_np[-1].reshape(28, 28), cmap='gray')
            plt.title("Reconstructed Data")
            plt.show()
    print(f"AVG MSE: {total_mse / len(test_loader)}")


def attack(attack, model, loaders):
  '''
  Calculate MSE between attacker reconstruction and original training data 
  
  Args:
    attack (nn.Module)
    model (nn.Module)
    loaders (dict) -> loaders{
                            'train' : training_data_loader
                            'test' : test_data_loader
                            
                        }
  '''
  total_mse = 0
  for i, (data, targets) in enumerate(loaders['train']):
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
  print(f"AVG MSE: {total_mse / len(loaders['train'])}")