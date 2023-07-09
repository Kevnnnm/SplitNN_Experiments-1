import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from datas import only_numbers, FEMNIST

from utils_mmd_loss import sync_train
from utils_basic import test, train_attack, attack
from experiments_basic import test_models, model_attack
from time import time

import classes

'''
MMD-Loss function weight can be modified in classes.py
'''

device = 'mps'

def train_models(client, shadow, attack_model, client_loaders, shadow_loaders):
    epochs = 12
    client_criterion = nn.NLLLoss()
    client_optimizer = optim.SGD(client.parameters(), lr=0.003, momentum=0.9)
    shadow_criterion = nn.NLLLoss()
    shadow_optimizer = optim.SGD(shadow.parameters(), lr=0.003, momentum=0.9)
    attack_optimizer = optim.Adam(attack_model.parameters(), lr = 1e-3)

    client.train()
    shadow.train()
    sync_train(epochs, client, shadow, client_optimizer, shadow_optimizer, client_criterion, shadow_criterion, client_loaders, shadow_loaders)
    client.eval()
    shadow.eval()
    test(client, client_loaders['test'])
    test(shadow, shadow_loaders['test'])

    attack_model.train()
    train_attack(epochs, attack_model, shadow, attack_optimizer, shadow_loaders['train'], shadow_loaders['test'])
    
    torch.save(client.state_dict(), "SplitNN_Experiments/Trained_Models/nums/mmd_loss/client.pth")
    print("Saved PyTorch Model State to plitNN_Experiments/Trained_Models/nums/mmd_loss/client.pth")
    torch.save(shadow.state_dict(), "SplitNN_Experiments/Trained_Models/nums/mmd_loss/shadow.pth")
    print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/mmd_loss/shadow.pth")
    torch.save(attack_model.state_dict(), "SplitNN_Experiments/Trained_Models/nums/mmd_loss/attack.pth")
    print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/mmd_loss/attack.pth")

def main():
    transform = transforms.Normalize((-0.5), (0.5))
    client_model, shadow_model, attack_model = classes.SplitNN().to(device), classes.ShadowNN().to(device), classes.Attacker().to(device)

    client_train_ds = only_numbers(FEMNIST(train=True, transform = transform, client_num = 1))
    client_test_ds = only_numbers(FEMNIST(train=False, transform = transform, client_num = 1))

    shadow_train_ds = only_numbers(FEMNIST(train=True, transform = transform, client_num = 2))
    shadow_test_ds = only_numbers(FEMNIST(train=False, transform = transform, client_num = 2))

    client_loaders = {
        'train' : DataLoader(client_train_ds, batch_size=64, shuffle=True),
        'test'  : DataLoader(client_test_ds, batch_size=64,  shuffle=True),
    }
    shadow_loaders = {
        'train' : DataLoader(shadow_train_ds, batch_size=64, shuffle=True),
        'test'  : DataLoader(shadow_test_ds, batch_size=64,  shuffle=True),
    }

    train_models(client_model, shadow_model, attack_model, client_loaders, shadow_loaders)
    test_models(client_model, shadow_model, client_loaders, shadow_loaders)
    model_attack(client_model, attack_model, client_loaders)
main()
