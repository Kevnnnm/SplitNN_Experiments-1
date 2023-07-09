import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from datas import only_numbers, FEMNIST
from utils_basic import train, test, train_attack, attack
import classes

device = 'mps'

def train_models(client, shadow, attack_model, client_loaders, shadow_loaders, initial):
    epochs = 12

    client_criterion = nn.NLLLoss()
    client_optimizer = optim.SGD(client.parameters(), lr=0.003, momentum=0.9)
    shadow_criterion = nn.NLLLoss()
    shadow_optimizer = optim.SGD(shadow.parameters(), lr=0.003, momentum=0.9)
    attack_optimizer = optim.Adam(attack_model.parameters(), lr = 1e-3)

    client.train()
    train(epochs, client, client_loaders['train'], client_criterion, client_optimizer)
    client.eval()
    test(client, client_loaders['test'])
    
    
    shadow.train()
    train(epochs, shadow, shadow_loaders['train'], shadow_criterion, shadow_optimizer)
    shadow.eval()
    test(shadow, shadow_loaders['test'])
    
    
    attack_model.train()
    train_attack(epochs, attack_model, shadow, attack_optimizer, shadow_loaders['train'], shadow_loaders['test'])

    if initial:
        torch.save(client.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/client_initialed.pth")
        torch.save(shadow.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/shadow_initialed.pth")
        torch.save(attack_model.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/attack_initialed.pth")

        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/client_initialed.pth")
        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/shadow_initialed.pth")
        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/attack_initialed.pth")
    else:
        torch.save(client.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/client.pth")
        torch.save(shadow.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/shadow.pth")
        torch.save(attack_model.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/attack.pth")

        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/client.pth")
        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/shadow.pth")
        print("Saved PyTorch Model State to SplitNN_Experiments/Trained_Models/nums/basic/attack.pth")
    

def test_models(client, shadow, client_loaders, shadow_loaders):
    client.eval()
    shadow.eval()
    test(client, client_loaders['test'])
    test(shadow, shadow_loaders['test'])


def model_attack(client, attack_model, client_loaders):
    attack_model.eval()
    attack(attack_model, client, client_loaders)


def use_initializations(client, shadow, attack_model):
    torch.save(client.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/client_initialed.pth")
    torch.save(client.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/shadow_initialed.pth")
    torch.save(attack_model.state_dict(), "SplitNN_Experiments/Trained_Models/nums/basic/attack_initialed.pth") #just for file path purposes

    client.load_state_dict(torch.load("SplitNN_Experiments/Trained_Models/nums/basic/client_initialed.pth"))
    shadow.load_state_dict(torch.load("SplitNN_Experiments/Trained_Models/nums/basic/client_initialed.pth"))

    return client, shadow, attack_model


def main(initial = False):
    transform = transforms.Normalize((-0.5), (0.5))
    client_model, shadow_model, attack_model = classes.SplitNN().to(device), classes.ShadowNN().to(device), classes.Attacker().to(device)

    if initial: client_model, shadow_model, attack_model = use_initializations(client_model, shadow_model, attack_model)

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

    train_models(client_model, shadow_model, attack_model, client_loaders, shadow_loaders, initial)

    test_models(client_model, shadow_model, client_loaders, shadow_loaders)

    model_attack(client_model, attack_model, client_loaders)

#main(True)
#main()
