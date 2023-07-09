import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import torch

"""
This dataset is derived from the Leaf repository
(https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
dataset, grouping examples by writer. Details about Leaf were published in
"LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
"""

CACHE_DIR = 'data_cache'

class FEMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, client_num = 0):
        '''
        client_num = 0 - loads whole dataset with label tensor of size 3: Image ID, Character ID, User ID
        client_num = 1 - loads data of users with odd user-id, label tensor only holds Character ID
        client_num = 2 - ^^^ but for even users

        Character IDs:
            0-9: Digits 0-9
            10-25: Lower (or uppercase characters, I don't remember)
            26-61: Remaining characters from ^^^
        '''


        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        '''
        Preprocessing for data: 
            Check if there exists a preprocessed data cache for desired dataset
            If not, then process the data and store it into a .npy file
        '''
        try:
            # Load the preprocessed data from the cache
            data = np.load(os.path.join(CACHE_DIR, f"data_{client_num}_{train}.npy"), allow_pickle=True)
            data = torch.Tensor(data)
            label = np.load(os.path.join(CACHE_DIR, f"label_{client_num}_{train}.npy"), allow_pickle=True)
            if train: 
                user_label = np.load(os.path.join(CACHE_DIR, f"user_label_{client_num}_{train}.npy"), allow_pickle = True)
                self.dic_users = user_label
            self.data = data
            self.label = label
            
            print('Data loaded')
        #process data normally if preprocess does not exist
        except Exception as e:
            print(e)
            print('No Preprocessed data, processing now')
            train_clients, train_groups, train_data_temp, test_data_temp = read_data("leaf/data/femnist/data/train",
                                                                                     "leaf/data/femnist/data/test")
            #client_num 0 processes all data
            if client_num == 0:
                if self.train:
                    self.dic_users = {}
                    train_data_x = []
                    train_data_y = []
                    headers = ['Image ID', 'Char ID', 'Writer ID']
                    for i in range(len(train_clients)):
                        # if i == 100:
                        #     break
                        self.dic_users[i] = set()
                        l = len(train_data_x)
                        cur_x = train_data_temp[train_clients[i]]['x']
                        cur_y = train_data_temp[train_clients[i]]['y']
                        for j in range(len(cur_x)):
                            self.dic_users[i].add(j + l)
                            train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                            train_data_y.append((j + l, cur_y[j], i))
                    self.data = np.array(train_data_x)
                    self.label = np.array(train_data_y)
                    #self.label = pd.DataFrame(train_data_y, columns = headers)
                    self.dic_users = np.array(list(self.dic_users.items()))
                else:
                    test_data_x = []
                    test_data_y = []
                    for i in range(len(train_clients)):
                        cur_x = test_data_temp[train_clients[i]]['x']
                        cur_y = test_data_temp[train_clients[i]]['y']
                        l = len(test_data_x)
                        for j in range(len(cur_x)):
                            test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                            test_data_y.append((j + l, cur_y[j], i))
                    self.data = np.array(test_data_x)
                    self.label = np.array(test_data_y)
            #process partition half one of data
            if client_num == 1:
                if self.train:
                    self.dic_users = {}
                    train_data_x = []
                    train_data_y = []
                    for i in range(len(train_clients)):
                        if i % 2 == 1:
                            # if i == 100:
                            #     break
                            self.dic_users[i] = set()
                            l = len(train_data_x)
                            cur_x = train_data_temp[train_clients[i]]['x']
                            cur_y = train_data_temp[train_clients[i]]['y']
                            for j in range(len(cur_x)):
                                self.dic_users[i].add(j + l)
                                train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                                train_data_y.append(cur_y[j])
                    self.data = np.array(train_data_x)
                    self.label = np.array(train_data_y)
                    self.dic_users = np.array(list(self.dic_users.items()))
                else:
                    test_data_x = []
                    test_data_y = []
                    for i in range(len(train_clients)):
                        if i % 2 == 1:
                            cur_x = test_data_temp[train_clients[i]]['x']
                            cur_y = test_data_temp[train_clients[i]]['y']
                            for j in range(len(cur_x)):
                                test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                                test_data_y.append(cur_y[j])
                    self.data = np.array(test_data_x)
                    self.label = np.array(test_data_y)
            #processes partition half 2 of data
            if client_num == 2:
                if self.train:
                    self.dic_users = {}
                    train_data_x = []
                    train_data_y = []
                    for i in range(len(train_clients)):
                        if i % 2 == 0:
                            # if i == 100:
                            #     break
                            self.dic_users[i] = set()
                            l = len(train_data_x)
                            cur_x = train_data_temp[train_clients[i]]['x']
                            cur_y = train_data_temp[train_clients[i]]['y']
                            for j in range(len(cur_x)):
                                self.dic_users[i].add(j + l)
                                train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                                train_data_y.append(cur_y[j])
                    self.data = np.array(train_data_x)
                    self.label = np.array(train_data_y)
                    self.dic_users = np.array(list(self.dic_users.items()))
                else:
                    test_data_x = []
                    test_data_y = []
                    for i in range(len(train_clients)):
                        if i % 2 == 0:
                            cur_x = test_data_temp[train_clients[i]]['x']
                            cur_y = test_data_temp[train_clients[i]]['y']
                            for j in range(len(cur_x)):
                                test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                                test_data_y.append(cur_y[j])
                    self.data = np.array(test_data_x)
                    self.label = np.array(test_data_y)

            os.makedirs(CACHE_DIR, exist_ok=True)        
            np.save(os.path.join(CACHE_DIR, f"data_{client_num}_{train}.npy"), self.data)
            np.save(os.path.join(CACHE_DIR, f"label_{client_num}_{train}.npy"), self.label)
            if train: np.save(os.path.join(CACHE_DIR, f"user_label_{client_num}_{train}.npy"), self.dic_users)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array(img)
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    print(len(files))
    for i, f in enumerate(files):
        print(i)
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    print("read_dir")
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    assert train_clients == test_clients
    assert train_groups == test_groups
    print("read_data")
    return train_clients, train_groups, train_data, test_data


def only_numbers(ds):
        '''
        Takes in an existing dataset and filters it to return dataset of just digits 0-9
        '''
        digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        #image, label = ds.data, ds.label
        filtered_dataset = [(image, label) for image, label in ds
                            if str(label) in digit_labels]

        images = torch.stack([image for image, _ in filtered_dataset])
        labels = torch.tensor([label for _, label in filtered_dataset])

        filtered_dataset = TensorDataset(images, labels)
        return filtered_dataset

def overlap_only_numbers(ds):
        '''
        For overlapping data experiments only
        Takes in existing datset and filters to return dataset of just digits 0-9
        '''
        digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        #filtered_dataset = [(image, label) for image, label in ds
                            #if str(label) in digit_labels]
        filtered_dataset = []
        for image, label in ds: 
            if str(label[1]) in digit_labels: filtered_dataset.append((image, label))
        images = torch.stack([image for image, _ in filtered_dataset])
        labels = torch.tensor(np.array([label for _, label in filtered_dataset]))

        filtered_dataset = TensorDataset(images, labels)
        return filtered_dataset