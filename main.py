import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from nilearn.image import load_img
from sklearn.linear_model import LinearRegression


class VGGNet(nn.Module):
    """Convolutional blocks followed by a linear layer"""
    def __init__(self):
        super(VGGNet, self).__init__()
        blocks = []
        blocks.append(VGGBlock(1, 16, 3))
        blocks.append(VGGBlock(16, 32, 3))
        blocks.append(VGGBlock(32, 64, 3))
        blocks.append(VGGBlock(64, 128, 3))
        blocks.append(VGGBlock(128, 256, 3))
        self.blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16384, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class VGGBlock(nn.Module):
    """3D convolution, activation and pooling"""
    def __init__(self, n_in, n_out, k_size):
        super(VGGBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv3d(n_in, n_out, k_size, stride=1),
                nn.ReLU(),
                nn.Conv3d(n_out, n_out, k_size, stride=1),
                # nn.BatchNorm3d(n_out),
                nn.ReLU(),
                nn.MaxPool3d(2),
                )

    def forward(self, x):
        x = self.block(x)
        return x


class AllConvNet(nn.Module):
    """Convolutional blocks followed by a linear layer"""
    def __init__(self):
        super(AllConvNet, self).__init__()
        blocks = []
        blocks.append(AllConvBlock(1, 16, 3))
        blocks.append(AllConvBlock(16, 32, 3))
        blocks.append(AllConvBlock(32, 64, 3))
        blocks.append(AllConvBlock(64, 128, 3))
        blocks.append(AllConvBlock(128, 256, 3))
        self.blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32000, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class AllConvBlock(nn.Module):
    """3D strided convolution and activation"""
    def __init__(self, n_in, n_out, k_size):
        super(AllConvBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv3d(n_in, n_out, k_size, stride=1),
                nn.ReLU(),
                nn.Conv3d(n_out, n_out, k_size, stride=2),
                # nn.BatchNorm3d(n_out),
                nn.ReLU(),
                )

    def forward(self, x):
        x = self.block(x)
        return x


class BasicBlock(nn.Module):
    """Convolutional block in the ResNet architecture"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """The ResNet architecture"""
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        base = 16
        self.in_planes = base

        self.conv1 = nn.Conv3d(1, base, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base)
        self.layer1 = self._make_layer(block, base, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(65536*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class BasicBlock_NoBN(nn.Module):
    """Convolutional block in ResNet, without batch normalization"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_NoBN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_NoBN(nn.Module):
    """ResNet without batch normalization"""
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet_NoBN, self).__init__()
        base = 16
        self.in_planes = base

        self.conv1 = nn.Conv3d(1, base, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, base, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(65536*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10_NoBN():
    return ResNet_NoBN(BasicBlock_NoBN, [1, 1, 1, 1])


def ResNet18_NoBN():
    return ResNet_NoBN(BasicBlock_NoBN, [2, 2, 2, 2])


def clean_oasis(data_dir, new_dir):
    """Copies all norm.mgz files in OASIS-3 into a new directory"""
    names = os.listdir(data_dir)
    names.sort()
    for name in tqdm(names):
        short_name = name + ".mgz"
        old_path = os.path.join(os.path.join(os.path.join(data_dir, name), "mri"), "norm.mgz")
        new_path = os.path.join(new_dir, short_name)
        subprocess.run(["cp", old_path, new_path])

def clean_bacs(data_dir, new_dir):
    """Copies all norm.mgz files in BACS into a new directory"""
    matches = []
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith("norm.mgz"):
                matches.append(os.path.join(root, filename))
    for sample in tqdm(matches):
        new_path = os.path.join(new_dir, sample[50:57] + sample[61:72] + ".mgz")
        subprocess.run(["cp", sample, new_path])


class ScanDataSet(Dataset):
    """Dataset that loads data based on (file_path, age) pairs"""
    def __init__(self, mapping, device):
        self.mapping = mapping
        self.device = device

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        data_name, age = self.mapping[idx]
        data = load_img(data_name).get_fdata()
        x = torch.unsqueeze(torch.FloatTensor(data), 0).to(self.device)
        y = torch.FloatTensor([age]).to(self.device)
        return x, y


def mapping_bacs(data_dir, mmse_excel, mapping_excel, mmse_th=25, healthy=True):
    """Creates (file_path, age) pairs using the BACS spreadsheets"""
    paths = os.listdir(data_dir)
    paths.sort()
    df1 = pd.read_excel(mapping_excel)
    bio_bac = dict(zip(df1.iloc[:, 1], df1.iloc[:, 0]))
    df2 = pd.read_excel(mmse_excel)
    df2 = df2[df2['BAC ID'].notna()]
    if healthy:
        df2 = df2[df2['MMSE'] >= mmse_th]
    else:
        df2 = df2[df2['MMSE'] < mmse_th]
    bac_birth = dict(zip(df2.iloc[:, 0], df2.iloc[:, 1]))
    id_dict = {}
    for path in paths:
        if bio_bac[path[:7]] in bac_birth:
            scan_ts = pd.Timestamp(path[8:18])
            birth_ts = bac_birth[bio_bac[path[:7]]]
            id_dict[os.path.join(data_dir, path)] = (scan_ts - birth_ts).days / 365
    return list(id_dict.items())


def mapping_oasis(data_dir, mmse_csv, mmse_th=25, healthy=True):
    """Creates (file_path, age) pairs using the OASIS-3 csv"""
    df2 = pd.read_csv(mmse_csv)
    df2 = df2[df2['MMSE'].notna()]
    if healthy:
        df2 = df2[df2['MMSE'] >= mmse_th]
    else:
        df2 = df2[df2['MMSE'] < mmse_th]
    df2 = df2[df2['days_to_visit'] == 0]
    oasisid_age_at_entry = dict(zip(df2.iloc[:, 0], df2.iloc[:, 3]))
    paths = os.listdir(data_dir)
    paths.sort()
    id_dict = {}
    for path in paths:
        if path[:8] in oasisid_age_at_entry:
            days = path[13:17]
            age_at_entry = oasisid_age_at_entry[path[:8]]
            id_dict[os.path.join(data_dir, path)] = float(age_at_entry) + int(days) / 365
    return list(id_dict.items())


def weighted_mse_loss(input, target, weight):
    """Weighted MSE loss for subjects with multiple scans"""
    return torch.sum(weight * (input - target) ** 2)


def weighted_l1_loss(input, target, weight):
    """Weighted MAE loss for subjects with multiple scans"""
    return torch.sum(weight * torch.abs(input - target))


def train(net, loss_func, optim, num_epochs, train_loader, val_loader, checkpoint_name):
    """Main training loop"""
    train_losses, val_losses = [], []
    min_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, val_loss = 0, 0
        # Training
        net.train()
        for x, y in tqdm(train_loader):
            x = net(x)
            if loss_func == "l1":
                # All scans are currently equally weighted. To be implemented
                loss = weighted_l1_loss(x, y, 1)
            else:
                loss = weighted_mse_loss(x, y, 1)
            train_loss += loss.detach().cpu()
            optim.zero_grad()
            loss.backward()
            optim.step()
        # Validation
        net.eval()
        with torch.no_grad():
            absolute_errors = []
            for x, y in tqdm(val_loader):
                x = net(x)
                if loss_func == "l1":
                    loss = weighted_l1_loss(x, y, 1)
                else:
                    loss = weighted_mse_loss(x, y, 1)
                absolute_error = (x - y).abs().cpu().numpy()
                val_loss += loss.detach().cpu()
                absolute_errors.extend(absolute_error.flatten())
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        mae = np.mean(absolute_errors)
        print("Epoch {}: {:.3f} train loss, {:.3f} val loss, {:.3f} val MAE".format(epoch, train_loss, val_loss, mae))
        # Save the model if we achieve the lowest validation loss so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            name = checkpoint_name + ".pt"
            print("Saving model to", name)
            torch.save(net.state_dict(), name)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses


def plot_losses(train_losses, val_losses, checkpoint_name):
    """Plot training and validation losses per epoch"""
    plt.rcParams['savefig.facecolor']='white'
    plt.plot(range(len(train_losses)), train_losses, label='Train loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.savefig(checkpoint_name + "_loss.png", dpi=200)


def plot_scatter(ad_preds, ad_ys, hc_preds, hc_ys, checkpoint_name):
    """Creates scatterplot of predicted vs actual age for HC and AD"""
    plt.figure(figsize=(8, 6))
    plt.scatter(hc_ys, hc_preds, s=10, color='turquoise', label="HC")
    plt.scatter(ad_ys, ad_preds, s=10, color='lightcoral', label="AD")
    plt.axis("square")
    plt.xlim(40, 100)
    plt.ylim(40, 100)
    plt.xlabel("Actual age")
    plt.ylabel("Predicted age")
    plt.title("Predicted vs actual age for HC and AD")
    plt.plot([40, 100], [40, 100], linestyle = 'dotted', color='purple', label="y = x")
    model = LinearRegression().fit(np.array(hc_ys).reshape(-1, 1), np.array(hc_preds).reshape(-1, 1))
    a = (model.coef_ * 100 + model.intercept_)[0][0]
    b = (model.coef_ * 40 + model.intercept_)[0][0]
    plt.plot([100, 40], [a, b], color='teal', label="HC Best fit")
    model = LinearRegression().fit(np.array(ad_ys).reshape(-1, 1), np.array(ad_preds).reshape(-1, 1))
    a = (model.coef_ * 100 + model.intercept_)[0][0]
    b = (model.coef_ * 40 + model.intercept_)[0][0]
    plt.plot([100, 40], [a, b], color='brown', label="AD Best fit")
    plt.legend()
    name = checkpoint_name + "_scatter.png"
    plt.savefig(name, dpi=200, bbox_inches="tight")
    print("Scatterplot saved to", name)


def test(net, test_ad_loader, test_hc_loader, checkpoint_name):
    """Evaluates trained model on a test set"""
    net.eval()
    name = checkpoint_name + ".pt"
    net.load_state_dict(torch.load(name))
    with torch.no_grad():
        ad_preds = []
        ad_ys = []
        for x, y in tqdm(test_ad_loader):
            x = net(x)
            ad_preds.append(x.cpu().numpy().flatten()[0])
            ad_ys.append(y.cpu().numpy().flatten()[0])
        hc_preds = []
        hc_ys = []
        for x, y in tqdm(test_hc_loader):
            x = net(x)
            hc_preds.append(x.cpu().numpy().flatten()[0])
            hc_ys.append(y.cpu().numpy().flatten()[0])
    ad_preds = np.array(ad_preds)
    ad_ys = np.array(ad_ys)
    hc_preds = np.array(hc_preds)
    hc_ys = np.array(hc_ys)
    ad_me = np.mean(ad_preds - ad_ys)
    ad_mae = np.mean(np.abs(ad_preds - ad_ys))
    hc_me = np.mean(hc_preds - hc_ys)
    hc_mae = np.mean(np.abs(hc_preds - hc_ys))
    print("AD ME: {} AD MAE: {} HC ME: {} HC MAE: {}".format(ad_me, ad_mae, hc_me, hc_mae))
    plot_scatter(ad_preds, ad_ys, hc_preds, hc_ys, checkpoint_name)

torch.manual_seed(42)
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'
# Should print "Using device cuda"
print("Using device", device)

# Specify data path. Directory containing data should be organized as follows:
# data
# ---bacs
# ------B01-201_2000-01-01.mgz
# ------...
# ---first_session
# ------OAS30001_MR_d0001.mgz
# ------...
# ---last_session
# ------OAS30001_MR_d0002.mgz
# ------...
# ---BACS MMSE.xlsx
# ---BACS_Biomed_mapping.xlsx
# ---OASIS3_UDSb4_cdr.csv
# clean_bacs and clean_oasis may be useful for organizational purposes

data_dir = "C:/Users/naton/Desktop/ds/data"
bacs_dir = os.path.join(data_dir, "bacs")
bacs_mmse_excel = os.path.join(data_dir, "BACS MMSE.xlsx")
bacs_mapping_excel = os.path.join(data_dir, "BACS_Biomed_mapping.xlsx")
oasis_first_dir = os.path.join(data_dir, "first_session")
oasis_last_dir = os.path.join(data_dir, "last_session")
oasis_mmse_csv = os.path.join(data_dir, "OASIS3_UDSb4_cdr.csv")

# Create HC and AD mapping for the two datasets
hc_bacs = mapping_bacs(bacs_dir, bacs_mmse_excel, bacs_mapping_excel)
ad_bacs = mapping_bacs(bacs_dir, bacs_mmse_excel, bacs_mapping_excel, healthy=False)

hc_oasis_first = mapping_oasis(oasis_first_dir, oasis_mmse_csv)
hc_oasis_last = mapping_oasis(oasis_last_dir, oasis_mmse_csv)
ad_oasis_first = mapping_oasis(oasis_first_dir, oasis_mmse_csv, healthy=False)
ad_oasis_last = mapping_oasis(oasis_last_dir, oasis_mmse_csv, healthy=False)

hc_oasis = hc_oasis_first + hc_oasis_last
ad_oasis = ad_oasis_first + ad_oasis_last

# 80/10/10 train/validation/test split for each dataset
hc_bacs_num_train = int(len(hc_bacs) * 0.8)
hc_bacs_num_val = int(len(hc_bacs) * 0.1)
hc_bacs_num_test = len(hc_bacs) - hc_bacs_num_train - hc_bacs_num_val
hc_oasis_num_train = int(len(hc_oasis) * 0.8)
hc_oasis_num_val = int(len(hc_oasis) * 0.1)
hc_oasis_num_test = len(hc_oasis) - hc_oasis_num_train - hc_oasis_num_val

hc_bacs_train, hc_bacs_val, hc_bacs_test = hc_bacs[:hc_bacs_num_train], hc_bacs[hc_bacs_num_train:hc_bacs_num_train+hc_bacs_num_val], hc_bacs[hc_bacs_num_train+hc_bacs_num_val:]
hc_oasis_train, hc_oasis_val, hc_oasis_test = hc_oasis[:hc_oasis_num_train], hc_oasis[hc_oasis_num_train:hc_oasis_num_train+hc_oasis_num_val], hc_oasis[hc_oasis_num_train+hc_oasis_num_val:]

print(len(hc_bacs_train), len(hc_bacs_val), len(hc_bacs_test))
print(len(hc_oasis_train), len(hc_oasis_val), len(hc_oasis_test))

# Combine the two datasets
train_mapping = hc_bacs_train + hc_oasis_train
val_mapping = hc_bacs_val + hc_oasis_val
test_hc_mapping = hc_bacs_test + hc_oasis_test
test_ad_mapping = ad_bacs + ad_oasis

# Define hyperparameters and checkpoint name
net = AllConvNet().to(device)
checkpoint_name = "allconvnet"
batch_size = 2
num_epochs = 20
loss_func = "mse"
optim = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.01)

# Define the DataSet and DataLoader
train_set = ScanDataSet(train_mapping, device=device)
val_set = ScanDataSet(val_mapping, device=device)
test_hc_set = ScanDataSet(test_hc_mapping, device=device)
test_ad_set = ScanDataSet(test_ad_mapping, device=device)
print("Train: {} Val: {} HC Test: {} AD Test: {}".format(len(train_set), len(val_set), len(test_hc_set), len(test_ad_set)))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_hc_loader = DataLoader(test_hc_set, batch_size=1, shuffle=False)
test_ad_loader = DataLoader(test_ad_set, batch_size=1, shuffle=False)

# Training. Comment out the following two lines for inference
train_losses, val_losses = train(net, loss_func, optim, num_epochs, train_loader, val_loader, checkpoint_name)
plot_losses(train_losses, val_losses, checkpoint_name)

# Evaluate on HC and AD test set
test(net, test_ad_loader, test_hc_loader, checkpoint_name)
