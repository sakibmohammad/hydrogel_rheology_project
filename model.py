import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim = 7, output_dim = 2, hidden_layers = 3, neurons = 512, dropout_rate= 0):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, neurons))
        for i in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neurons, neurons))
        self.output_layer = nn.Linear(neurons, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, 10)
        self.fc22 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(12, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 7)

    def encode(self, x, condition):
        combined = torch.cat([x, condition], 1)
        h1 = F.relu(self.fc1(combined))
        h2 = F.relu(self.fc2(self.dropout(h1)))
        h3 = F.relu(self.fc3(self.dropout(h2)))
        return self.fc21(h3), self.fc22(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, condition):
        combined = torch.cat([z, condition], 1)
        h4 = F.relu(self.fc4(combined))
        h5 = F.relu(self.fc5(self.dropout(h4)))
        h6 = F.relu(self.fc6(self.dropout(h5)))
        return torch.sigmoid(self.fc7(self.dropout(h6)))

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

def load_mlp(model_path, device = "cpu"):
  model = MLP().to(device)
  model.load_state_dict(torch.load(f="model_regression.pth", map_location=device))
  return model

def load_CVAE(model_path, device = "cpu"):
  model = CVAE().to(device)
  model.load_state_dict(torch.load(f="model_cvae.pth", map_location=device))
  return model
