import torch
import torch.nn as nn
import einops
import numpy as np
import copy

class MnistConfig:
    """A configuration class for MNIST models"""
    def __init__(self, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = 784
        self.n_layer = 1
        self.d_hidden = 300
        self.num_classes = 10
        self.activation = 'bilinear'
        self.embed = True
        self.random_seed = 0
        self.rms_norm = False
        self.bias = False
        self.noise_sparse = 0
        self.noise_dense = 0
        self.layer_noise = 0
        self.logit_bias = True
    
        # training params
        self.num_epochs = 10
        self.lr = 0.001
        self.weight_decay = 0
        self.lr_decay = 0.5
        self.lr_decay_step = 2

        self.__dict__.update(kwargs)


class Relu(nn.Module):
    def __init__(self, cfg):
        super(Relu, self).__init__()
        self.cfg = cfg

        self.linear = nn.Linear(cfg.d_hidden, cfg.d_hidden)
        self.act = nn.ReLU()
        self.rms_norm = RmsNorm(cfg)
    
    def forward(self, x):
        out = self.linear(x)
        self.out_prenorm = self.act(out)
        self.out = self.rms_norm(self.out_prenorm)
        return self.out

class Bilinear(nn.Module):
    def __init__(self, cfg):
        super(Bilinear, self).__init__()
        self.cfg = cfg

        self.linear1 = nn.Linear(cfg.d_hidden, cfg.d_hidden, bias = cfg.bias)
        self.linear2 = nn.Linear(cfg.d_hidden, cfg.d_hidden, bias = cfg.bias)
        self.rms_norm = RmsNorm(cfg)
        
    def forward(self, x):
        self.input = x
        out1 = self.linear1(self.input)
        out2 = self.linear2(self.input)
        self.out_prenorm = out1 * out2
        self.out = self.rms_norm(self.out_prenorm)
        return self.out

class RmsNorm(nn.Module):
    def __init__(self, cfg):
        super(RmsNorm, self).__init__()
        self.cfg = cfg
      
    def forward(self, x):
        if self.cfg.rms_norm:
            self.rms_scale = torch.sqrt((x**2).sum(dim=-1, keepdim=True))
            self.out = x/self.rms_scale
        else:
            self.out = x
        return self.out

class MnistModel(nn.Module):
    # TODO: make validation and training work for different image inputs
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.random_seed is not None:
            torch.manual_seed(cfg.random_seed)

        self.input_norm = RmsNorm(cfg)
        if cfg.embed:
            self.linear_in = nn.Linear(cfg.input_size, cfg.d_hidden, bias=False).to(cfg.device)

        layers = []
        for idx in range(cfg.n_layers):
            if cfg.activation == 'bilinear':
                mlp = Bilinear(cfg)
            elif cfg.activation == 'relu':
                mlp = Relu(cfg)
            layers.append(mlp.to(cfg.device))

        self.layers = nn.ModuleList(layers)
        self.linear_out = nn.Linear(cfg.d_hidden, cfg.num_classes, bias=cfg.logit_bias).to(cfg.device)

    def forward(self, x, inference=False):
        self.input_prenorm = x
        x = self.input_norm(x)
        self.input = x

        if self.cfg.embed:
            x = self.linear_in(x)
            
        for layer in self.layers:
            x = layer(x)
            if (not inference):
                x = x+ self.cfg.layer_noise * x.std() * torch.randn_like(x)

        self.out = self.linear_out(x)
        return self.out

    def criterion(self, output, labels):
        return nn.CrossEntropyLoss()(output, labels)

    def validation_accuracy(self, test_loader, print_acc=True):
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            loss_sum = 0
            count = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                outputs = self.forward(images, inference=True)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                loss_sum += self.criterion(outputs, labels).item()
                count += 1

            acc = 100.0 * n_correct / n_samples
            loss = loss_sum / count
            if print_acc:
              print(f'Accuracy on validation set: {acc} %')
        return acc, loss

    def train(self, train_loader, test_loader, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)

        num_epochs = self.cfg.num_epochs
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            _ = self.validation_accuracy(test_loader)
            for i, (images, labels) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(self.cfg.device)
                labels = labels.to(self.cfg.device)

                # input noise
                noise_mask = torch.bernoulli(self.cfg.noise_sparse * torch.ones_like(images)).bool()
                images[noise_mask] = 1 - images[noise_mask]
                images += self.cfg.noise_dense * torch.randn_like(images)
                

                # Forward pass
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            if (scheduler is not None):
                scheduler.step()
                print(f'learning rate = {scheduler.get_last_lr()[0]}')
        _ = self.validation_accuracy(test_loader)


class BilinearTopK(torch.nn.Module):
    def __init__(self, B, requires_grad = False, norm = True, bias = False):
        super().__init__()
        self.B = torch.nn.Parameter(B, requires_grad=requires_grad)
        self.out = None
        self.bias = bias

        self.norm = norm
        if norm:
          self.rms_norm = RmsNorm()

    def forward(self, x):
        device = self.B.device
        if x.device != device:
            x = x.to(device)

        if self.bias:
            ones =  torch.ones(x.size(0), 1).to(device)
            self.input = torch.cat((x, ones), dim=-1)
        else:
            self.input = x

        self.out_prenorm = einops.einsum(self.input, self.B, self.input, 'b d0, s d0 d1, b d1 -> b s')
        if self.norm:
            self.out = self.rms_norm(self.out_prenorm)
        else:
            self.out = self.out_prenorm
        return self.out

class BilinearModelTopK(torch.nn.Module):
    # TODO: make validation_accuracy work for non-image datasets
    def __init__(self, model, svds, sing_val_type, input_idxs = None):
        super().__init__()
        self.svds = svds
        self.sing_val_type = sing_val_type

        self.cfg = copy.deepcopy(model.cfg)
        self.device = svds[0].V.device
        self.W_out = model.linear_out.weight.detach().to(self.device)
        self.bias_out = model.linear_out.bias.detach().to(self.device)
        self.final_norm = model.cfg.rms_norm
        
        self.input_idxs = input_idxs

        if self.cfg.rms_norm:
            self.rms_norm = RmsNorm()

    def set_parameters(self, topKs):
        B_tensors, R_tensors = self.get_tensors(topKs)
        W_out = self.W_out @ R_tensors[-1]
        bias_out = self.bias_out

        layers = []
        for layer_idx, B in enumerate(B_tensors):
            if layer_idx < len(B_tensors) - 1:
                layers.append(BilinearTopK(B, norm = self.cfg.rms_norm, bias=self.cfg.bias))
            else:
                layers.append(BilinearTopK(B, norm = self.final_norm, bias=self.cfg.bias))
        self.layers = nn.Sequential(*layers)

        self.linear_out = torch.nn.Linear(*W_out.T.shape)
        with torch.no_grad():
            self.linear_out.weight = torch.nn.Parameter(W_out)
            self.linear_out.bias = torch.nn.Parameter(bias_out)

    def get_tensors(self, topKs):
        device = self.svds[0].V.device

        B_tensors = []
        R_tensors = []

        for layer_idx, svd in enumerate(self.svds):
            topK = topKs[layer_idx]
            svd_components = self.svds[layer_idx].V.shape[1]
            dim = np.sqrt(svd.V.shape[0]).astype(int)

            if self.input_idxs is None:
                Q_dim = np.sqrt(svd.V.shape[0]).astype(int) if layer_idx == 0 else topKs[layer_idx-1]
                Q_idxs = torch.arange(Q_dim)
                if self.cfg.bias:
                    Q_idxs = torch.cat([Q_idxs, torch.tensor([svd_components])])
                
                if self.sing_val_type == 'with R':
                    Q = svd.V.reshape(dim,dim,svd_components)[:,:,:topK][Q_idxs, :, :][:,Q_idxs,:]
                    R = svd.U[:,:topK] @ torch.diag(svd.S[:topK])
                elif self.sing_val_type == 'with Q':
                    Q = svd.V.reshape(dim,dim,svd_components)[:,:,:topK][Q_idxs, :, :][:,Q_idxs,:]
                    Q = Q @ torch.diag(svd.S[:topK])
                    R = svd.U[:,:topK]
                
                B = einops.rearrange(Q, "i j svd -> svd i j")
                B_tensors.append(B)
                R_tensors.append(R)

            else:
                if layer_idx == 0:
                    idxs = input_idxs.clone().to(device)
                    Q_idxs = input_idxs.clone().to(device)
                else:
                    Q_idxs = torch.arange(topK_list[layer_idx-1]).to(device)
                    if bias:
                        idxs = torch.arange(svd_components+1).to(device)
                        Q_idxs = torch.cat([Q_idxs, torch.tensor([svd_components])]).to(device)
                    else:
                        idxs = torch.arange(svd_components).to(device)

                topK = topK_list[layer_idx]
                B = torch.zeros((topK, len(Q_idxs), len(Q_idxs))).to(device)

                idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)
                mask0 = torch.isin(idx_pairs[:,0], Q_idxs)
                mask1 = torch.isin(idx_pairs[:,1], Q_idxs)
                mask = torch.logical_and(mask0, mask1)
                idx_pairs_reduced = idx_pairs[mask]
                if sing_val_type == 'with R':
                    Q_reduced = svd.V[mask, :topK]
                    R = svd.U[:,:topK] @ torch.diag(svd.S[:topK])
                elif sing_val_type == 'with Q':
                    Q_reduced = svd.V[mask, :topK] @ torch.diag(svd.S[:topK])
                    R = svd.U[:,:topK]

                idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(range(len(Q_idxs)),2))).to(device)
                B[:, idx_pairs[:,0],idx_pairs[:,1]] = Q_reduced.T
                B[:, idx_pairs[:,1],idx_pairs[:,0]] = Q_reduced.T

                B_tensors.append(B)
                R_tensors.append(R)
        return B_tensors, R_tensors

    def forward(self, x):
        if self.cfg.rms_norm:
            x = self.rms_norm(x)
        x = self.get_input(x)
        self.input = x
        for layer in self.layers:
            x = layer(x)
        self.out = self.linear_out(x)
        # no activation and no softmax at the end
        return self.out

    def get_input(self,x):
        if self.input_idxs is None:
            return x

        if self.cfg.bias:
            input_idxs = self.input_idxs[:-1]
            return x[:,input_idxs]
        else:
            return x[:,input_idxs]

    def validation_accuracy(self, test_loader, print_acc=True):
        # In test phase, we don't need to compute gradients (for memory efficiency)
        device = self.layers[0].B.device
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            if print_acc:
              print(f'Accuracy on validation set: {acc} %')
        return acc

class MaxActivationConfig():
    def __init__(self):
        self.warmup_epochs = 2
        self.epochs = 5
        self.steps = 1000
        self.lr = 0.1
        self.lr_decay = .5
        self.lr_decay_step = 1
        self.print_log = False

class MaxActivationModel(torch.nn.Module):
    def __init__(self, cfg, Q):
        super().__init__()
        self.cfg = cfg
        device = Q.device
        self.x = torch.nn.Parameter(torch.rand(Q.shape[0], device=device), requires_grad=True)
        with torch.no_grad():
            self.Q = torch.nn.Parameter(Q, requires_grad=False).to(device)

    def forward(self):
        act = self.get_activation()
        return act.T @ self.Q @ act

    def get_activation(self):
        return torch.sigmoid(self.x)

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        linearLR = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, 
            end_factor=1, total_iters = self.cfg.warmup_epochs)
        stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linearLR, stepLR], milestones=[self.cfg.warmup_epochs])

        epochs = self.cfg.warmup_epochs + self.cfg.epochs
        steps = self.cfg.steps
        for epoch in range(epochs):
            for step in range(steps):

                # Forward pass
                output = self.forward()
                loss = -output

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            if self.cfg.print_log:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Learning rate = {learning_rate}')
