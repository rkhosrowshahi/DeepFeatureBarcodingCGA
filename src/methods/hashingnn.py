import torch
import torch.nn as nn
import torch.optim as optim


class HashingNN(nn.Module):
    def __init__(self, num_features, num_bits, loss_fn='pairwise', n_epochs=100, device='cuda', *args, **kwargs):
        super(HashingNN, self).__init__()
        # self.fc = nn.Linear(input_dim, hash_bits).requires_grad_(True) # Learn binary encoding
        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_bits),
            nn.Tanh()   # tanh bounds outputs between -1 and 1 (common choice for hashing)
        )
        self.num_features = num_features
        self.num_bits = num_bits  
        self.tanh = nn.Tanh()  # Enforce binary-like activation
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Move the entire model to the specified device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as model
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        x = self.fc(x)
        # x = self.tanh(x)
        return x  # Convert to binary hash (±1)
    
    def pairwise_loss(self, hash_codes, labels):
        similarity_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(self.device)
        hash_sim = torch.matmul(hash_codes, hash_codes.T) / self.num_bits
        return torch.mean((similarity_matrix - hash_sim) ** 2)
    
    def contrastive_loss(self, hash_codes, labels, margin=1):
        batch_size = hash_codes.shape[0]
        similarity_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(self.device)
        hash_sim = torch.matmul(hash_codes, hash_codes.T) / self.num_bits
        pos_pairs = similarity_matrix * (1 - hash_sim)
        neg_pairs = (1 - similarity_matrix) * torch.clamp(hash_sim - margin, min=0)
        return torch.mean(pos_pairs + neg_pairs)
    
    def dsh_loss(self, hash_codes, labels):
        margin = 0.2
        alpha = 0.01
        batch_size = hash_codes.size(0)
        # Compute pairwise distances
        dist_matrix = torch.cdist(hash_codes, hash_codes, p=2)
        # Create label similarity matrix
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_matrix = label_matrix.float()
        # Contrastive loss
        positive_loss = label_matrix * dist_matrix.pow(2)
        negative_loss = (1 - label_matrix) * torch.clamp(margin - dist_matrix, min=0).pow(2)
        pairwise_loss = positive_loss + negative_loss
        pairwise_loss = pairwise_loss.sum() / (batch_size * (batch_size - 1))
        # Regularization to encourage binary-like outputs
        reg_loss = alpha * (1 - hash_codes.abs()).abs().mean()
        return pairwise_loss + reg_loss
    
    def dhn_loss(self, hash_codes, labels):
        alpha = 0.1
        batch_size = hash_codes.size(0)
        # Compute similarity matrix
        similarity_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # Compute inner product
        inner_product = torch.matmul(hash_codes, hash_codes.t()) / hash_codes.size(1)
        # Apply sigmoid to get probabilities
        prob = torch.sigmoid(inner_product)
        # Cross-entropy loss
        ce_loss = - (similarity_matrix * torch.log(prob + 1e-10) + (1 - similarity_matrix) * torch.log(1 - prob + 1e-10))
        ce_loss = ce_loss.sum() / (batch_size * (batch_size - 1))
        # Quantization loss
        quant_loss = (hash_codes.abs() - 1).pow(2).mean()
        return ce_loss + alpha * quant_loss

    def quantization_loss(self, hash_codes, labels):
        return torch.mean((hash_codes - torch.sign(hash_codes)) ** 2)
    
    def triplet_loss(self, hash_codes, labels, margin=1):
        batch_size = hash_codes.shape[0]
        loss = 0
        for i in range(batch_size):
            anchor = hash_codes[i]
            pos_idx = (labels == labels[i]).nonzero(as_tuple=True)[0]
            neg_idx = (labels != labels[i]).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 1 and len(neg_idx) > 0:
                pos = hash_codes[pos_idx[1]]
                neg = hash_codes[neg_idx[0]]
                loss += torch.clamp(torch.norm(anchor - pos) - torch.norm(anchor - neg) + margin, min=0)
        return loss / batch_size
    
    def fit(self, train_features, train_labels):
        # Convert features to tensor
        train_features = torch.from_numpy(train_features).float().to(self.device)
        train_labels = torch.from_numpy(train_labels).long().to(self.device)

        # Define loss function and optimizer
        # criterion = nn.CrossEntropyLoss()
        if self.loss_fn == 'pairwise':
            criterion = self.pairwise_loss
        elif self.loss_fn == 'contrastive':
            criterion = self.contrastive_loss
        elif self.loss_fn == 'quantization':
            criterion = self.quantization_loss
        elif self.loss_fn == 'triplet':
            criterion = self.triplet_loss
        elif self.loss_fn == 'dsh':
            criterion = self.dsh_loss
        elif self.loss_fn == 'dhn':
            criterion = self.dhn_loss

        optimizer = optim.SGD(self.parameters(), lr=0.01)

        # Training loop
        self.train()
        for epoch in range(self.n_epochs):  # 10 epochs as per the paper
            optimizer.zero_grad()
            outputs = self(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 or epoch == 9:
                avg_loss = loss.item()
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        return self
    
    def transform(self, features):
        with torch.no_grad():
            features = torch.from_numpy(features).float().to(self.device)
            return self(features).sign().cpu().numpy()