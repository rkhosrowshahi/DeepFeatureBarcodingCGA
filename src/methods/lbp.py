import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LBP(nn.Module):
    def __init__(self, num_features, n_epochs=100, batch_size=128, lr=0.001, device='cuda', *args, **kwargs):
        super(LBP, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_features = num_features
        self.threshold = nn.Parameter(torch.randn(num_features))
        self.temperature = 0.1  # Sharper sigmoid for more binary-like outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        # Move the entire module to device after all parameters are registered
        self.to(self.device)

    def forward(self, x):
        # Ensure x is on the same device as parameters
        if x.device != self.threshold.device:
            x = x.to(self.threshold.device)
        # Use sigmoid for smooth approximation of step function
        return torch.sigmoid((x - self.threshold) / self.temperature)
    
    def compute_similarity_loss(self, outputs, labels):
        # Compute pairwise similarities between binary codes
        dot_product = torch.matmul(outputs, outputs.t())
        # Normalize to [0, 1]
        similarities = (dot_product / outputs.size(1))
        
        # Compute label similarities (1 if same label, 0 if different)
        label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Binary cross entropy between similarities
        loss = -(label_sim * torch.log(similarities + 1e-8) + 
                (1 - label_sim) * torch.log(1 - similarities + 1e-8))
        return loss.mean()
    
    def fit(self, train_features, train_labels):
        train_features = torch.from_numpy(train_features).float().to(self.device)
        train_labels = torch.from_numpy(train_labels).long().to(self.device)
        
        # Create optimizer after ensuring parameters are properly registered
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        batch_size = min(self.batch_size, len(train_features))
        n_batches = len(train_features) // batch_size if len(train_features) > 0 else 1

        for epoch in range(self.n_epochs):
            total_loss = 0
            # Shuffle data
            perm = torch.randperm(len(train_features), device=self.device)
            train_features = train_features[perm]
            train_labels = train_labels[perm]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(train_features))
                
                batch_features = train_features[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self(batch_features)
                
                # Compute similarity preservation loss
                loss = self.compute_similarity_loss(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

        return self
    
    def transform(self, features):
        test_features = torch.from_numpy(features).float().to(self.device)
        with torch.no_grad():
            outputs = self(test_features)
            return (outputs > 0.5).cpu().numpy().astype(np.uint8)
        
