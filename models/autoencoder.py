import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use sigmoid for better reconstruction
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderModel:
    def __init__(self, input_dim, hidden_dim=64, encoding_dim=32, device='cpu', random_state=42):
        # Set random seeds for reproducible results
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.autoencoder = Autoencoder(input_dim, hidden_dim, encoding_dim).to(self.device)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
    def fit(self, X, epochs=50, batch_size=32, verbose=1):
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalize data for better training
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.autoencoder(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:  # Early stopping patience
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    def predict(self, X, threshold=None):
        self.autoencoder.eval()
        with torch.no_grad():
            # Convert to numpy array if it's a DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Normalize data using training statistics
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            recon = self.autoencoder(X_tensor)
            recon = recon.cpu().numpy()
            
            # Denormalize reconstruction
            recon_denorm = recon * (X.std(axis=0) + 1e-8) + X.mean(axis=0)
            
        errors = np.mean(np.square(X - recon_denorm), axis=1)
            
        if threshold is None:
            threshold = np.percentile(errors, 95)
        preds = (errors > threshold).astype(int)
        return preds, errors
    
    def get_reconstruction_errors(self, X):
        self.autoencoder.eval()
        with torch.no_grad():
            # Convert to numpy array if it's a DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Normalize data using training statistics
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            recon = self.autoencoder(X_tensor)
            recon = recon.cpu().numpy()
            
            # Denormalize reconstruction
            recon_denorm = recon * (X.std(axis=0) + 1e-8) + X.mean(axis=0)
            
        return np.mean(np.square(X - recon_denorm), axis=1)

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], encoding_dim=16):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final encoding layer
        encoder_layers.extend([
            nn.Linear(prev_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = encoding_dim
        
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoderModel:
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], encoding_dim=16, batch_size=32, epochs=100, learning_rate=0.001, device=None, random_state=42):
        # Set random seeds for reproducible results
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.autoencoder = DeepAutoencoder(input_dim, hidden_dims, encoding_dim).to(self.device)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.batch_size = batch_size
        self.epochs = epochs
        
    def fit(self, X, verbose=1):
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalize data for better training
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.autoencoder(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 15:  # Early stopping patience
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}')
    
    def get_reconstruction_errors(self, X):
        self.autoencoder.eval()
        with torch.no_grad():
            # Convert to numpy array if it's a DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Normalize data using training statistics
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            recon = self.autoencoder(X_tensor)
            recon = recon.cpu().numpy()
            
            # Denormalize reconstruction
            recon_denorm = recon * (X.std(axis=0) + 1e-8) + X.mean(axis=0)
            
        return np.mean(np.square(X - recon_denorm), axis=1)
    
    def predict(self, X, threshold=None):
        errors = self.get_reconstruction_errors(X)
        if threshold is None:
            threshold = np.percentile(errors, 95)
        preds = (errors > threshold).astype(int)
        return preds, errors 