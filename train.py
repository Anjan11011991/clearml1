# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name='Clearml-Dockerized', task_name='Dockerized', task_type=Task.TaskTypes.optimizer)

train_dataset = datasets.MNIST(root="data", download=False, train=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to(device)
optimizer = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
def train_model(model, train_loader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader: 
            X, y = batch 
            X, y = X.to(device), y.to(device) 
            yhat = model(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the model state
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Load the model state
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()

# Main execution
if __name__ == "__main__": 
    train_model(clf, train_loader, optimizer, loss_fn)

    model_filepath = 'model_state.pt'
    save_model(clf, model_filepath)

    # Load model for inference
    load_model(clf, model_filepath)
