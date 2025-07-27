# Sparse Auto encoder


# Load the libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import pandas as pd


# Set cude, if available
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# load the dataset annd apply transformations
print('Loading the MNIST dataset...')
time.sleep(2)
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5),
                                                     (0.5))])
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               download=True,
                               transform=transform)


# Here we are loading only 9000 samples from the MNIST dataset
# This is done to speed up the training process for demonstration purposes
subset_indices = list(range(5000))
subset_data_set = Subset(train_dataset, subset_indices)

data_loader = DataLoader(subset_data_set, batch_size=100, shuffle=True)

# Define the U-net Auto encoder Class for Sparse Autoencoder
# This class is a modified version of the U-net architecture without skip connections
# It consists of an encoder, a bottleneck, and a decoder


class UnetAutoEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # Encoder
        self.encoder_ = nn.Sequential(
            # Encoder 1
            nn.Conv2d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Encoder 2
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Encoder 3
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Encoder 4
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck_ = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder_ = nn.Sequential(  # Decoder 1
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            # Decoder 2
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Decoder 3
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Decoder 4
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # output Layer
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        self.activation = []  # Cache activation
        self.w_squarred = []  # Cache weight Norms

        current_x = x

        for layer in self.encoder_:
            if isinstance(layer, nn.Conv2d):
                current_x = layer(current_x)
                w = layer.weight
                self.w_squarred = w.pow(2).sum([1, 2, 3])  # output channels
            elif isinstance(layer, nn.ReLU):
                current_x = layer(current_x)
                self.activation.append(current_x)
            elif isinstance(layer, nn.MaxPool2d):
                current_x = layer(current_x)
            else:
                current_x = layer(current_x)

        x = current_x
        encoder = x.clone()
        x = self.bottleneck_(x)
        x = self.decoder_(x)
        return (x, encoder)

    # Encoder
    def encoder(self, x):
        x = self.encoder_(x)
        return (x)

    # Bottleneck
    def bottleneck(self, x):
        x = self.bottleneck_(x)
        # print('bottleneck')
        return (x)

    # Decoder
    def decoder(self, x):
        x = self.decoder_(x)
        return x


# Define the Sparse Autoencoder Class
class ContractiveAutoEncoder(UnetAutoEncoder):
    def __init__(self,in_dim,s_lambda=1e-6,xavier_nprm_init=True):
        super().__init__(in_dim)
        self.jacobian_lambda=s_lambda
        self.xavier_nprm_init=xavier_nprm_init
        # if self.xavier_nprm_init:
        #     nn.init.xavier_uniform_(self.encoder_.weight)
        #     nn.init.xavier_uniform_(self.encoder_.bias,0)
    '''This code to find the jacobian is computationally demanding as our dataset is big, so we have come up with an approximation of the jacobian which is below
        
        def jacobian_penalty(self,input):
            jacobian_matrix=jacobian(self.encoder_,input)
            epsilon= 1e-3
            jacobain_square=jacobian_matrix.pow(2)        
            jacobian_penalty=jacobain_square.sum()

            return self.jacobian_lambda*jacobian_penalty
    We cannot use the below also as it is also taking more time, 
    lets calculate the contractive loss function using   the cached activation defiined in forward function instead of recomputing them
    
    def contractive_loss_approximation(self,x):
        penalty=0.0
        current_x=x

        for layer in self.encoder_:
            if isinstance (layer,nn.Conv2d):
                current_x=layer(current_x)
                w=layer.weight
                w_squarred=w.pow(2).sum([1,2,3]) #output channels
            elif isinstance(layer,nn.ReLU):
                act_deriv=(current_x>0).float()
                penalty +=(act_deriv.pow(2)*w_squarred.view(1,-1,1,1)).sum()
                current_x=layer(current_x)
            elif isinstance(layer,nn.MaxPool2d):
                current_x=layer(current_x)
            else:
                current_x=layer(current_x)
            
        penalty=self.jacobian_lambda*penalty/x.size(0)
        return penalty
    '''
    def contractive_loss_approximation(self):
        penalty=0

        #Iterate throught the cached activation and W_squarred
        for act,w_sq in zip(self.activation,self.w_squarred):
            act_deriv=(act>0).float()
            penalty+=(act_deriv.pow(2)*w_sq.view(1,-1,1,1)).sum()
        penalty=self.jacobian_lambda*penalty/act.size(0)
        return(penalty)

        
    

    def loss_function(self,x_hat,x):
        mse=F.mse_loss(x_hat,x)
        contractive_loss=self.contractive_loss_approximation()
        # print(f'MSE={mse}')
        # print(f'Sparsity={contractive_loss}')
        return mse+contractive_loss
# Function to train the model


def train_model(model, dataloader, epochs, optimizer):
    for epoch in range(epochs):
        total_loss = 0
        for records, _ in dataloader:
            records = records.to(device)
            optimizer.zero_grad()
            decoded, encoded = model.forward(records)
            loss = model.loss_function(decoded, records, encoded)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(
            f'Epoch: {epoch+1}/{epochs}- Train L: {float(total_loss/len(data_loader))}')
        # Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'Unet_Contractive_autoencoder.pkl')
    torch.save(model.state_dict(), 'Unet_Contractive_autoencoder.pkl')


# Initialize the model, optimizer and loss function
# input_features=1 for grayscale images like MNIST
model = ContractiveAutoEncoder(1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# number of epochs for training
epochs = 20

print(f'Summary of the model: \n')
time.sleep(2)
print(f'{summary(model,(1,64,64))}')


# Clear the output
# os.system('cls' if os.name == 'nt' else 'clear')

# Train the model
intake = input(f'Do you want to train the model? If yes, enter y, else n: ')
if intake.lower() == 'y':
    # os.system('cls' if os.name == 'nt' else 'clear')
    train_model(model, data_loader, epochs, optimizer)
else:
    print("Training skipped. \The model will be loaded from the saved file.")
    time.sleep(2)
    # os.system('cls' if os.name == 'nt' else 'clear')
    model = ContractiveAutoEncoder(1)
    model.load_state_dict(torch.load(
        'Unet_Contractive_autoencoder.pkl', map_location=torch.device('cpu')))
    model.to(device)

# Get one sample
print("Loading one sample from the dataset for testing...")
time.sleep(2)
data_loader1 = DataLoader(subset_data_set, batch_size=1, shuffle=True)
data_iter = iter(data_loader1)
img, label = next(data_iter)  # img shape: (1,1,64,64)

# Pass through model
model.eval()
with torch.no_grad():
    img = img.to(device)
    output, _ = model(img)

# Prepare for plotting
input_img = img.squeeze().cpu().numpy()
output_img = output.squeeze().cpu().numpy()

# De-normalize if needed (from [-1,1] back to [0,1])
input_img = (input_img * 0.5) + 0.5
output_img = (output_img * 0.5) + 0.5

# Plot side by side
plt.figure(figsize=(8, 4))

# Input
plt.subplot(1, 2, 1)
plt.imshow(input_img, cmap='gray')
plt.title("Input Image")
plt.axis('off')

# Output
plt.subplot(1, 2, 2)
plt.imshow(output_img, cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')

plt.show()


print('The model is trained and the one smaple output is displayed.')

print('Question 1:\n Plot the t-SNE (use inbuilt function) on the embeddings obtained using the respective auto-encoders.\nColor the clusters using the respective ground-truth class labels.')
time.sleep(2)


# Function to plot t-SNE
def plot_tsne(model):
    subset_indices = list(range(10000))
    subset_data_set = Subset(train_dataset, subset_indices)
    data_loader_eval = DataLoader(
        subset_data_set, batch_size=100, shuffle=True)
    embeddings = []
    labels = []
    print('Plotting t-SNE...')
    model.eval()
    with torch.no_grad():
        for data, label in data_loader_eval:
            data = data.to(device)

            # encoder Embeddings
            encoder_embeddings = model.encoder(data)

            # Flatening the sample for t-sne
            encoder_embeddings_flattened = encoder_embeddings.view(
                encoder_embeddings.size(0), -1)

            embeddings.append(encoder_embeddings_flattened)
            labels.append(label)

    # concatinate the embeddings and labels
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Apply t-sne
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_in_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        embeddings_in_2d[:, 0], embeddings_in_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.title('t-sne of Encoder elements coloured by ground truth')
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.show()


plot_tsne(model)
time.sleep(2)

# Clear the output
os.system('cls' if os.name == 'nt' else 'clear')


print(
    'Question 2:\n Randomly select two images I� and I� from different digit classes, obtain their embeddings h� and h� using the encoder E, and for each � in {0, 0.2, 0.4, 0.6, 0.8, 1}, create a new image I� = �I� + (1�)I�, find its embedding h� = E(I�) as well as the linear interpolation h2� = �h� + (1�)h�, then decode both embeddings to obtain reconstructions (� = D(h�) and (2� = D(h2�), plot these side by side for all �, repeat for 20 such pairs, and report the PSNR between (� and (2� as well as the L2 distance %h�  h2�%� for each �.')


def calculate_psnr(img1, img2, max_value=1):
    '''The images are tensors of same dimension with values [0,1]'''
    mse = torch.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        psnr_value = 20*torch.log10(max_value/mse)
    return psnr_value


def l2norm(img1, img2):
    l2diff = torch.norm(img1-img2, p=2)
    return l2diff


def generated_images():
    subset_indices = list(range(1000))
    subset_data_set = Subset(train_dataset, subset_indices)
    data_loader_q2 = DataLoader(subset_data_set, batch_size=2, shuffle=True)

    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.]
    for i in range(1, 21):
        for featues, labels in data_loader_q2:
            if labels[0] != labels[1]:
                break
            else:
                pass
        print(
            f'\n--------------------------------> For the image Pair {i} <--------------------------------\n')
        for alpha in alphas:
            I1 = featues[0]
            I2 = featues[1]
            I1 = I1.to(device)
            I2 = I2.to(device)

            I_alpha = alpha*I1+(1-alpha)*I2

            model.eval()
            with torch.no_grad():
                # encoder Embeddings
                h_alpha = model.encoder(I_alpha)
                h1 = model.encoder(I1)
                h2 = model.encoder(I2)

                h_dash_alpha = alpha*h1+(1-alpha)*h2

                h_alpha_bottle_neck = model.bottleneck(h_alpha)
                h_dash_alpha_bottle_neck = model.bottleneck(h_dash_alpha)

                i_alpha_hat = model.decoder(h_alpha_bottle_neck)
                i_dash_alpha_hat = model.decoder(h_dash_alpha_bottle_neck)

                psnr = calculate_psnr(i_alpha_hat, i_dash_alpha_hat)
                l2 = l2norm(i_alpha_hat, i_dash_alpha_hat)

                # Prepare for plotting
                i_alpha_hat_img = i_alpha_hat.squeeze().cpu().numpy()
                i_dash_alpha__img = i_dash_alpha_hat.squeeze().cpu().numpy()

                # De-normalize if needed (from [-1,1] back to [0,1])
                i_alpha_hat_img = (i_alpha_hat_img * 0.5) + 0.5
                i_dash_alpha__img = (i_dash_alpha__img * 0.5) + 0.5

                # Plot side by side
                plt.figure(figsize=(8, 4))

                # Input
                plt.subplot(1, 2, 1)
                plt.imshow(i_alpha_hat_img, cmap='gray')
                plt.title("(�")
                plt.axis('off')

                # Output
                plt.subplot(1, 2, 2)
                plt.imshow(i_dash_alpha__img, cmap='gray')
                plt.title("('�")
                plt.axis('off')

                plt.show()
                print(f'PSNR Value {psnr} ----- L2 value {l2}')


# Function to genate images as per the question 2
generated_images()

print('The images are generated as per the question 2 and displayed.\n')


print('Question 3:\n  After training the autoencoders, you want to check if the embeddings of different digits  are different and embeddings within a class are similar. \nFor this purpose, you propose to perform the classification of the digits based on the embeddings obtained by the encoders and check the  accuracy of classifications for each of the Auto-encoder. \nReport the classification accuracy for each  of the AE and report which one is better. Use any inbuilt classifier to solve the classification  problem.')
time.sleep(2)

# Clear the output
# os.system('cls' if os.name == 'nt' else 'clear')


def classify_embeddings():
    subset_indices = list(range(3000))
    subset_data_set = Subset(train_dataset, subset_indices)
    data_loader_q3 = DataLoader(subset_data_set, batch_size=100, shuffle=True)

    sample_features = []
    labels = []
    for featues, label in data_loader_q3:
        featues = featues.to(device)
        model.eval()
        with torch.no_grad():
            # encoder Embeddings
            embedded = model.encoder(featues)
            embedded_flat = embedded.view(embedded.size(0), -1)
            sample_features.append(embedded_flat)
            labels.append(label)
    sample_features = torch.cat(sample_features, dim=0)
    labels = torch.cat(labels, dim=0)

    X_train, X_test, y_train, y_test = train_test_split(
        sample_features, labels, test_size=0.2, random_state=42)
    print(f'Sample Features shape: {sample_features.shape}')
    print(f'labels Shape {labels.shape}')

    # Model Definition
    log_reg_model = LogisticRegression(max_iter=1000)

    # Fit the model
    log_reg_model.fit(X_train.cpu(), y_train.cpu())

    # Predict
    y_pred = log_reg_model.predict(X_test.cpu())

    # Evaluate
    acc = accuracy_score(y_test, y_pred)

    print(f'Accuracy of Classification using Sparse Auto Encoder is: {acc}')

    return acc


# Calling the function to classify embeddings
contractive_accuracy = classify_embeddings()

df=pd.read_csv('AccuracyComparison.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

data={'model':[],'Time':[],'Accuracy':[]}
start_time = time.time()
start_time=time.strftime('%H:%M:%S', time.gmtime(start_time))
data['Time'].append(start_time)
data['model'].append('Contractive Autoencoder')
data['Accuracy'].append(contractive_accuracy)
df1=pd.DataFrame(data)
df=pd.concat([df,df1])

df.to_csv('AccuracyComparison.csv')

print(f'Accuracy details of the models are {df}')

