import torch #pytorch
import torch.nn as nn #pytorch using neural networks basics
import torch.optim as optim #optim is just an algorithm used to change trestbpss and learning rates to reduce losses by the neural network
import torchvision #computer vision related thing used for image transforms and stuff
import torchvision.datasets as datasets #to do dataset related stuff
from torch.utils.data import DataLoader, TensorDataset #torch.utils.data comtains the dataloader that you use to train and test data kinda
import torchvision.transforms as transforms #used for image transforms to use data as images in a usable way for the GAN
import torch.nn.functional as F #for all the functions that dont have parameters
from torch.utils.tensorboard import SummaryWriter #to print to the tensorboard so that we can utilize it

from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
import requests
import numpy as np

#lets make our discriminator class
class Discriminator(nn.Module): #we are inheriting from nn.Module  (inheritance)
    def __init__(self, img_dim): #by using self you can easily access all the instances of one class and all methods and attributes most programming languages already have this inbuilt but python doesnt, you have to specifically call all the instances, methods and attributes
        super().__init__() #like a constructor, when called.. it initializes the objects of the class, also for some reason its better to not explicitly use the name of the base class so its better to just implicitly use __init__, aslo a constructor
        self.disc = nn.Sequential( #nn.sequential is faster than when not using it and also is like a submodule for bigger and more complex modules
            nn.Linear(img_dim, 128), #first linear layer which takes all the in_features 
            nn.LeakyReLU(0.01), #(an activation layer) negative slope is 0.01 and relu is rectified linear unit, the graph is y = x for the positive part but for the negative part there is a small slope too. when you take relu (0 for negative numbers) you might get a vanishing gradient but that doesnt happen in leaky relu
            nn.Linear(128, 64),
            nn.LeakyReLU(0.001),
            nn.Linear(64, 1), #second and last linear layer here.. fake is 0, real is 1
            nn.Sigmoid(), #(second activation layer) to ensure the value is between 0 and 1, we use sigmoid
        )

    def forward(self, x): #the forward function computes the output tensor from the input tensor, and the backward takes the output tensor with respect to some scalar and gets the input with respect to that scalar
        return self.disc(x) 



class Generator(nn.Module): #inhertiting fro nn.Module again
    def __init__(self, z_dim, img_dim): #z_dim is the dimension of hidden noise, img_dim is image dimension
        super().__init__()
        self.gen = nn.Sequential( 
            nn.Linear(z_dim, 256), #converting the noise to 256
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, img_dim), #img_dim will be 784, because we are flattening the 28x28 mnist thing
            nn.Tanh(),  #to make sure values are between -1 and 1, why would pixel values be negative
        )

    def forward(self, x):
        return self.gen(x)



#gans are very sensitive to hyperparameters
#main hyperparameters:
device = "cuda" if torch.cuda.is_available() else "cpu" #using cuda and just checking if its gpu or cpu and telling the user
lr = 3e-4 #learning rate is 3 x 10^(-4), this gets multiplied with the trestbpss and bias, andn the whole line shifts then to occupy more relevent and appropriate dots
z_dim = 64
img_dim = 7  # 784 because trying to make it linear and as a one dimensional tensor, because mnist dataset 28x28
batch_size = 32 #idk why but better to keep in multiples of 2; 32 standard; sending 32 per batch, at a time
num_epochs = 10 # make this 50 later; how many runs should you do, epoch is one run of one batch
exp1_num = 3 #change often -file name
exp1_name = "heart_data" #like 4 layer linear


disc = Discriminator(img_dim).to(device) #initializing discriminator, the previous time you saw this, that is like a constructor 
gen = Generator(z_dim, img_dim).to(device) #initializing generator
fixed_noise = torch.randn((batch_size, z_dim)).to(device) #we also set up fixed noise so that we can see how it has changed, just as a reference of the beginning
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),] #0.1307 and 0.3081 mean and standard deviation (you can approximate value too)
)



df = pd.read_csv(
    "heart.csv", 
    na_values=['NA', '?'])

COLS_USED = ['age', 'sex', 'chest_pain', 'trestbps', 
          'cholestrol', 'fbs', 'restecg','target'],
COLS_TRAIN = ['age', 'sex', 'chest_pain', 'trestbps', 
          'cholestrol', 'fbs', 'restecg']

# all_columns = df[df.columns[0:6]] #error check that website
# df = dataframe

# Handle missing value
df['cholestrol'] = df['cholestrol'] / 300
df['chest_pain'] = df['chest_pain'].fillna(df['chest_pain'].median())
df['trestbps'] = df['trestbps'] / 300
df['age'] = df['age'] / 100

# Split into training and test sets
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df.loc[df['target'] == 1].drop("target", axis=1), ##select fake ones and send it into naive bayes
    df.loc[df['target'] == 1]["target"],
    test_size=0.05,
    random_state=42,
)

# Create dataframe versions for tabular GAN
df_x_test, df_y_test = df_x_test.reset_index(drop=True), \
df_y_test.reset_index(drop=True)
df_y_train = pd.DataFrame(df_y_train)
df_y_test = pd.DataFrame(df_y_test)

# Pandas to Numpy
x_train = df_x_train.values 
x_test = df_x_test.values 
y_train = df_y_train.values 
y_test = df_y_test.values 


tensor_a = torch.from_numpy(x_train).type(torch.FloatTensor)
tensor_b = torch.from_numpy(y_train).type(torch.FloatTensor)
my_dataset = TensorDataset(tensor_a,tensor_b) # create your datset
loader = DataLoader(my_dataset, batch_size = batch_size, shuffle=True) # create your dataloader

for data1 in loader:
    print("data of the heart table shape: ",data1[0].shape)
    break

opt_disc = optim.Adam(disc.parameters(), lr=lr) #adam is like a type of optimizer, its an algorithm specified specifically for training deep neural networks
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss() #measures the performance of a classification model whose output is a probability value between 0 and 1. 
writer_fake = SummaryWriter(f'logs/fake_heart{exp1_name}_{exp1_num}') #to not lose all the old fake ones
writer_real = SummaryWriter(f'logs/real_heart{exp1_name}_{exp1_num}')
step = 0

for epoch in range(num_epochs): #we already defined num_epochs to be 50
    for batch_idx, (real, _) in enumerate(loader): #we take images as real, adn labels we arent taking (because gans are unsupervised?)
        real = real.view(-1, 7).to(device) # flattening, -1 to keep the same number of examples without losing any in our batch and flattening it to 784
        batch_size = real.shape[0] #[0] means 1st dimension 
        
        #Train Discriminator: max log(D(real)) + log(1 - D(G(noise)))
        noise = torch.randn(batch_size, z_dim).to(device) #gaussian distribution (mean=0, std deviation=1)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #cuz real is 1
        disc_fake = disc(fake).view(-1) #flattening everything
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #and fake is 0
        lossD = (lossD_real + lossD_fake) / 2 #net loss is just average
        disc.zero_grad() 
        lossD.backward(retain_graph=True) 
        opt_disc.step() 

        #Train Generator: min log(1 - D(G(noise))) <-> max log(D(G(noise)) (where the second option of maximizing doesn't suffer from saturating gradients)
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.5f}, Loss G: {lossG:.4f}"
            ) 

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 32, 7)
                data = real.reshape(-1, 1, 32, 7) 
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
            
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1


            with open(f'output_simple_gan{epoch}.npy', 'wb') as data_file:
                np.save(data_file, fake.numpy())



#x is the independant variable and y is the dependant variable obviously, and its basically all about mapping the given input independant variable to the respective dependant variable.
#The training set is applied to train, or fit, your model. For example, you use the training set to find the optimal weights, or coefficients
#the whole point of your training set is to find the necessary and workable weights and biases for the data, basically to find a pattern.
#the x_train and x_test are positive integers, but y_train and y_test are only 0s or 1s.

# print(y_test[2]) #outputs correspondingly for x_test
# print(y_train[16]) #outputs correspondingly for x_train
# print(x_test) #this is a given value by the testloader
# print(x_train[16]) #34 index not matching with y_train? #this is a given value by the trainloader
# print(output[0]) #values bw 0 and 1 --accuracy
# print(disc_fake) #values bw 0 and 1
# print(fake[0]) #this is just noise going into the generator
# print(disc_real) #1s or close to 1
# print(real[0]) #where did this real dataset come from?
# print(data) #same as just doing (data)