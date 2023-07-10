import argparse
import ROOT
from ROOT import TMVA, TFile, TTree, TCut
import os.path

import torch
import torch.nn as nn

from subprocess import call

def DNN(n_classes, n_features, hidden_size, layers,activation_type, dropout):
    
    first_layer = nn.Linear(n_features, hidden_size)
    hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layers-1)])
    output_layer = nn.Linear(hidden_size, n_classes)
    if activation_type == "relu": activation = nn.ReLU()
    elif activation_type == "tanh": activation = nn.Tanh()
    drop=nn.Dropout(p=dropout)
    model=nn.Sequential(first_layer,activation,drop)
    for k in range(layers-1):                                           
        model.append(hidden_layers[k])
        model.append(activation)
        model.append(drop)
    model.append(output_layer)  
    model.append(nn.Softmax(dim=1)) 
    return model

def train(model, train_loader, val_loader, num_epochs, batch_size, optimizer, criterion, save_best, scheduler):
    
    trainer = optimizer(model.parameters(), lr=0.01, weight_decay=0.0)
    schedule, schedulerSteps = scheduler
    best_val = None
 
    for epoch in range(num_epochs):
        # Training Loop
        # Set to train mode
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            trainer.zero_grad()
            output = model(X)
            train_loss = criterion(output, y)
            train_loss.backward()
            trainer.step()
 
            # print train statistics
            running_train_loss += train_loss.item()
            if i % 32 == 31:    # print every 32 mini-batches
                print("[{}, {}] train loss: {:.3f}".format(epoch+1, i+1, running_train_loss / 32))
                running_train_loss = 0.0
 
        if schedule:
            schedule(optimizer, epoch, schedulerSteps)
 
        # Validation Loop
        # Set to eval mode
        model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                output = model(X)
                val_loss = criterion(output, y)
                running_val_loss += val_loss.item()
 
            curr_val = running_val_loss / len(val_loader)
            if save_best:
               if best_val==None:
                   best_val = curr_val
               best_val = save_best(model, curr_val, best_val)
 
            # print val statistics per epoch
            print("[{}] val loss: {:.3f}".format(epoch+1, curr_val))
            running_val_loss = 0.0
 
    print("Finished Training on {} Epochs!".format(epoch+1))
 
    return model

def predict(model, test_X, batch_size=32):
    # Set to eval mode
    model.eval()
 
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_X))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X = data[0]
            outputs = model(X)
            predictions.append(outputs)
        preds = torch.cat(predictions)
 
    return preds.numpy()

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', default=20, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-hidden_size', type=int, default=64)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-activation',
                    choices=['tanh', 'relu'], default='tanh')
parser.add_argument('-optimizer',
                    choices=['sgd', 'adam'], default='adam')
opt = parser.parse_args()

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

cuts=""
cutb=""

mycutS=TCut(cuts)
mycutB=TCut(cutb)

if not os.path.exists("dataset/results/rootfiles"):
    os.makedirs("dataset/results/rootfiles")
if not os.path.exists("dataset/weights"):
    os.makedirs("dataset/weights")

outfname='dataset/results/rootfiles/TMVA_pytorch.root' 
outweightname='TMVA_pytorch' 
output = TFile.Open(outfname, 'RECREATE') 

factory = TMVA.Factory('TMVAClassification', output, '!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification')

dataloader = TMVA.DataLoader('dataset')

if not os.path.isfile('tmva_class_example.root'):
    call(['curl', '-L', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])
 
data = TFile.Open('tmva_class_example.root')
signal = data.Get('TreeS')
background = data.Get('TreeB')

for branch in signal.GetListOfBranches():
    dataloader.AddVariable(branch.GetName())

if not os.path.exists("dataset/weights/%s" % outweightname):
    os.makedirs("dataset/weights/%s" % outweightname)

(ROOT.TMVA.gConfig().GetIONames()).fWeightFileDir ="weights/%s" % outweightname 

signalWeight     = 1.0
backgroundWeight = 1.0

dataloader.AddSignalTree( signal, signalWeight )
dataloader.AddBackgroundTree( background, backgroundWeight )

dataloader.PrepareTrainingAndTestTree( mycutS, mycutB, "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:SplitSeed=101:NormMode=NumEvents:!V" )
n_classes = 2
n_feats = 4

model = DNN(
    n_classes,
    n_feats,
    opt.hidden_size,
    opt.layers,
    opt.activation,
    opt.dropout
)

# get an optimizer
if opt.optimizer =="sgd":
    optimizer=torch.optim.SGD
elif opt.optimizer =="adam":
    optimizer=torch.optim.Adam

# get a loss criterion
loss = nn.CrossEntropyLoss()

load_model_custom_objects = {"optimizer": optimizer, "criterion": loss, "train_func": train, "predict_func": predict}

m = torch.jit.script(model)
torch.jit.save(m, "dataset/results/modelClassification.pt")
print(m)


# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyTorch, 'PyNN',
                "H:!V:VarTransform=N:FilenameModel=dataset/results/modelClassification.pt:FilenameTrainedModel=dataset/results/trainedModelClassification.pt:NumEpochs=%s:BatchSize=%s" % (opt.epochs,opt.batch_size)) 
factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTs",
                "!H:!V:NTrees=200:MinNodeSize=5.0%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.50:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=30")


# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()


# Plot ROC Curves
roc = factory.GetROCCurve(dataloader)
roc.SaveAs('dataset/results/ROC_ClassificationPyTorch.png')
cutsr=ROOT.TString("cuts")
cutbr=ROOT.TString("cutb")
varinfor=ROOT.TString("varinfo")
output.cd("dataset")
info = TTree("tmvainfo", "TMVA info")
info.Branch("cuts", cutsr)
info.Branch("cutb", cutbr)
info.Branch("var", varinfor)
info.Fill()
info.Write()

output.Close()

ROOT.TMVA.TMVAGui(outfname)

