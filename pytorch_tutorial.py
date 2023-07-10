import argparse
import torch
import os.path
from subprocess import call
import uproot
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        
        super().__init__()

        self.first_layer = nn.Linear(n_features, hidden_size)
        
        # Implement the rest of the model here, try and do a simple model with 3 layers, you can make a layer by calling nn.Linear(input_size,output_size)        
        # you can append the layers to nn.Sequential() and then the forward step will go trough them sequentially. Don't forget the activation function.
        # Ps: since we are using cross entropy for the loss function you dont need to do softmax at the end since this loss function already has it incorporated.

        self.order=nn.Sequential(self.first_layer) 
        
    def forward(self, x, **kwargs):
        
        x = self.order(x)                             
        return x


def train_batch(X, y, model, optimizer, criterion, **kwargs):

    optimizer.zero_grad()    # reset the grad value

    # finish this function. You need to call your model to obtain the predicted labels
    # then you can do criterion(y_predicted,y_true) to obtain your loss
    # finally simple do loss.backward() to performed backwards propagation and optimizer.step() to update the weights
    # you can do loss.item() to only obtain the loss value with nothing else attached

    return "Error:Not implemented"
    


def predict(model, X):
    
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
   
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, xlabel="Epoch", ylabel='', name=''):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    opt = parser.parse_args()

    
    if not os.path.isfile('tmva_class_example.root'):
        call(['curl', '-L', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])
 
    data = uproot.open('tmva_class_example.root')
    x,y = utils.prepdata(data) 
    dataset = utils.ClassificationDataset(x,y)
    train_set,test_set,val_set =torch.utils.data.random_split(dataset, [0.5,0.25,0.25])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    n_classes = torch.unique(dataset.y).shape[0]  
    n_feats = dataset.X.shape[1]

    # initialize the model

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot


    config = "{}-{}-{}-{}-{}-{}-{}".format(opt.learning_rate, opt.hidden_size, opt.layers, opt.dropout, opt.activation, opt.optimizer, opt.batch_size)

    plot(epochs, train_mean_losses, ylabel='Loss', name='NN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='NN-validation-accuracy-{}'.format(config))

    #plot ROC curve
    
    yscore = torch.nn.functional.softmax(model(test_X),dim=1)[:,1]
    nn_fpr, nn_tpr, nn_thresholds = metrics.roc_curve(test_y.detach().numpy(), yscore.detach().numpy())
    plot(nn_fpr,nn_tpr,ylabel="true positive rate", xlabel="false positive rate",name='ROC-curve-{}'.format(config))



if __name__ == '__main__':
    main()