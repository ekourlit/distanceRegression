import torch

activationDict = torch.nn.ModuleDict([
    ['lrelu', torch.nn.LeakyReLU()],
    ['relu', torch.nn.ReLU()],
    ['softmax',torch.nn.Softmax()],
    ['sigmoid',torch.nn.Sigmoid()],
    ['tanh',torch.nn.Tanh()],
])


def regression(inputShape, outputDim, hiddenSpaceSizes=[50], activations=['relu']):
    """! 
    This function assumes that we will be using a linear multiclass model.
    """
    modules = [torch.nn.Linear(inputShape, hiddenSpaceSizes[0]), activationDict[activations[0]]]
    
    for hiddenSpaceSizeI in range(1, len(hiddenSpaceSizes)-1):
        modules.extend([torch.nn.Linear(hiddenSpaceSizes[hiddenSpaceSizeI], hiddenSpaceSizes[hiddenSpaceSizeI+1]), activationDict[activations[hiddenSpaceSizeI]]])
    modules.extend([torch.nn.Linear(hiddenSpaceSizes[-1], outputDim)])
    model = torch.nn.Sequential(*modules)
               
    return model

