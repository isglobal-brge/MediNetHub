layer_types = [
        # Linear layers
        {'name': 'Linear', 'category': 'linear', 'description': 'Fully connected layer'},
        
        # Convolutional layers
        {'name': 'Conv1d', 'category': 'conv', 'description': '1D convolution'},
        {'name': 'Conv2d', 'category': 'conv', 'description': '2D convolution'},
        
        # Pooling layers
        {'name': 'MaxPool1d', 'category': 'pool', 'description': 'Max pooling 1D'},
        {'name': 'MaxPool2d', 'category': 'pool', 'description': 'Max pooling 2D'},
        {'name': 'AvgPool1d', 'category': 'pool', 'description': 'Average pooling 1D'},
        {'name': 'AvgPool2d', 'category': 'pool', 'description': 'Average pooling 2D'},
        
        # Activation layers
        {'name': 'ReLU', 'category': 'activation', 'description': 'Rectified Linear Unit'},
        {'name': 'Sigmoid', 'category': 'activation', 'description': 'Sigmoid function'},
        {'name': 'Tanh', 'category': 'activation', 'description': 'Hyperbolic tangent'},
        {'name': 'LeakyReLU', 'category': 'activation', 'description': 'Leaky ReLU'},
        
        # Normalization layers
        {'name': 'BatchNorm1d', 'category': 'norm', 'description': 'Batch normalization 1D'},
        {'name': 'BatchNorm2d', 'category': 'norm', 'description': 'Batch normalization 2D'},
        
        # Recurrent layers
        {'name': 'LSTM', 'category': 'recurrent', 'description': 'Long Short-Term Memory'},
        {'name': 'GRU', 'category': 'recurrent', 'description': 'Gated Recurrent Unit'},
        
        # Utility layers
        {'name': 'Dropout', 'category': 'utility', 'description': 'Dropout for regularization'},
        {'name': 'Flatten', 'category': 'utility', 'description': 'Flatten the input tensor'},
    ]
    
    # Initialize optimizer types
optimizer_types = [
    {'name': 'SGD', 'description': 'Stochastic gradient descent'},
    {'name': 'Adam', 'description': 'Adam optimizer'},
    {'name': 'AdamW', 'description': 'Adam with weight decay'},
    {'name': 'RMSprop', 'description': 'Root Mean Square Propagation'},
    {'name': 'Adagrad', 'description': 'Adaptive Gradient Algorithm'},
]

# Initialize loss function types
loss_types = [
    {'name': 'MSELoss', 'description': 'Mean squared error'},
    {'name': 'BCELoss', 'description': 'Binary cross entropy'},
    {'name': 'BCEWithLogitsLoss', 'description': 'BCE with Sigmoid integrated'},
    {'name': 'CrossEntropyLoss', 'description': 'Multiclass cross entropy'},
    {'name': 'L1Loss', 'description': 'Mean absolute error (L1)'},
]

# Initialize federated strategy types
strategy_types = [
    {'name': 'FedAvg', 'description': 'Federated Averaging'},
    {'name': 'FedProx', 'description': 'Federated Proximal with regularization'},
    {'name': 'FedAdagrad', 'description': 'Federated with Adaptive Gradient'},
]