import pennylane as qml

symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0],
[0.0000000000,   -0.0000000000,    2.0],
[1.5000000000,   -0.0000000000,    0.0],
[1.5000000000,   -0.0000000000,    2.0],
                ], requires_grad=False) 

charge = 2
basis = 'STO-3G'