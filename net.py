from paddle import fluid


class MyNet(fluid.dygraph.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        pass
    
    def forward(self, inputs, label=None):
        x = 0
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        return x