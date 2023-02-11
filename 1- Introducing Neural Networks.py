#The formula for a single neuron might look something like:
output = sum(inputs * weights) + bias

#We then usually apply an activation function to this output, noted by activation():
output = activation(output)

#Python code for the forward pass of an example neural network model.
loss = -np.log(
    np.sum(
        y + np.exp(
            np.dot(
                np.maximum(
                    0,
                    np.dot(
                        np.maximum(
                            0,
                            np.dot(
                                X,
                                w1.T
                            ) + b1
                        ),
                        w2.T
                    ) + b2
                ),
                w3.T
            ) + b3
        ) /
        np.sum(
            np.exp(
                np.dot(
                    np.maximum(
                        0,
                        np.dot(
                            np.maximum(
                                0,
                                np.dot(
                                    np.maximum(
                                        0,
                                        np.dot(
                                            X,
                                            w1.T
                                        ) + b1
                                    ),
                                    w2.T
                                ) + b2
                            ),
                            w3.T
                        ) + b3
                    ),
                    axis=1,
                    keepdims=True
                )
            )
        )
    )
)