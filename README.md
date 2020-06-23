# BUAA_MNIST_Experiment
e(t) &= \begin{cases} \frac{acutalSpikeCount - desiredSpikeCount}{targetRegionLength} & \text{for }t \in targetRegion\\ \left(\varepsilon * (output - desired)\right)(t) & \text{otherwise} \end{cases} E &= \int_0^T e(t)^2 \text{d}t
