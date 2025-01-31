![Alexnet mature activation function performance vs non-mature](https://github.com/Xutao-hub95/deep-learning-fundamental/blob/main/relu%20error%20rate%20vs%20non-maturing.jpg?raw=true)

The training error rate of mature activation function(such as Relu) decays faster than traditional smooth non-mature activation curves(i.e.,tanh,(1+e^-x)^-1)
But actually at that time the reasons why relu better or faster are most incorrect, but it is simple so it is generally used up till now.
It introduced how they trained on multiple GPUs, we can ignore this part now since it is more technical rather than the model itself, so we can ignore till we want to run such models by ourselves in the future. Since here we want to work on the more theoretical stuff like how the model worked, so we won't spend too much time on this.
And it has the local response normalization part to prevent the model from saturating but these days it seems not that important and we did have some other better methods for this.
Then it introduced the overlapping pooling,which is generally useful in CNN.
Here is why pooling useful from ChatGPT:
In neural networks, pooling is a technique used primarily in convolutional neural networks (CNNs) to reduce the spatial dimensions (height and width) of an input while retaining important information. The goal of pooling is to downsample the input, making the network more computationally efficient, reducing the risk of overfitting, and allowing it to focus on the most important features.

There are two main types of pooling commonly used:

1. Max Pooling
What it does: Max pooling takes a small window (e.g., a 2x2 or 3x3 grid) and outputs the maximum value from that window.
Purpose: This helps in retaining the most prominent feature in that region while reducing the dimensionality of the feature map.
Example: If you apply max pooling with a 2x2 filter, you take the maximum value from each 2x2 region in your input.
Example of Max Pooling: For a 4x4 matrix:


[1  3  2  4

5  6  1  2

3  2  4  5

7  8  9  1]

After applying 2x2 max pooling, you would get:


[6  4

8  9]
The max value in each 2x2 block is selected.

2. Average Pooling
What it does: Average pooling takes a small window (like max pooling) and computes the average of the values in that window.
Purpose: It tends to smooth out the feature map more than max pooling and can be useful in some cases where averaging the features might be more appropriate than selecting the maximum.
Example: For the same 4x4 matrix, applying average pooling with a 2x2 window would give:


(1+3+5+6)/4 = 3.75

(2+4+1+2)/4 = 2.25

(3+2+7+8)/4 = 5.0

(4+5+9+1)/4 = 4.75


Resulting in:

[3.75  2.25

5.0   4.75]

Why Use Pooling?
Dimensionality Reduction: Pooling reduces the size of the feature maps, which leads to fewer parameters and less computation in the subsequent layers.
Translation Invariance: Pooling helps the model become less sensitive to small translations of features. For instance, an object in an image might shift slightly, but max pooling allows the model to still recognize it.
Avoiding Overfitting: By reducing the dimensionality and making the model focus on the most important features, pooling helps prevent overfitting, especially in deep networks with a large number of parameters.
Focusing on Key Features: Pooling encourages the network to capture the most prominent aspects of features (like edges or textures in images) by selecting either the maximum or average value from a region.
Common Pooling Configurations:
Pooling Size (Filter Size): Common choices are 2x2, 3x3.
Stride: The number of steps the pooling window moves after each operation (often set to the same size as the pooling window).
Padding: Sometimes padding is used to ensure that the input size is divisible by the pool size.
In summary, pooling is used to reduce the spatial dimensions of the input while retaining the most important information, which helps to reduce computational costs and make the neural network more robust to variations in input.

![Alexnet mature activation function performance vs non-mature](https://github.com/Xutao-hub95/deep-learning-fundamental/blob/main/deep%20learning%20model.jpg?raw=true)
This is the deep CNN model.

Methods to handle overfitting:
1. data augmentation:
  randomly extract 224*224 from 256*256 and combine them so we have 2048 pictures, but here actually its not 2048 times enlarge or something since some of them seem similar/identical.
2. PCA:
   channels exchanged and colors gonna change.
3. dropout:
   model ensemble for deep learning is really expensive. Randomly set some amount(0.1) of total nerous to be 0. But actually dropout is equivalent to L2 normalization.However, dropout will make training the model slower than original model without dropout.

They used SGD to train the model and this method then gradually became the most popular methods to train deep learning models.
weight decay: L2 normalization(avoid gradient boost)

Learning details:

![Alexnet mature activation function performance vs non-mature](https://github.com/Xutao-hub95/deep-learning-fundamental/blob/main/details%20of%20learning.jpg?raw=true)

What is momentum? 
ChatGPT: In deep learning, momentum is a technique used to accelerate gradient descent optimization and help improve convergence during the training process. It is particularly useful for speeding up training, preventing oscillations, and helping the model converge to a minimum more smoothly.

Momentum is inspired by the concept from physics, where an object in motion tends to keep moving in the same direction unless acted upon by a force. In the context of neural networks, momentum adds a "memory" of past gradients to the current update of the model’s weights, which helps smooth out the optimization process.

How Momentum Works
In traditional gradient descent, the weights are updated in the opposite direction of the gradient of the loss function with respect to the weights. However, with momentum, the weight updates are influenced not only by the current gradient but also by the previous weight updates.

Formula for Momentum Update


![Alexnet mature activation function performance vs non-mature](https://github.com/Xutao-hub95/deep-learning-fundamental/blob/main/momentem.jpg?raw=true)


How Momentum Helps:
Faster Convergence: Momentum helps the optimizer move faster in the relevant directions (along the contours of the loss function) and slow down in the irrelevant directions (perpendicular to the contours). This can result in faster convergence to a minimum, especially in the presence of "long, narrow" valleys (like in deep networks).

Smoothing Oscillations: In scenarios where gradient descent oscillates (especially in steep areas of the loss surface), momentum helps smooth out these oscillations. It tends to dampen the "zig-zag" behavior caused by rapidly changing gradients.

Escape from Local Minima: Momentum can help the optimization process escape shallow local minima or saddle points in the loss function, enabling the model to reach a better global minimum.

Momentum Hyperparameter (\beta)

β controls how much of the previous gradient (velocity) is carried over to the current update.
A high value (close to 1, e.g., 0.9 or 0.99) gives more "memory" to previous gradients, leading to faster convergence.
A low value (e.g., 0.5 or lower) gives less weight to previous updates, making the process more like traditional gradient descent.
A typical value for momentum is β=0.9, but this can vary depending on the problem and dataset.

Momentum in Practice:
Vanilla Momentum: As described above, this adds the weighted sum of previous gradients to the current gradient.
Nesterov Accelerated Gradient (NAG): A variation of momentum that looks ahead by computing the gradient at the "lookahead" position (current position + momentum). This can give even better performance by providing a more accurate gradient direction.
Example:
In the training process:

The model computes the gradient of the loss with respect to the weights.
The velocity is updated by adding a fraction of the previous velocity and the current gradient.
The weights are updated using the velocity.
Momentum helps smooth out the noise and provides a more stable and efficient path to the optimal solution.

Summary
Momentum in deep learning is a technique to accelerate the convergence of gradient-based optimization algorithms, reduce oscillations in the loss function, and escape local minima. It works by adding a fraction of the previous weight update (velocity) to the current update, thereby "remembering" past gradients. This often results in faster and more stable training, especially in deep neural networks.

What is learning rate warmup?
Learning rate warmup is a technique used in training deep learning models to gradually increase the learning rate from a small value to a target value over a certain number of training steps or epochs. This approach can help stabilize training, particularly in the early stages of training, and improve overall performance.

Why Use Learning Rate Warmup?
Stabilizing Early Training: When training a model, especially with large networks and complex architectures, starting with a high learning rate can cause the model’s weights to change too drastically, leading to unstable training or even divergence. A low learning rate initially allows the model to start with smaller, more controlled updates, which can help prevent large updates from destabilizing the training process.

Avoiding Overshooting: Early in training, the model parameters are often far from the optimal solution. Using a high learning rate right away might cause the optimizer to overshoot the optimal minimum, making it hard to converge to a good solution. By gradually increasing the learning rate, the model is more likely to settle into a better minimum.

Improving Convergence: Warmup can help the model converge faster in some cases. A small learning rate initially allows for stable exploration, and once the model has learned enough about the problem, the learning rate can be increased to speed up convergence.

How Learning Rate Warmup Works
In typical training, the learning rate is constant or decreases according to a pre-defined schedule. With learning rate warmup, the learning rate starts at a small value and increases progressively over a set number of steps or epochs. After the warmup period is over, the learning rate may either remain constant or follow a decay schedule.

Common Warmup Strategies
Linear Warmup: The learning rate increases linearly from a small value to the target learning rate over a fixed number of epochs or iterations.

Formula for linear warmup:
lr=lr_start+(lr_end−lr_start)*(t/T)
​
 
Where:
lr_start is the initial learning rate (usually a very small value).
lr_end is the target learning rate (the learning rate after warmup).
t is the current training step.
T is the total number of warmup steps.
Exponential Warmup: The learning rate increases exponentially during the warmup phase, allowing for a more rapid ramp-up.

Cosine Warmup: This method uses a cosine function to smoothly increase the learning rate from the start value to the target value. It’s often used with a later cosine decay.

When to Use Learning Rate Warmup
Large models and architectures: Warmup is particularly useful for training very large models (like deep transformers or large convolutional networks), where starting with a high learning rate can lead to instability.
Transfer learning: When fine-tuning a pre-trained model, warmup can help the model adjust to the new data without making large changes too quickly.
Aggressive Learning Rate Schedules: When using aggressive learning rate schedules (like cosine annealing), warmup helps in avoiding training instability at the start.
Example of Learning Rate Warmup in Practice
Imagine you're training a neural network with a target learning rate of 0.1 and you want to warm up the learning rate over the first 5 epochs, starting from 0.001.

Warmup period (first 5 epochs):
Epoch 1: Learning rate = 0.001
Epoch 2: Learning rate = 0.02
Epoch 3: Learning rate = 0.04
Epoch 4: Learning rate = 0.06
Epoch 5: Learning rate = 0.08
Epoch 6 (and beyond): Learning rate = 0.1 (constant or decaying based on the schedule)
This gradual increase prevents the model from making large weight updates too early and helps it converge more smoothly.

How to Implement Learning Rate Warmup
Most deep learning frameworks, such as PyTorch and TensorFlow, support learning rate warmup either directly or through custom callbacks.

In PyTorch, for example, you can use torch.optim.lr_scheduler to implement a learning rate schedule and combine it with a warmup phase. Libraries like Hugging Face Transformers also have built-in support for learning rate warmup in their training loops.


In TensorFlow, you can use the tf.keras.optimizers.schedules API, where you can combine warmup and decay schedules as needed.

Summary
Learning rate warmup is a technique to gradually increase the learning rate from a small value to a target value over a number of steps or epochs, particularly at the start of training. This helps stabilize training, avoid overshooting, and improve convergence, especially in deep or complex models. It is often used in combination with other learning rate schedules to ensure smoother training and better final performance.


Learning rate tendency:
This really depends on different models and your dataset. Here are several methods that popular used.
1. ResNet: similar to the methods we saw in this paper, resnet first set to be a default value such as 0.1 then decayed by 0.01 after every 120 epochs.
2. The method introduced from this paper, starting from an initial and then check the validation error, once the error seems not decay efficiently, we gonna multiply the learning rate by 0.1.
3. Another popular method used in industry is that we first choose a relatively small initial learning rate then increase it under a linear function, when it attained some point, we use some smooth decaying function such as cosine or something like that to get smaller and smaller learning rate to fit our model.(less hyperparameters to tune)
