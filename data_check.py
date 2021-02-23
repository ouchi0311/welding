import load_mnist as lm
dataset = lm.load_mnist()

import matplotlib.pyplot as plt

for i in range(20):
   plt.subplot(4, 5, i+1)
   plt.imshow(dataset['x_train'][i,:].reshape(28,28))

plt.show()
