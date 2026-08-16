import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = 4.5, 3
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300

softplus = lambda x, beta: 1/beta * np.logaddexp(0, beta*x)
x = np.linspace(-0.2, 0.1, 200)
relu = np.maximum(x, 0)
b40 = softplus(x, 40)
b80 = softplus(x, 80)

plt.plot(x, np.column_stack((relu, b40, b80)), label=['y=ReLU(x)', 'y=softplus(x,40)', 'y=softplus(x,80)'])
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('softplus.pdf')
