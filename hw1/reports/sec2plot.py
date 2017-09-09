import matplotlib.pyplot as plt
import numpy as np

def main():
    X = np.genfromtxt("sec2.txt", delimiter=" ")
    steps = X[:,0]
    loss = X[:,1]

    plt.figure()
    plt.semilogy(steps, loss)
    plt.grid()
    plt.xlabel("Training iteration")
    plt.ylabel("Training loss (average mean squared error)")
    plt.savefig("sec2.pdf")

if __name__ == "__main__":
    main()
