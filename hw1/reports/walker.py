import matplotlib.pyplot as plt

def main():
    iterations = range(10)
    mean_return = [
        662.1915603403993,
        3060.1011266998203,
        4395.385817483376,
        4167.215856920285,
        3000.6859654715668,
        2338.9396495212627,
        2907.8122196431345,
        5438.066084807636,
        5229.306945107508,
        5406.487939104338
    ]
    stddev_returns = [
        511.89535942195806,
        1176.8854997228036,
        1292.745136590904,
        1937.203378396513,
        1945.770135296836,
        1484.7331398249435,
        1639.7986169655333,
        55.9247134713606,
        956.5951562405739,
        347.2899280588822
    ]
    expert_mean = [5517 for _ in iterations]
    bc_mean = [662 for _ in iterations]

    plt.figure()
    plt.grid()
    plt.errorbar(iterations, mean_return, yerr=stddev_returns)
    plt.plot(iterations, expert_mean, label="expert")
    plt.plot(iterations, bc_mean, label="behavioral cloning")
    plt.legend()
    plt.xlabel("Dagger iteration")
    plt.ylabel("Mean reward")
    plt.savefig("walker.pdf")

if __name__ == "__main__":
    main()
