import matplotlib.pyplot as plt

def main():
    data = [
        [50, 822.6085254266504, 768.312383026875],
        [100, 2174.8730689285276, 1092.392417542385],
        [250, 2676.178809351866, 755.8759764143606],
        [500, 3568.986645109727, 503.1202954747127],
        [750, 3420.611544793093, 926.9308268765042],
        [1000, 3721.5713696595667, 128.77735285437737],
        [1250, 3883.6777078472614, 191.11675115863864],
        [1500, 3966.848784839093, 156.4866802341518],
    ]
    num_rollouts, means, stddevs = zip(*data)

    plt.figure()
    plt.grid()
    plt.errorbar(num_rollouts, means, yerr=stddevs)
    plt.xlabel("Number of expert rollouts")
    plt.ylabel("Mean reward")
    plt.savefig("cheetah.pdf")

if __name__ == "__main__":
    main()
