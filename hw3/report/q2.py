import matplotlib.pyplot as plt
import pickle
import argparse

def main(args):
    filenames = [args.buffer1000, args.buffer10000, args.buffer100000, args.buffer1000000]
    metrics = []
    for filename in filenames:
        with open(filename, "rb") as f:
            metrics.append(pickle.load(f))

    n = min([len(m["timestep"]) for m in metrics])

    plt.figure()
    labels = ["1000", "10000", "100000", "1000000"]
    for (m, l) in zip(metrics, labels):
        plt.plot(m["timestep"][:n], m["mean_reward"][:n], label=l)
    plt.legend()
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.savefig(args.output_filename)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Output filename"
    )
    parser.add_argument("buffer1000", type=str)
    parser.add_argument("buffer10000", type=str)
    parser.add_argument("buffer100000", type=str)
    parser.add_argument("buffer1000000", type=str)
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
