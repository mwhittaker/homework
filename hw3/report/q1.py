import matplotlib.pyplot as plt
import pickle
import argparse

def main(args):
    with open(args.metrics_file, "rb") as f:
        metrics = pickle.load(f)
    timestep = metrics["timestep"]
    mean_reward = metrics["mean_reward"]
    best_mean_reward = metrics["best_mean_reward"]

    plt.figure()
    plt.plot(timestep, best_mean_reward, label="best mean reward")
    plt.plot(timestep, mean_reward, label="mean reward")
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
    parser.add_argument(
        "metrics_file",
        type=str,
        help="Pickle file with pong metrics"
    )
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
