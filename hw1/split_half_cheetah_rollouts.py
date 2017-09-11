import argparse
import os
import pickle

def main(args):
    if not os.path.exists(args.cheetah_rollouts):
        print("{} does not exist.".format(args.cheetah_rollouts))

    if not os.path.isdir(args.output_dir):
        print("{} does not exist.".format(args.output_dir))

    with open(args.cheetah_rollouts, "rb") as f:
        rollouts = pickle.load(f)
    observations = rollouts["observations"]
    actions = rollouts["actions"]
    assert(observations.shape[0] == 1500000)
    assert(actions.shape[0] == 1500000)

    for n in [50, 100, 250, 500, 750, 1000, 1250, 1500]:
        m = n * 10 * 10000
        filename = "HalfCheetah-v1_{}rollouts.pkl".format(n)
        path = os.path.join(args.output_dir, filename)
        rollouts = {"observations": observations[:m], "actions": actions[:m]}

        with open(path, "wb") as f:
            pickle.dump(rollouts, f)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cheetah_rollouts",
        type=str,
        help="Rollouts file with 1500 HalfCheetah-v1 rollouts."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to place output rollouts",
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
