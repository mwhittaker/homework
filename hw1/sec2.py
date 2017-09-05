import argparse
import logging
import pickle
import tensorflow as tf

def main(args):
    # Set logging verbosity.
    format = "[%(asctime)-15s %(pathname)s:%(lineno)-3s] %(message)s"
    if args.verbose:
        logging.basicConfig(format=format, level=logging.DEBUG)
    else:
        logging.basicConfig(format=format)

    # Load rollouts.
    with open(args.rollouts, "rb") as f:
        rollouts = pickle.load(f)
    observations = rollouts["observations"]
    actions = rollouts["actions"]
    logging.debug("observations.shape = %s", observations.shape)
    logging.debug("actions.shape = %s", actions.shape)


    # Build DNN.
    feature_columns = [tf.feature_column.numeric_column("x", shape=[11])]
    classifier = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                           hidden_units=[128],
                                           label_dimension=3,
                                           model_dir="/tmp/sec2")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": observations},
        y=actions,
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=10000)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "rollouts",
        type=str,
        help="Pickle file with rollouts",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
