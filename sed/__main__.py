"""Command-line interface.

    python -m sed prepare                 # cache frames from the dataset
    python -m sed sweep knn|random_forest|boosting|baseline
    python -m sed evaluate                # select on val, threshold, score on test
    python -m sed export                  # ship the selected model to models/
    python -m sed figures                 # regenerate docs/figures/
    python -m sed predict FILE.npz        # frame-level detections for one file
"""
import argparse

from .models import GRIDS


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="build cache/frames.npz and print split statistics")
    p = sub.add_parser("sweep", help="run a model's hyperparameter grid")
    p.add_argument("model", choices=list(GRIDS))
    sub.add_parser("evaluate", help="select configs on val and evaluate on test")
    p = sub.add_parser("export", help="save the selected model to models/")
    p.add_argument("model", nargs="?", default="boosting", choices=list(GRIDS))
    sub.add_parser("figures", help="regenerate README figures")
    p = sub.add_parser("predict", help="detect events in one .npz feature file")
    p.add_argument("path")
    p.add_argument("--plot", help="optional path to save a timeline figure (.png)")
    args = parser.parse_args()

    if args.cmd == "prepare":
        from .data import build_cache, describe, load_dataset
        build_cache()
        print(describe(load_dataset()).to_string())
    elif args.cmd == "sweep":
        from .experiment import sweep
        sweep(args.model)
    elif args.cmd == "evaluate":
        from .experiment import evaluate
        evaluate()
    elif args.cmd == "export":
        from .experiment import export_model
        export_model(args.model)
    elif args.cmd == "figures":
        from .visualization.figures import make_all
        make_all()
    elif args.cmd == "predict":
        from .inference import predict_file
        predict_file(args.path, plot_path=args.plot)


if __name__ == "__main__":
    main()
