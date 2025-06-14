import argparse
from MultiGeoDTA.parsing import add_train_args
from MultiGeoDTA.experiment import DTAExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KDBNet experiment')
    add_train_args(parser)
    args = parser.parse_args()

    exp = DTAExperiment(
        task=args.task, #
        contact_cutoff=args.contact_cutoff,
        num_rbf=args.num_rbf,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        lr=args.lr,
        setting=args.split_method,
        thre=args.thre,
        parallel=args.parallel,
        output_dir=args.output_dir,
        save_log=args.save_log,
        save_checkpoint=args.save_checkpoint,
        device=args.device
    )

    if args.save_prediction or args.save_log or args.save_checkpoint:
        exp.saver.save_config(args.__dict__, 'args.yaml')

    exp.test_model_saved(model_file=args.model_file,save_prediction=args.save_prediction)



