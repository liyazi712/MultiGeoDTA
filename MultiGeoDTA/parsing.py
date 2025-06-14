def add_train_args(parser):
    # Dataset parameters
    parser.add_argument('--task', help='Task name')
    parser.add_argument('--split_method', default='random',
        choices=['new_compound', 'new_protein', 'new_new'],
        help='Split method: unseen protein, drug, or both')
    parser.add_argument('--thre', default='0.5',
        choices=['0.3', '0.4', '0.5', '0.6'],
        help='cluster threhold')
    parser.add_argument('--seed', type=int, default=42,
        help='Random Seed')

    # Data representation parameters
    parser.add_argument('--contact_cutoff', type=float, default=8.,
        help='cutoff of C-alpha distance to define protein contact graph')
    parser.add_argument('--num_pos_emb', type=int, default=16,
        help='number of positional embeddings')
    parser.add_argument('--num_rbf', type=int, default=16,
        help='number of RBF kernels')

    # Protein model parameters
    parser.add_argument('--prot_gcn_dims', type=int, nargs='+', default=[128, 256, 256],
        help='protein GCN layers dimensions')
    parser.add_argument('--prot_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='protein FC layers dimensions')

    # Drug model parameters
    parser.add_argument('--drug_gcn_dims', type=int, nargs='+', default=[128, 64],
        help='drug GVP hidden layers dimensions')
    parser.add_argument('--drug_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='drug FC layers dimensions')

    # Top model parameters
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512],
        help='top MLP layers dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
        help='dropout rate in top MLP')

    # Training parameters
    parser.add_argument('--n_ensembles', type=int, default=5,
        help='number of ensembles')
    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100,
        help='number of epochs')
    parser.add_argument('--patience', action='store', type=int,
        help='patience for early stopping')
    parser.add_argument('--eval_freq', type=int, default=1,
        help='evaluation frequency')
    parser.add_argument('--test_freq', type=int,
        help='test frequency')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument('--monitor_metric', default='mse',
        help='validation metric to monitor for deciding best checkpoint')
    parser.add_argument('--parallel', action='store_true',
        help='run ensembles in parallel on multiple GPUs')
    parser.add_argument('--device', type=int, help='device id: 0/1/2/....', default=0)

    # Save parameters
    parser.add_argument('--output_dir', action='store', default='../output', help='output folder')
    parser.add_argument('--save_log', action='store_true', default=True, help='save log file')
    parser.add_argument('--save_checkpoint', action='store_true', default=True, help='save checkpoint')
    parser.add_argument('--save_prediction', action='store_true', default=True, help='save prediction')

    # test parameters
    parser.add_argument('--model_file', action='store', default=False, help='test model file dir')


