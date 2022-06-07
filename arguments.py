import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Holistic edge attention transformer', add_help=False)
    parser.add_argument('--exp_dataset', default='outdoor',
                        help='the dataset for experiments, outdoor/s3d_floorplan')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr_drop', default=600, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--print_freq', default=40, type=int)
    parser.add_argument('--output_dir', default='./checkpoints/ckpts_heat_outdoor_256',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--max_corner_num', default=150, type=int,
                        help='the max number of corners allowed in the experiments')
    parser.add_argument('--corner_to_edge_multiplier', default=3, type=int,
                        help='the max number of edges based on the number of corner candidates (assuming the '
                             'average degree never greater than 6)')
    parser.add_argument('--lambda_corner', default=0.05, type=float,
                        help='the max number of corners allowed in the experiments')
    parser.add_argument('--run_validation', action='store_true',
            help='Whether run validation or not, default: False')
    return parser
