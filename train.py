import torch
import torch.nn as nn
import os
import time
import datetime
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.building_corners import BuildingCornerDataset, collate_fn_corner, get_pixel_features
from models.corner_models import CornerEnum
from models.edge_models import EdgeEnum
from models.unet import ResNetBackbone
from models.loss import CornerCriterion, EdgeCriterion
from models.corner_to_edge import prepare_edge_data
import utils.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr_drop', default=600, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficient/
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./ckpts_heat_256',
                        help='path where to save, empty for no saving')
    parser.add_argument('--corner_model', default='unet',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    return parser


def train_one_epoch(image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, data_loader,
                    optimizer,
                    epoch, max_norm, args):
    backbone.train()
    corner_model.train()
    edge_model.train()
    corner_criterion.train()
    edge_criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40

    pixels, pixel_features = get_pixel_features(image_size)
    pixel_features = pixel_features.cuda()

    for data in metric_logger.log_every(data_loader, print_freq, header):
        corner_outputs, corner_loss, corner_recall, s1_logits, s2_logits_hb, s2_logits_rel, s1_losses, s2_losses_hb, \
        s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel = run_model(
            data,
            pixels,
            pixel_features,
            backbone,
            corner_model,
            edge_model,
            epoch,
            corner_criterion,
            edge_criterion,
            args)

        loss = s1_losses + s2_losses_hb + corner_loss * 0.05 + s2_losses_rel

        loss_dict = {'loss_e_s1': s1_losses, 'loss_e_s2_hb': s2_losses_hb, 'loss_e_s2_rel': s2_losses_rel,
                     'edge_acc_s1': s1_acc, 'edge_acc_s2_hb': s2_acc_hb, 'edge_acc_s2_rel': s2_acc_rel,
                     'loss_c_s1': corner_loss, 'corner_recall': corner_recall}
        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(corner_model.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(edge_model.parameters(), max_norm)

        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run_model(data, pixels, pixel_features, backbone, corner_model, edge_model, epoch, corner_criterion, edge_criterion,
              args):
    image = data['img'].cuda()
    annots = data['annot']
    raw_images = data['raw_img']
    pixel_labels = data['pixel_labels'].cuda()
    gauss_labels = data['gauss_labels'].cuda()

    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)

    # get corner preds from corner model
    image_feats, feat_mask, all_image_feats = backbone(image)
    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)

    corner_loss_s1, corner_recall = corner_criterion(preds_s1, pixel_labels, gauss_labels, epoch)

    # get edge candidates and corresponding G.T.
    c_outputs = preds_s1
    edge_data = prepare_edge_data(c_outputs, annots, raw_images)

    edge_coords = edge_data['edge_coords'].cuda()
    edge_mask = edge_data['edge_coords_mask'].cuda()
    edge_lengths = edge_data['edge_coords_lengths'].cuda()
    edge_labels = edge_data['edge_labels'].cuda()
    corner_nums = edge_data['processed_corners_lengths']

    # run the edge model
    max_candidates = torch.stack([corner_nums.max() * 3] * len(corner_nums), dim=0)
    logits_s1, logits_s2_hb, logits_s2_rel, s2_ids, s2_edge_mask, s2_gt_values = edge_model(image_feats, feat_mask, pixel_features,
                                                                              edge_coords, edge_mask, edge_labels,
                                                                              corner_nums,
                                                                              max_candidates)
    s1_losses, s1_acc, s2_losses_hb, s2_acc_hb, s2_losses_rel, s2_acc_rel = edge_criterion(logits_s1, logits_s2_hb,
                                                                                           logits_s2_rel, s2_ids,
                                                                                           s2_edge_mask,
                                                                                           edge_labels, edge_lengths,
                                                                                           edge_mask, s2_gt_values)

    return c_outputs, corner_loss_s1, corner_recall, logits_s1, logits_s2_hb, logits_s2_rel, s1_losses, \
           s2_losses_hb, s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel


@torch.no_grad()
def evaluate(image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, data_loader, epoch,
             args):
    backbone.eval()
    corner_model.eval()
    edge_model.eval()
    corner_criterion.eval()
    edge_criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    pixels, pixel_features = get_pixel_features(image_size)
    pixel_features = pixel_features.cuda()

    for data in metric_logger.log_every(data_loader, 10, header):
        c_outputs, corner_loss, corner_recall, s1_logits, \
        s2_logits_hb, s2_logits_rel, s1_losses, s2_losses_hb, s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel = run_model(
            data,
            pixels,
            pixel_features,
            backbone,
            corner_model,
            edge_model,
            epoch,
            corner_criterion,
            edge_criterion,
            args)

        loss_dict = {'loss_e_s1': s1_losses,
                     'loss_e_s2_hb': s2_losses_hb,
                     'loss_e_s2_rel': s2_losses_rel,
                     'edge_acc_s1': s1_acc,
                     'edge_acc_s2_hb': s2_acc_hb,
                     'edge_acc_s2_rel': s2_acc_rel,
                     'loss_c_s1': corner_loss,
                     'corner_recall': corner_recall}

        loss = s1_losses + s2_losses_hb + corner_loss * 0.05 + s2_losses_rel
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    DATAPATH = './data/cities_dataset'
    DET_PATH = './data/det_final'
    image_size = 256
    train_dataset = BuildingCornerDataset(DATAPATH, DET_PATH, phase='train', image_size=image_size, rand_aug=True,
                                          training_split='full', inference=False)
    test_dataset = BuildingCornerDataset(DATAPATH, DET_PATH, phase='valid', image_size=image_size, rand_aug=False,
                                         training_split=None, inference=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8,
                                  collate_fn=collate_fn_corner, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4,
                                 collate_fn=collate_fn_corner)

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels

    corner_model = CornerEnum(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                              backbone_num_channels=num_channels)
    backbone = nn.DataParallel(backbone)
    backbone = backbone.cuda()
    corner_model = nn.DataParallel(corner_model)
    corner_model = corner_model.cuda()

    edge_model = EdgeEnum(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                          backbone_num_channels=num_channels)
    edge_model = nn.DataParallel(edge_model)
    edge_model = edge_model.cuda()

    corner_criterion = CornerCriterion(image_size=image_size)
    edge_criterion = EdgeCriterion()

    backbone_params = [p for p in backbone.parameters()]
    corner_params = [p for p in corner_model.parameters()]
    edge_params = [p for p in edge_model.parameters()]

    all_params = corner_params + edge_params + backbone_params
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = args.start_epoch

    if args.resume:
        ckpt = torch.load(args.resume)
        backbone.load_state_dict(ckpt['backbone'])
        corner_model.load_state_dict(ckpt['corner_model'])
        edge_model.load_state_dict(ckpt['edge_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        lr_scheduler.step_size = args.lr_drop

        print('Resume from ckpt file {}, starting from epoch {}'.format(args.resume, ckpt['epoch']))
        start_epoch = ckpt['epoch'] + 1

    n_backbone_parameters = sum(p.numel() for p in backbone_params if p.requires_grad)
    n_corner_parameters = sum(p.numel() for p in corner_params if p.requires_grad)
    n_edge_parameters = sum(p.numel() for p in edge_params if p.requires_grad)
    n_all_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print('number of trainable backbone params:', n_backbone_parameters)
    print('number of trainable corner params:', n_corner_parameters)
    print('number of trainable edge params:', n_edge_parameters)
    print('number of all trainable params:', n_all_parameters)

    print("Start training")
    start_time = time.time()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_acc = 0
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, train_dataloader,
            optimizer,
            epoch, args.clip_max_norm, args)
        lr_scheduler.step()

        val_stats = evaluate(
            image_size, backbone, corner_model, edge_model, corner_criterion, edge_criterion, test_dataloader,
            epoch, args
        )

        val_acc = (val_stats['edge_acc_s1'] + val_stats['edge_acc_s2_hb']) / 2
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
        else:
            is_best = False

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if is_best:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'backbone': backbone.state_dict(),
                    'corner_model': corner_model.state_dict(),
                    'edge_model': edge_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_acc': val_acc,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeoVAE training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
