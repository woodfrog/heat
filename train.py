import torch
import torch.nn as nn
import os
import time
import datetime
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from arguments import get_args_parser
from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.s3d_floorplans import S3DFloorplanDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.corner_models import CornerEnum
from models.edge_models import EdgeEnum
from models.resnet import ResNetBackbone
from models.loss import CornerCriterion, EdgeCriterion
from models.corner_to_edge import prepare_edge_data
import utils.misc as utils


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
    print_freq = args.print_freq

    # get the positional encodings for all pixels
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

        loss = s1_losses + s2_losses_hb + s2_losses_rel + corner_loss * args.lambda_corner

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
    edge_data = prepare_edge_data(c_outputs, annots, raw_images, args.max_corner_num)

    edge_coords = edge_data['edge_coords'].cuda()
    edge_mask = edge_data['edge_coords_mask'].cuda()
    edge_lengths = edge_data['edge_coords_lengths'].cuda()
    edge_labels = edge_data['edge_labels'].cuda()
    corner_nums = edge_data['processed_corners_lengths']

    # run the edge model
    max_candidates = torch.stack([corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0)
    logits_s1, logits_s2_hb, logits_s2_rel, s2_ids, s2_edge_mask, s2_gt_values = edge_model(image_feats, feat_mask,
                                                                                            pixel_features,
                                                                                            edge_coords, edge_mask,
                                                                                            edge_labels,
                                                                                            corner_nums,
                                                                                            max_candidates)

    s1_losses, s1_acc, s2_losses_hb, s2_acc_hb, s2_losses_rel, s2_acc_rel = edge_criterion(logits_s1, logits_s2_hb,
                                                                                           logits_s2_rel, s2_ids,
                                                                                           s2_edge_mask,
                                                                                           edge_labels, edge_lengths,
                                                                                           edge_mask, s2_gt_values)

    return c_outputs, corner_loss_s1, corner_recall, logits_s1, logits_s2_hb, logits_s2_rel, s1_losses, s2_losses_hb, \
            s2_losses_rel, s1_acc, s2_acc_hb, s2_acc_rel


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

        loss = s1_losses + s2_losses_hb + s2_losses_rel + corner_loss * args.lambda_corner
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    parser = argparse.ArgumentParser('HEAT training', parents=[get_args_parser()])
    args = parser.parse_args()
    image_size = args.image_size
    if args.exp_dataset == 'outdoor':
        data_path = './data/outdoor/cities_dataset'
        det_path = './data/outdoor/det_final'
        train_dataset = OutdoorBuildingDataset(data_path, det_path, phase='train', image_size=image_size, rand_aug=True,
                                               inference=False)
        test_dataset = OutdoorBuildingDataset(data_path, det_path, phase='valid', image_size=image_size, rand_aug=False,
                                              inference=False)
    elif args.exp_dataset == 's3d_floorplan':
        data_path = './data/s3d_floorplan'
        train_dataset = S3DFloorplanDataset(data_path, phase='train', rand_aug=True, inference=False)
        test_dataset = S3DFloorplanDataset(data_path, phase='valid', rand_aug=False, inference=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.exp_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn)

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

        if args.run_validation:
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
        else:
            val_acc = 0
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
    main()
