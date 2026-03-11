import logging
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.reranking import re_ranking, euclidean_distance, cosine_distance
from utils.reid_metric import eval_func


def extract_features_for_rerank(model, val_loader, device):
    """
    从 val_loader 中提取全部特征、pid、camid
    返回：
        feats: Tensor [N, D]
        pids: np.ndarray [N]
        camids: np.ndarray [N]
        img_paths: list
    """
    model.eval()
    feats = []
    pids = []
    camids = []
    img_paths = []

    for _, (img, pid, camid, camids_tensor, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids_tensor = camids_tensor.to(device)
            target_view = target_view.to(device)

            feat = model(img, cam_label=camids_tensor, view_label=target_view)

            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
            img_paths.extend(imgpath)

    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    return feats, pids, camids, img_paths


def evaluate_with_rerank(cfg, model, val_loader, num_query, device, logger, epoch=None):
    """
    使用 re-ranking 重新评估
    """
    feats, pids, camids, _ = extract_features_for_rerank(model, val_loader, device)

    if cfg.TEST.FEAT_NORM:
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    # re-ranking 参数，可按需调整
    k1 = getattr(cfg.TEST, "RE_RANK_K1", 20)
    k2 = getattr(cfg.TEST, "RE_RANK_K2", 6)
    lambda_value = getattr(cfg.TEST, "RE_RANK_LAMBDA", 0.3)

    distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)

    if epoch is not None:
        logger.info("Validation Results with Re-Ranking - Epoch: {}".format(epoch))
    else:
        logger.info("Validation Results with Re-Ranking")

    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc, mAP


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             local_rank):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = torch.device("cuda", local_rank)
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')

    if torch.cuda.is_available():
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), img.shape[0])

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                try:
                    current_lr = scheduler._get_lr(epoch)[0]
                except:
                    current_lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        current_lr
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    save_model = model.module if hasattr(model, "module") else model
                    torch.save(
                        save_model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                    )
            else:
                save_model = model.module if hasattr(model, "module") else model
                torch.save(
                    save_model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                )

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    # 先输出原始评估结果
                    model.eval()
                    evaluator.reset()

                    for _, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    # 再输出 re-ranking 结果
                    if getattr(cfg.TEST, "RE_RANKING", True):
                        evaluate_with_rerank(cfg, model, val_loader, num_query, device, logger, epoch)

                    torch.cuda.empty_cache()
            else:
                model.eval()
                evaluator.reset()

                for _, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                if getattr(cfg.TEST, "RE_RANKING", True):
                    evaluate_with_rerank(cfg, model, val_loader, num_query, device, logger, epoch)

                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = torch.device("cuda")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # 先做原始评估
    model.eval()
    img_path_list = []

    for _, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results (Original)")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # 再做 re-ranking 评估
    if getattr(cfg.TEST, "RE_RANKING", True):
        cmc_rerank, mAP_rerank = evaluate_with_rerank(cfg, model, val_loader, num_query, device, logger)
        return cmc_rerank[0], cmc_rerank[4]

    return cmc[0], cmc[4]