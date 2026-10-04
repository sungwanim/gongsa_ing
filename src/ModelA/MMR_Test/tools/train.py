import os
import random

import numpy as np
import pandas as pd
import torch
import logging

from utils import get_dataloaders, load_backbones
from utils.common import freeze_paras, scratch_MAE_decoder, best_f1_threshold, compute_image_classification_metrics

from models.MMR import MMR_base, MMR_pipeline_

import timm.optim.optim_factory as optim_factory

LOGGER = logging.getLogger(__name__)

RESULT_KEYS = ["AUROC", "Pixel-AUROC", "per-region-overlap (PRO)",
               "Recall", "FPR", "Precision", "F1", "Accuracy"]


def _set_seed(cfg):
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)


def _get_test_dataloader_dict(cfg):
    """
    get test dataloader (include each category) for every domain shift setup.
    """
    if cfg.DATASET.name in ["aebad_S", "aebad_V", "mvtec"]:
        if cfg.DATASET.name == "aebad_S":
            measured_list = ["same", "background", "illumination", "view"]
        elif cfg.DATASET.name == "aebad_V":
            measured_list = ["video1", "video2", "video3"]
        else:
            measured_list = ["same"]

        test_dataloader_dict = {}
        for each_class in measured_list:
            cfg.DATASET.domain_shift_category = each_class
            test_dataloaders_ = get_dataloaders(cfg=cfg, mode='test')
            test_dataloader_dict[each_class] = test_dataloaders_
    else:
        raise NotImplementedError("DATASET {} does not include in target datasets".format(cfg.DATASET.name))
    return measured_list, test_dataloader_dict


def _load_models(cfg, load_pretrain_model):
    if cfg.TRAIN.method not in ['MMR']:
        raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

    # target model
    cur_model = load_backbones(cfg.TRAIN.backbone)
    freeze_paras(cur_model)

    # mask model prepare
    mmr_base = MMR_base(cfg=cfg,
                        scale_factors=cfg.TRAIN.MMR.scale_factors,
                        FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim)

    if load_pretrain_model:
        checkpoint = torch.load(cfg.TRAIN.MMR.model_chkpt)
        checkpoint = scratch_MAE_decoder(checkpoint)
        LOGGER.info("train the decoder FPN of MMR from scratch!")

        msg = mmr_base.load_state_dict(checkpoint['model'], strict=False)
        LOGGER.info("MAE load meg: {}".format(msg))
    return cur_model, mmr_base


def _checkpoint_path(cfg, dataloader_name):
    return os.path.join(cfg.OUTPUT_DIR, "checkpoints", "MMR_{}.pth".format(dataloader_name))


def _evaluate_domains(cfg, MMR_instance, measured_list, test_dataloader_dict, idx, dataloader_name, result_collect):
    """
    pixel level: AUROC, PRO
    image level: AUROC, and recall / FPR / precision / F1 / accuracy / confusion matrix at one threshold
    """
    threshold = cfg.TEST.image_threshold if cfg.TEST.image_threshold >= 0 else None

    for each_class in measured_list:
        LOGGER.info(f"current domain shift mode is {each_class}!")
        test_dataloaders = test_dataloader_dict[each_class]

        torch.cuda.empty_cache()
        measured_test_dataloaders = test_dataloaders[idx]
        LOGGER.info("current test individual_dataloader is {}.".format(measured_test_dataloaders.name))
        LOGGER.info("the test data in current individual_dataloader {} are {}.".format(measured_test_dataloaders.name,
                                                                                       len(measured_test_dataloaders.dataset)))
        LOGGER.info("Computing evaluation metrics.")
        """
                            prediction
                        ______1________0____
                      1 |    TP   |   FN   |
        ground truth  0 |    FP   |   TN   |

        ACC = (TP + TN) / (TP + FP + FN + TN)

        precision = TP / (TP + FP)

        recall (TPR) = TP / (TP + FN)

        FPR（False Positive Rate）= FP / (FP + TN)
        """
        if cfg.TRAIN.method == 'MMR':
            result = MMR_instance.evaluation(test_dataloader=measured_test_dataloaders)
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        result_collect["AUROC"].append(result["image_auroc"])
        LOGGER.info("{}'s Image_Level AUROC is {:2f}.%".format(dataloader_name, result["image_auroc"] * 100))

        result_collect["Pixel-AUROC"].append(result["pixel_auroc"])
        LOGGER.info(
            "{}'s Full_Pixel_Level AUROC is {:2f}.%".format(dataloader_name, result["pixel_auroc"] * 100))

        result_collect["per-region-overlap (PRO)"].append(result["pro"])
        LOGGER.info(
            "{}'s per-region-overlap (PRO) AUROC is {:2f}.%".format(dataloader_name, result["pro"] * 100))

        # image level good / defect classification
        if threshold is None:
            threshold = best_f1_threshold(result["image_scores"], result["image_labels"])
            LOGGER.info("image threshold {:.6f} is chosen by best F1 on {}, and reused for the other domains."
                        .format(threshold, each_class))
        cls = compute_image_classification_metrics(result["image_scores"], result["image_labels"], threshold)
        for key, name in [("recall", "Recall"), ("fpr", "FPR"), ("precision", "Precision"),
                          ("f1", "F1"), ("accuracy", "Accuracy")]:
            result_collect[name].append(cls[key])
        LOGGER.info("{}'s image classification @ threshold {:.6f}: Recall {:.2f}%, FPR {:.2f}%, "
                    "Precision {:.2f}%, F1 {:.2f}%, Accuracy {:.2f}%".format(
                        dataloader_name, threshold, cls["recall"] * 100, cls["fpr"] * 100,
                        cls["precision"] * 100, cls["f1"] * 100, cls["accuracy"] * 100))
        LOGGER.info("{}'s confusion matrix (rows: GT defect/good, cols: pred defect/good): "
                    "[[TP {}, FN {}], [FP {}, TN {}]]".format(
                        dataloader_name, cls["TP"], cls["FN"], cls["FP"], cls["TN"]))

        # per-image scores, so the threshold can be revisited without re-running the model
        scores_path = os.path.join(cfg.OUTPUT_DIR, "image_scores_{}_{}.csv".format(dataloader_name, each_class))
        pd.DataFrame({"image_path": result["image_paths"],
                      "label": result["image_labels"],
                      "score": result["image_scores"],
                      "prediction": (result["image_scores"] >= threshold).astype(int)}).to_csv(scores_path, index=False)
        LOGGER.info("image scores saved to {}".format(scores_path))

        # raw anomaly score and anomaly map of every image (rows align with the csv above)
        maps_path = os.path.join(cfg.OUTPUT_DIR, "anomaly_maps_{}_{}.npz".format(dataloader_name, each_class))
        np.savez_compressed(maps_path,
                            image_paths=np.array(result["image_paths"]),
                            labels=result["image_labels"],
                            scores=result["image_scores"],
                            anomaly_maps=result["anomaly_maps"])
        LOGGER.info("anomaly scores and maps saved to {}".format(maps_path))


def _log_mean_results(result_collect):
    for key, values in result_collect.items():
        LOGGER.info(
            "Mean {} is {:2f}.%".format(key, np.mean(np.array(values)) * 100))


def train(cfg=None):
    """
    include data loader load, model load, optimizer, training and test.
    """
    # Set random seed from configs.
    _set_seed(cfg)

    LOGGER.info("load dataset!")
    # get train dataloader (include each category)
    train_dataloaders = get_dataloaders(cfg=cfg, mode='train')
    measured_list, test_dataloader_dict = _get_test_dataloader_dict(cfg)

    cur_device = torch.device("cuda:0")

    result_collect = {key: [] for key in RESULT_KEYS}

    # training process
    for idx, individual_dataloader in enumerate(train_dataloaders):
        LOGGER.info("current individual_dataloader is {}.".format(individual_dataloader.name))
        LOGGER.info("the data in current individual_dataloader {} are {}.".format(individual_dataloader.name,
                                                                                  len(individual_dataloader.dataset)))

        # load model
        cur_model, mmr_base = _load_models(cfg, cfg.TRAIN.MMR.load_pretrain_model)
        if not cfg.TRAIN.MMR.load_pretrain_model:
            LOGGER.info("MAE train from scratch!")

        # optimizer load
        optimizer = None
        if cfg.TRAIN.method in ['MMR']:
            # following timm: set wd as 0 for bias and norm layers (AdamW)
            param_groups = optim_factory.add_weight_decay(mmr_base, cfg.TRAIN_SETUPS.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=cfg.TRAIN_SETUPS.learning_rate, betas=(0.9, 0.95))
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        # start training
        torch.cuda.empty_cache()
        if cfg.TRAIN.method == 'MMR':
            MMR_instance = MMR_pipeline_(cur_model=cur_model,
                                         mmr_model=mmr_base,
                                         optimizer=optimizer,
                                         device=cur_device,
                                         cfg=cfg)
            MMR_instance.fit(individual_dataloader)
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        if cfg.TRAIN.save_model:
            MMR_instance.save_model(_checkpoint_path(cfg, individual_dataloader.name))

        _evaluate_domains(cfg, MMR_instance, measured_list, test_dataloader_dict, idx,
                          individual_dataloader.name, result_collect)

    LOGGER.info("Method training phase complete!")
    _log_mean_results(result_collect)


def test(cfg=None):
    """
    evaluation only: load a trained MMR checkpoint (TEST.checkpoint) and run the test of every domain.
    """
    _set_seed(cfg)

    LOGGER.info("load dataset!")
    measured_list, test_dataloader_dict = _get_test_dataloader_dict(cfg)

    cur_device = torch.device("cuda:0")

    result_collect = {key: [] for key in RESULT_KEYS}

    for idx, subdataset in enumerate(cfg.DATASET.subdatasets):
        dataloader_name = test_dataloader_dict[measured_list[0]][idx].name
        checkpoint = cfg.TEST.checkpoint
        if not checkpoint:
            raise ValueError("TEST.checkpoint is empty; set it to the trained MMR_<name>.pth")

        # the MAE weights are not needed: the whole MMR model comes from the checkpoint
        cur_model, mmr_base = _load_models(cfg, load_pretrain_model=False)
        MMR_instance = MMR_pipeline_(cur_model=cur_model,
                                     mmr_model=mmr_base,
                                     optimizer=None,
                                     device=cur_device,
                                     cfg=cfg)
        MMR_instance.load_model(checkpoint)

        _evaluate_domains(cfg, MMR_instance, measured_list, test_dataloader_dict, idx,
                          dataloader_name, result_collect)

    LOGGER.info("Method test phase complete!")
    _log_mean_results(result_collect)
