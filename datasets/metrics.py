from typing import List, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from radiance_fields import RadianceField


def compute_valid_depth_rmse(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the root mean squared error (RMSE) between the predicted and target depth values,
    only considering the valid rays (where target > 0).

    Args:
    - prediction (Tensor): predicted depth values
    - target (Tensor): target depth values

    Returns:
    - float: RMSE between the predicted and target depth values, only considering the valid rays
    """
    prediction, target = prediction.squeeze(), target.squeeze()
    valid_mask = target > 0
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    return F.mse_loss(prediction, target).sqrt().item()


def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def compute_ssim(
    prediction: Union[Tensor, np.ndarray], target: Union[Tensor, np.ndarray]
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between the prediction and target images.

    Args:
        prediction (Union[Tensor, np.ndarray]): The predicted image.
        target (Union[Tensor, np.ndarray]): The target image.

    Returns:
        float: The SSIM value between the prediction and target images.
    """
    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()
    assert target.max() <= 1.0 and target.min() >= 0.0, "target must be in range [0, 1]"
    assert (
        prediction.max() <= 1.0 and prediction.min() >= 0.0
    ), "prediction must be in range [0, 1]"
    return ssim(target, prediction, data_range=1.0, channel_axis=-1)


def compute_scene_flow_metrics(pred: Tensor, labels: Tensor):
    """
    Computes the scene flow metrics between the predicted and target scene flow values.
    # modified from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/0e4f403c73cb3fcd5503294a7c461926a4cdd1ad/utils.py#L12

    Args:
        pred (Tensor): predicted scene flow values
        labels (Tensor): target scene flow values
    Returns:
        dict: scene flow metrics
    """
    l2_norm = torch.sqrt(
        torch.sum((pred - labels) ** 2, 2)
    ).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / (labels.norm(dim=-1, keepdim=True) + 1e-7)
    unit_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-7)

    # it doesn't make sense to compute angle error on zero vectors
    # we use a threshold of 0.1 to avoid noisy gt flow
    non_zero_flow_mask = labels_norm > 0.1
    unit_label = unit_label[non_zero_flow_mask]
    unit_pred = unit_pred[non_zero_flow_mask]

    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(-1).clamp(min=-1 + eps, max=1 - eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return {
        "EPE3D": EPE3D,
        "acc3d_strict": acc3d_strict,
        "acc3d_relax": acc3d_relax,
        "outlier": outlier,
        "angle_error": angle_error,
    }


def knn_predict(
    queries: Tensor,
    memory_bank: Tensor,
    memory_labels: Tensor,
    n_classes: int,
    knn_k: int = 1,
    knn_t: float = 0.1,
) -> Tensor:
    """
    Compute kNN predictions for each query sample in memory_bank based on memory_labels.

    Args:
        queries (Tensor): query feature vectors
        memory_bank (Tensor): memory feature vectors
        memory_labels (Tensor): memory labels
        n_classes (int): number of classes
        knn_k (int, optional): number of nearest neighbors. Defaults to 1.
        knn_t (float, optional): temperature for softmax. Defaults to 0.1.

    Returns:
        Tensor: kNN predictions for each query sample in queries based on memory_bank and memory_labels
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(queries, memory_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        memory_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        queries.size(0) * knn_k, n_classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(queries.size(0), -1, n_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_predict(
    queries: Tensor,
    memory_bank: Tensor,
    memory_labels: Tensor,
    n_classes: int,
    knn_k: int = 1,
    knn_t: float = 0.1,
    similarity: str = "cosine",
) -> Tensor:
    """
    Compute kNN predictions for each query sample in memory_bank based on memory_labels.

    Args:
        queries (Tensor): query feature vectors [N_q, D]
        memory_bank (Tensor): Transposed memory feature vectors: [D, N_m]
        memory_labels (Tensor): memory labels
        n_classes (int): number of classes
        knn_k (int, optional): number of nearest neighbors. Defaults to 1.
        knn_t (float, optional): temperature for softmax. Defaults to 0.1.
        similarity (str, optional): similarity metric to use. Defaults to "cosine".

    Returns:
        Tensor: kNN predictions for each query sample in queries based on memory_bank and memory_labels
    """
    if similarity == "cosine":
        # compute cos similarity between each feature vector and feature bank ---> [N_q, N_m]
        assert queries.size(-1) == memory_bank.size(0)
        memory_bank = memory_bank.T
        memory_bank = memory_bank / (memory_bank.norm(dim=-1, keepdim=True) + 1e-7)
        similarity_matrix = torch.mm(
            queries / (queries.norm(dim=-1, keepdim=True) + 1e-7),
            memory_bank.T,
        )
    elif similarity == "l2":
        # compute the L2 distance using broadcasting
        queries_expanded = queries.unsqueeze(1)  # Shape becomes [N_q, 1, D]
        memory_bank_expanded = memory_bank.T.unsqueeze(0)  # Shape becomes [1, N_m, D]
        dist_matrix = torch.norm(
            queries_expanded - memory_bank_expanded, dim=2
        )  # Shape becomes [N_q, N_m]
        # Invert the distances to get the similarity
        similarity_matrix = 1 / (dist_matrix + 1e-9)  # Shape remains [N_q, N_m]
    else:
        raise ValueError(f"similarity {similarity} is not supported")
    # [N_q, K]
    sim_weight, sim_indices = similarity_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(
        memory_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices
    )
    # scale by temperature
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        queries.size(0) * knn_k, n_classes, device=sim_labels.device
    )
    # [N_q * K, num_class]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # [N_q, num_class]
    pred_scores = torch.sum(
        one_hot_label.view(queries.size(0), -1, n_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def collect_centroids(
    train_indices: List[int],
    dataset,  # a WaymoDataset object
    model: RadianceField,
    device: torch.device,
):
    """
    Collects centroids for each class in the dataset (indexed by the train_indices)

    Args:
        train_indices (list): List of indices to use for training.
        dataset (Dataset): Dataset object containing the data.
        model (Model): Model object to use for querying attributes.
        device (str): Device to use for computation.

    Returns:
        tuple: A tuple containing:
            - centroids_bank (Tensor): Tensor containing the centroids for each label.
            - label_bank (Tensor): Tensor containing the label for each centroid.
    """
    memory_bank, label_bank = [], []
    for i in tqdm(train_indices, desc="Collecting Centroids", dynamic_ncols=True):
        world_occ_coords, occ_labels, normed_time = dataset.get_occ(i)
        world_occ_coords = world_occ_coords.to(device)
        occ_labels = occ_labels.to(device)
        normed_time = normed_time.to(device)
        with torch.no_grad():
            reuslts = model.forward(
                positions=world_occ_coords,
                data_dict={"normed_timestamps": normed_time},
                combine_static_dynamic=True,
                query_feature_head=False,  # save some time by skipping dino prediction
                query_pe_head=False,
            )
        # occ3D's annotations are noisy and sometimes wrong on far-away objects like road.
        # hese outliers will affect the accuracy of the centroids, so we filter them out.
        # in addition, occ3D's annotations are obtained from 360 degree lidar,
        # while all emernerf can "see" are from the 3 (or potentially 5) cameras.
        # so we only evaluate on the intersection of the two.
        density_filter = (reuslts["density"] > 0.2).squeeze()
        occ_labels = occ_labels[density_filter]
        world_occ_coords = world_occ_coords[density_filter]
        normed_time = normed_time[density_filter]

        with torch.no_grad():
            reuslts = model.forward(
                positions=world_occ_coords,
                data_dict={"normed_timestamps": normed_time},
                combine_static_dynamic=True,
                query_feature_head=True,
                query_pe_head=False,
            )
        memory_bank.append(reuslts["dino_feat"])
        label_bank.append(occ_labels)
    memory_bank, label_bank = torch.cat(memory_bank, dim=0), torch.cat(
        label_bank, dim=0
    )
    # compute centroids for each class
    centroids = {}
    for label in torch.unique(label_bank):
        centroids[int(label)] = torch.mean(
            memory_bank[label_bank == label], dim=0, keepdim=True
        )

    # deal with classes that are not in the scene
    centroids_bank = []
    label_bank = []
    for i in range(len(dataset.label_mapping)):
        if i in centroids:
            centroids_bank.append(centroids[i])
        else:
            centroids_bank.append(torch.zeros(1, 64).to(device))
        label_bank.append(i)
    centroids_bank = torch.cat(centroids_bank, dim=0)
    label_bank = torch.tensor(label_bank).to(device)
    return centroids_bank, label_bank


def eval_few_shot_occ(
    test_indices: List[int],
    dataset,  # a WaymoDataset object
    model: RadianceField,
    device: torch.device,
    centroids_bank: Tensor,
    label_bank: Tensor,
):
    """
    Evaluates the few-shot voxel classification using nearest neighbor classifier.

    Args:
        test_indices (list): List of indices to use for testing.
        dataset (Dataset): Dataset object containing the data.
        model (Model): Model object to use for querying attributes.
        device (Device): Device to use for computation.
        centroids_bank (Tensor): Tensor containing the centroids for each label.
        label_bank (Tensor): Tensor containing the label for each centroid.

    Returns:
        dict: A dictionary containing the following metrics:
            - micro_accuracy (float): Sample-averaged Micro accuracy.
            - macro_accuracy (float): Class-averaged Macro accuracy.
            - per_class_accuracy (dict): Per class accuracy.
            - cover_rate (float): Cover rate of tested points.
            - num_measured_points (int): Number of measured points.
            - num_total_points (int): Number of total points.
    """
    # Some initialization to collect running mean of accuracy
    correct_predictions, total_predictions = 0, 0
    num_measured_points, num_total_points = 0, 0
    correct_per_class = {label: 0 for label in dataset.label_mapping}
    total_per_class = {label: 0 for label in dataset.label_mapping}

    pbar = tqdm(test_indices, dynamic_ncols=True, leave=True)
    with torch.no_grad():
        for i in pbar:
            world_occ_coords, occ_labels, normed_time = dataset.get_occ(i)
            world_occ_coords = world_occ_coords.to(device)
            occ_labels = occ_labels.to(device)
            normed_time = normed_time.to(device)
            # record the number of points from Occ3D before filtering
            num_total_points += len(occ_labels)

            with torch.no_grad():
                results_dict = model.forward(
                    positions=world_occ_coords,
                    data_dict={"normed_timestamps": normed_time},
                    combine_static_dynamic=True,
                    query_feature_head=False,  # save some time by skipping dino prediction
                    query_pe_head=False,
                )

            # occ3D's annotations are noisy and sometimes wrong on far-away objects like road.
            # hese outliers will affect the accuracy of the centroids, so we filter them out.
            # in addition, occ3D's annotations are obtained from 360 degree lidar,
            # while all emernerf can "see" are from the 3 (or potentially 5) cameras.
            # so we only evaluate on the intersection of the two.
            density_filter = (results_dict["density"] > 0.2).squeeze()

            # skip if there are no points left after filtering
            if density_filter.sum() == 0:
                continue
            world_occ_coords = world_occ_coords[density_filter]
            occ_labels = occ_labels[density_filter]
            normed_time = normed_time[density_filter]
            # record the number of points from Occ3D after filtering
            num_measured_points += len(occ_labels)

            with torch.no_grad():
                results_dict = model.forward(
                    positions=world_occ_coords,
                    data_dict={"normed_timestamps": normed_time},
                    combine_static_dynamic=True,
                    query_feature_head=True,
                    query_pe_head=False,
                )
            dino_feats = results_dict["dino_feat"]
            predicted_labels = knn_predict(
                dino_feats,
                centroids_bank.T,
                label_bank,
                n_classes=len(dataset.label_mapping),
                knn_k=1,  # nearest neighbor
                knn_t=0.1,  # temperature, not used when knn_k=1
            )[..., 0]

            # compute accuracy
            correct = predicted_labels == occ_labels
            # Update total_per_class and correct_per_class without for-loop
            unique_labels, counts = torch.unique(occ_labels, return_counts=True)
            total_per_class.update(
                {
                    label.item(): total_per_class[label.item()] + count.item()
                    for label, count in zip(unique_labels, counts)
                }
            )
            # count the number of correct predictions for each class
            correct_counts = torch.bincount(occ_labels[correct].long())
            correct_per_class.update(
                {
                    label.item(): correct_per_class[label.item()]
                    + correct_counts[label.long()].item()
                    for label in unique_labels
                    if label < len(correct_counts)
                }
            )

            correct_predictions += correct.sum().item()
            total_predictions += len(occ_labels)
            # compute running mean of micro accuracy
            accuracy = correct_predictions / (
                total_predictions + 1e-10
            )  # prevent division by zero

            # compute macro accuracy of non-zero classes
            accs = 0
            num_non_zero_classes = 0
            for label in dataset.label_mapping:
                if total_per_class[label] == 0:
                    accs += 0
                    num_non_zero_classes += 0
                else:
                    accs += correct_per_class[label] / total_per_class[label]
                    num_non_zero_classes += 1
            # compute running mean of macro accuracy
            macro_acc = accs / num_non_zero_classes
            cover_rate = num_measured_points / num_total_points
            description = f"Micro Acc: {accuracy * 100:.2f}% | Macro Acc: {macro_acc * 100:.2f}% | Cover Rate: {cover_rate * 100:.2f}%"
            pbar.set_description(description)
    final_accuracy = correct_predictions / total_predictions
    final_macro_accuracy = accs / num_non_zero_classes
    final_per_class_accuracy = {
        class_name: correct_per_class[class_id] / (total_per_class[class_id] + 1e-10)
        for class_id, class_name in dataset.label_mapping.items()
    }
    # the number of points from Occ3D after filtering / the number of points from Occ3D before filtering
    final_cover_rate = num_measured_points / num_total_points
    return {
        "micro_accuracy": final_accuracy,
        "macro_accuracy": final_macro_accuracy,
        "per_class_accuracy": final_per_class_accuracy,
        "cover_rate": final_cover_rate,  # it's usually 90%+
        "num_measured_points": num_measured_points,
        "num_total_points": num_total_points,
    }
