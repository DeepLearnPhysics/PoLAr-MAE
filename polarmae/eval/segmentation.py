from typing import Dict, List

import torch


def compute_shape_ious(
        log_probabilities: torch.Tensor,
        labels: torch.Tensor, lengths: torch.Tensor,
        category_to_seg_classes: Dict[str, int],
        seg_class_to_category: Dict[int, str]
    ) -> Dict[str, List[torch.Tensor]]:
        # log_probablities: (B, N, N_cls) \in -inf..<0
        # labels:       (B, N) \in 0..<N_cls
        # returns           { cat: (S, P) }

        shape_ious: Dict[str, List[torch.Tensor]] = {
            cat: [] for cat in category_to_seg_classes.keys()
        }

        for i in range(log_probabilities.shape[0]):
            if lengths[i] == 0:
                continue
            curr_logprobs = log_probabilities[i][: lengths[i]]
            curr_seg_labels = labels[i][: lengths[i]]
            # Since the batch only contains one class, get the correct prediction directly
            seg_preds = torch.argmax(curr_logprobs, dim=1)  # (N,)

            # Initialize IoU for the single class
            seg_class_iou = torch.empty(len(category_to_seg_classes.keys()))
            for c in seg_class_to_category.keys():
                if ((curr_seg_labels[i] == c).sum() == 0) and (
                    (seg_preds == c).sum() == 0
                ):  # part is not present, no prediction as well
                    seg_class_iou[c] = 1.0
                else:
                    intersection = ((curr_seg_labels[i] == c) & (seg_preds == c)).sum()
                    union = ((curr_seg_labels[i] == c) | (seg_preds == c)).sum()
                    seg_class_iou[c] = intersection / union
                shape_ious[seg_class_to_category[c]].append(seg_class_iou[c])
        return shape_ious