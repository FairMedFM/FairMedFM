import os
from copy import deepcopy
from typing import Optional, Tuple

import ipdb
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image
from tutils.trainer import LearnerModule, Trainer


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class SamLearner(LearnerModule):
    def __init__(
        self,
        sam_model,
        config=None,
        logger=None,
        data_engine=None,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.config = config
        self.logger = logger
        self.model = sam_model
        self.net = self.model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.data_engine = data_engine
        self.features = None

    @torch.no_grad()
    def encode(self, data):
        img = data['img']
        assert img.shape[1:] == (3, 1024, 1024) or img.shape[1:] == (
            3, 256, 256), f"{__file__} Got{img.shape}"
        self.set_torch_image(img, img.shape[2:])
        return self.features

    @torch.no_grad()
    def decode(self, data, batch_idx=None, flag="train", **kwargs):
        features = data['features']
        # shape: (b, 2) or (b, 2, n_points)
        prompt_point = data['prompt_point']
        batch_size = prompt_point.shape[0]  # TODO: Modify 1 to n_points
        # ipdb.set_trace()
        if prompt_point.dim() == 2:
            prompt_point = rearrange(prompt_point, "b c -> b 1 c")
        elif prompt_point.dim() == 3:
            prompt_point = rearrange(prompt_point, "b c n -> b n c")

        point_label = torch.ones((batch_size, prompt_point.shape[1]))
        prompt_box = data['prompt_box']
        prompt_box = rearrange(prompt_box, "b c -> b 1 c")
        # assert prompt_point.shape[1:] == (
        #     1, 2), f"{__file__} Got{prompt_point.shape}"
        # assert point_label.shape[1:] == (
        #     1,), f"{__file__} Got{point_label.shape}"
        # assert prompt_box.shape[1:] == (
        #     1, 4), f"{__file__} Got{prompt_box.shape}"

        # Select the prompt type outside instead of inside this function.
        if flag == "point":
            pred_masks, iou_predictions, _ = self.predict_torch(
                features=features,
                point_coords=prompt_point,
                point_labels=point_label,
                boxes=None,
                multimask_output=True,
                return_logits=True,
            )
        elif flag == "bbox":
            pred_masks, iou_predictions, _ = self.predict_torch(
                features=features,
                point_coords=None,
                point_labels=None,
                boxes=prompt_box,
                multimask_output=True,
                return_logits=True,
            )

        return pred_masks, iou_predictions

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(
            2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = transformed_image
        # input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(
                point_coords, self.original_size)
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None,
                                                      :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(
                box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # masks = masks[0].detach().cpu().numpy()
        # iou_predictions = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks = low_res_masks[0].detach().cpu().numpy()
        # return masks, iou_predictions, low_res_masks
        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def predict_torch(
        self,
        features,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ipdb.set_trace()
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self._get_prompt_embedding(
            points, boxes, mask_input)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size)
        # import ipdb; ipdb.set_trace()

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def _get_prompt_embedding(self, points, boxes, mask_input):
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        return sparse_embeddings, dense_embeddings

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
