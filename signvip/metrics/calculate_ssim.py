import cv2
import numpy as np
import torch
from tqdm import tqdm


def ssim_map(img1, img2):
    C1 = 0.01**2
    C2 = 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map


def ssim(img1, img2):
    return ssim_map(img1, img2).mean()


def calculate_hand_nonhand_ssim(images1, images2, hand_masks):
    """Return pixel-weighted hand/non-hand means from the same RGB SSIM map.

    Images are [N,C,H,W] tensors in [0,1]. Masks are [N,1,H,W] or [N,H,W].
    The five-pixel valid border matches the existing 11x11 Gaussian SSIM.
    """
    if images1.shape != images2.shape:
        raise ValueError("SSIM image tensors must have identical shapes")
    if hand_masks.ndim == 4:
        hand_masks = hand_masks[:, 0]
    if hand_masks.ndim != 3 or hand_masks.shape[0] != images1.shape[0]:
        raise ValueError("Expected hand masks [N,1,H,W] or [N,H,W]")
    hand_sum = 0.0
    hand_weight = 0.0
    nonhand_sum = 0.0
    nonhand_weight = 0.0
    for index in range(images1.shape[0]):
        first = images1[index].detach().float().cpu().numpy()
        second = images2[index].detach().float().cpu().numpy()
        channel_maps = [
            ssim_map(first[channel], second[channel])
            for channel in range(first.shape[0])
        ]
        rgb_map = np.mean(channel_maps, axis=0)
        mask = hand_masks[index].detach().float().cpu().numpy()[5:-5, 5:-5]
        mask = np.clip(mask, 0.0, 1.0)
        inverse = 1.0 - mask
        hand_sum += float((rgb_map * mask).sum())
        hand_weight += float(mask.sum())
        nonhand_sum += float((rgb_map * inverse).sum())
        nonhand_weight += float(inverse.sum())
    hand_value = hand_sum / hand_weight if hand_weight > 0 else float("nan")
    nonhand_value = (
        nonhand_sum / nonhand_weight if nonhand_weight > 0 else float("nan")
    )
    return hand_value, nonhand_value


def calculate_ssim_function(img1, img2):
    # [0,1]
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def trans(x):
    return x


def calculate_ssim(videos1, videos2, only_final=False):
    print("calculate_ssim...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    ssim_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].cpu().numpy()
            img2 = video2[clip_timestamp].cpu().numpy()

            # calculate ssim of a video
            ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))

        ssim_results.append(ssim_results_of_a_video)

    ssim_results = np.array(ssim_results)

    ssim = []
    ssim_std = []

    if only_final:

        ssim.append(np.mean(ssim_results))
        ssim_std.append(np.std(ssim_results))

    else:

        for clip_timestamp in range(len(video1)):
            ssim.append(np.mean(ssim_results[:, clip_timestamp]))
            ssim_std.append(np.std(ssim_results[:, clip_timestamp]))

    result = {
        "value": ssim,
        "value_std": ssim_std,
    }

    return result["value"][0]


# test code / using example


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    )
    videos2 = torch.ones(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    )

    result = calculate_ssim(videos1, videos2)
    print("[ssim avg]", result["value"])
    print("[ssim std]", result["value_std"])


if __name__ == "__main__":
    main()
