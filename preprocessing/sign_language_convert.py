import argparse
import json
import time
import cv2
import glob
import imageio
import numpy as np
import os
import re
import sys
import threading

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
clip_dir = os.path.join(current_dir, "CLIP-main")
if clip_dir not in sys.path:
    sys.path.insert(0, clip_dir)
import argparse


sys.path.append("./DWPose/ControlNet-v1-1-nightly")

from annotator.openpose import OpenposeDetector
from annotator.dwpose import DWposeDetector
from annotator.util import resize_image, HWC3
from dwpose_keypoints import (
    infer_keypoints_video,
    load_keypoint_cache,
    render_pose_video,
    save_keypoint_cache,
)
import os
import sys
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from PIL import Image
from tensorflow.python.platform import gfile
from tqdm import tqdm
import csv
try:
    from pytorch_transformers import BertModel, BertConfig, BertTokenizer
except ModuleNotFoundError:
    from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
from PIL import Image
import torch
try:
    import open_clip
except ModuleNotFoundError:
    open_clip = None
try:
    import clip as clip_lib
except ModuleNotFoundError:
    clip_lib = None
from PIL import Image
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

from datasets.h5 import HDF5Maker
try:
    from SignGen.fdv_call import compute_framewise_fvd_motion, preload_fvd_model
except ImportError:
    from fdv_call import compute_framewise_fvd_motion, preload_fvd_model

_DWPOSE_DETECTOR = None
_DEPTH_ESTIMATOR = None
_FEATURE_MODELS_INITIALIZED = False
_TORCH_RUNTIME_CONFIGURED = False
_DWPOSE_INIT_LOCK = threading.Lock()
_DEPTH_INIT_LOCK = threading.Lock()
_CUDA_INFERENCE_LOCK = threading.Lock()


def _read_bool_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_VERBOSE = _read_bool_env("SIGNGEN_VERBOSE", default=False)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def _read_int_env(name, default=None):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _log(*parts):
    if _VERBOSE:
        print(*parts)


def _sample_id(path):
    basename = os.path.basename(os.path.normpath(path))
    stem, extension = os.path.splitext(basename)
    if extension.lower() in _VIDEO_EXTENSIONS:
        return stem
    return basename


def _cache_key(path):
    sample_id = _sample_id(path)
    return sample_id.replace(os.sep, "_")


def _natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", os.path.basename(str(path)))
    ]


def _load_npy_cache_or_delete(path, label):
    try:
        return np.load(path)
    except Exception as exc:
        print(f"WARNING: failed to load {label} cache '{path}', deleting and regenerating: {exc}")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as remove_exc:
            print(f"WARNING: failed to delete broken {label} cache '{path}': {remove_exc}")
        return None


def _is_video_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _VIDEO_EXTENSIONS


def _is_image_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS


def _resolve_h5_compression(value):
    if value is None:
        return "lzf"

    normalized = str(value).strip().lower()
    if normalized in {"", "none", "false", "off", "0", "null"}:
        return None
    if normalized == "lzf":
        return "lzf"
    raise ValueError(f"Unsupported HDF5 compression mode: {value}")


def _cuda_guard(required):
    if required and torch.cuda.is_available():
        return _CUDA_INFERENCE_LOCK
    return nullcontext()


def _configure_torch_runtime():
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return

    torch_num_threads = _read_int_env("SIGNGEN_TORCH_NUM_THREADS")
    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    opencv_num_threads = _read_int_env("SIGNGEN_OPENCV_NUM_THREADS")
    if opencv_num_threads is not None and opencv_num_threads >= 0:
        cv2.setNumThreads(opencv_num_threads)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    _TORCH_RUNTIME_CONFIGURED = True

    print(
        "runtime threads: "
        f"torch={torch.get_num_threads()}, "
        f"opencv={cv2.getNumThreads() if hasattr(cv2, 'getNumThreads') else 'unknown'}"
    )


def _describe_torch_device(device):
    if device is None:
        return "unknown"

    device = torch.device(device)
    if device.type != "cuda":
        return "cpu"

    try:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        return f"gpu ({device}, {gpu_name})"
    except Exception:
        return f"gpu ({device})"


def _get_dwpose_detector():
    global _DWPOSE_DETECTOR
    if _DWPOSE_DETECTOR is not None:
        return _DWPOSE_DETECTOR

    with _DWPOSE_INIT_LOCK:
        if _DWPOSE_DETECTOR is None:
            try:
                _DWPOSE_DETECTOR = DWposeDetector()
            except Exception as exc:
                raise RuntimeError("Failed to initialize DWPose detector.") from exc
    return _DWPOSE_DETECTOR


def _get_depth_estimator():
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is not None:
        return _DEPTH_ESTIMATOR

    with _DEPTH_INIT_LOCK:
        if _DEPTH_ESTIMATOR is not None:
            return _DEPTH_ESTIMATOR

        _configure_torch_runtime()

        dpt_dir = os.path.join(current_dir, "DPT")
        if dpt_dir not in sys.path:
            sys.path.append(dpt_dir)

        import run_monodepth

        for model_type in ("dpt_hybrid", "dpt_large"):
            try:
                model_path = run_monodepth.resolve_model_path(model_type)
                _DEPTH_ESTIMATOR = run_monodepth.DepthEstimator(
                    model_path=model_path,
                    model_type=model_type,
                    optimize=True,
                )
                device_info = _describe_torch_device(
                    getattr(_DEPTH_ESTIMATOR, "device", None)
                )
                print(
                    f"initialized depth estimator: {model_type} "
                    f"({model_path}) on {device_info}"
                )
                return _DEPTH_ESTIMATOR
            except FileNotFoundError:
                continue

    raise FileNotFoundError(
        "No DPT model weights were found under SignGen/DPT/weights. "
        "Add a supported checkpoint such as dpt_large or dpt_hybrid."
    )


def initialize_feature_models():
    global _FEATURE_MODELS_INITIALIZED
    if _FEATURE_MODELS_INITIALIZED:
        return

    _configure_torch_runtime()
    preload_fvd_model()
    _get_depth_estimator()
    _FEATURE_MODELS_INITIALIZED = True
    print("feature models ready")


def _dwpose_uses_cuda():
    detector = _get_dwpose_detector()
    pose_estimation = getattr(detector, "pose_estimation", None)
    return bool(getattr(pose_estimation, "using_cuda", False))


def _generate_dwpose_video(frames):
    detector = _get_dwpose_detector()
    pose_frames = []
    for frame in frames:
        frame = HWC3(frame.astype(np.uint8, copy=False))
        pose_input = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pose_frame = detector(pose_input)
        pose_frames.append(HWC3(pose_frame.astype(np.uint8, copy=False)))
    return np.stack(pose_frames)


def textToBert_dim96(text):
    tokenizer = BertTokenizer.from_pretrained(r'/datasets/bertbasecased/vocab.txt')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    # print(tokens)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # print(tokens)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # print(ids.shape)

    model = BertModel.from_pretrained(r'datasets/bertbasecased/')
    # pooled 可以暂时理解为最后一层每一个句子第一个单词[cls]的结果
    outputs = model(ids)
    pooled = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
    # print( pooled)
    # print(pooled.shape)
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96),nn.Tanh())
    output = (net(pooled)-0.25)*0.1
    return output

def textToBert_dim96_csd(text):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # print(tokens)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    model = BertModel.from_pretrained('bert-base-cased')
    # pooled 可以暂时理解为最后一层每一个句子第一个单词[cls]的结果
    outputs = model(ids)
    pooled = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96),nn.Tanh())
    output = (net(pooled)-0.25)*0.1
    return output
def clip_text_encoder(text, model, device, tokenizer=None, normalize=False):
    if tokenizer is None:
        if clip_lib is None:
            raise ModuleNotFoundError(
                "OpenAI CLIP is not available. Install CLIP or use OpenCLIP."
            )
        text = clip_lib.tokenize(text).to(device=device)
    else:
        text = tokenizer([text]).to(device=device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)

    return text_features


def clip_text_encoder_batch(texts, model, device, tokenizer=None, normalize=False):
    if tokenizer is None:
        if clip_lib is None:
            raise ModuleNotFoundError(
                "OpenAI CLIP is not available. Install CLIP or use OpenCLIP."
            )
        text_tokens = clip_lib.tokenize(texts).to(device=device)
    else:
        text_tokens = tokenizer(texts).to(device=device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)

    return text_features


def load_text_encoder(
    device,
    backend="open_clip",
    model_name="ViT-H-14",
    pretrained="laion2b_s32b_b79k",
):
    backend = str(backend).strip().lower()
    if backend == "open_clip":
        if open_clip is None:
            raise ModuleNotFoundError(
                "open_clip is required for ViT-H/14 text embeddings. "
                "Install it with: pip install open_clip_torch"
            )
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer

    if backend == "clip":
        if clip_lib is None:
            raise ModuleNotFoundError(
                "OpenAI CLIP is not available. Install CLIP or use OpenCLIP."
            )
        model, _ = clip_lib.load(model_name, device=device)
        model.eval()
        return model, None

    raise ValueError(f"Unsupported text encoder backend: {backend}")

class SignLanguage_HDF5Maker(HDF5Maker):

    def __init__(self, *args, compression="lzf", pipeline_metadata=None, **kwargs):
        self.compression = _resolve_h5_compression(compression)
        self.pipeline_metadata = dict(pipeline_metadata or {})
        super().__init__(*args, **kwargs)

    def create_video_groups(self):
        if self.pipeline_metadata:
            self.writer.attrs['signgen_pipeline_metadata_json'] = json.dumps(
                self.pipeline_metadata, sort_keys=True)
        self.writer.create_group('video')
        self.writer.create_group('len')
        self.writer.create_group('text')
        self.writer.create_group('text_key')
        self.writer.create_group('pose_video')
        self.writer.create_group('motion')
        self.writer.create_group('depth')
        self.writer['video'].attrs['layout'] = 'THWC'
        self.writer['pose_video'].attrs['layout'] = 'THWC'
        self.writer['depth'].attrs['layout'] = 'THW'
        self.writer['motion'].attrs['layout'] = 'TCHW'
        self.writer['text'].attrs['layout'] = '1C'

    def add_video_data_all(self, data, deepth_data, motion, text, text_key, depth):
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['text'].create_dataset(str(self.count), data=text,)
        self.writer['text_key'].create_dataset(str(self.count), data=text_key,)
        self.writer['video'].create_dataset(
            str(self.count),
            data=np.ascontiguousarray(data),
            compression=self.compression,
        )
        self.writer['pose_video'].create_dataset(
            str(self.count),
            data=np.ascontiguousarray(deepth_data),
            compression=self.compression,
        )
        self.writer['motion'].create_dataset(
            str(self.count),
            data=np.ascontiguousarray(motion),
            compression=self.compression,
        )
        self.writer['depth'].create_dataset(
            str(self.count),
            data=np.ascontiguousarray(depth),
            compression=self.compression,
        )

        self.count += 1

        if self.count == self.num_per_shard:
            self.create_new_shard()


def center_crop(image):
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape={image.shape}")
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side, :]


def read_video(video_file, image_size, center_crop_before_resize=False):
    _log('Reading video frames from', video_file)
    new_h = int(image_size)
    new_w = int(image_size)

    if os.path.isdir(video_file):
        filenames = sorted(
            [
                path
                for path in gfile.Glob(os.path.join('%s' % (video_file), '*'))
                if os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS
            ],
            key=_natural_sort_key,
        )
        if len(filenames) == 0:
            raise FileNotFoundError(f"No image frames found under '{video_file}'")
        _log('frame count:', len(filenames))
        frames = np.empty((len(filenames), new_h, new_w, 3), dtype=np.uint8)
        for idx, path in enumerate(filenames):
            with Image.open(path) as pil_im:
                arr = np.asarray(pil_im.convert("RGB"), dtype=np.uint8)
                if center_crop_before_resize:
                    arr = center_crop(arr)
                pil_im_rsz = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
                frames[idx] = np.asarray(pil_im_rsz, dtype=np.uint8)
        return frames

    if not _is_video_file(video_file):
        raise FileNotFoundError(f"Input sample is neither a frame directory nor a supported video file: '{video_file}'")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file '{video_file}'")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if center_crop_before_resize:
                frame = center_crop(frame)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.uint8, copy=False))
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be decoded from video file '{video_file}'")

    _log('frame count:', len(frames))
    frames = np.stack(frames)

    return frames

def _resize_pose_video(pose_video, image_size):
    if pose_video.shape[1:3] == (image_size, image_size):
        return pose_video
    resized = [
        cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        for frame in pose_video
    ]
    return np.stack(resized).astype(np.uint8, copy=False)


def read_video_handpose(video_file, image_size, frames=None, center_crop_before_resize=False):
    split = str(args.split)
    keypoint_cache_root = os.environ.get('SIGNGEN_POSE_KEYPOINT_CACHE_ROOT')
    if keypoint_cache_root:
        if os.path.isabs(keypoint_cache_root):
            keypoint_parent_dir = os.path.join(keypoint_cache_root, split)
        else:
            keypoint_parent_dir = os.path.join(current_dir, keypoint_cache_root, split)
        filename = _cache_key(video_file)
        dwpose_infer_size = _read_int_env('SIGNGEN_DWPOSE_INFER_SIZE', image_size)
        dwpose_infer_size = max(1, int(dwpose_infer_size))
        keypoint_suffix = os.environ.get(
            'SIGNGEN_POSE_KEYPOINT_CACHE_SUFFIX',
            f'_dwpose{dwpose_infer_size}_keypoints_v1.npz',
        )
        if not keypoint_suffix.endswith('.npz'):
            raise ValueError("SIGNGEN_POSE_KEYPOINT_CACHE_SUFFIX must end with '.npz'.")
        keypoint_path = os.path.join(
            keypoint_parent_dir, filename + keypoint_suffix)

        keypoint_cache = None
        if os.path.exists(keypoint_path):
            try:
                keypoint_cache = load_keypoint_cache(keypoint_path)
                print(
                    f"  [cache] keypoints {os.path.basename(video_file)} "
                    f"({keypoint_suffix})"
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to load DWPose keypoint cache "
                    f"'{keypoint_path}', regenerating: {exc}"
                )

        if keypoint_cache is None:
            pose_frames = read_video(
                video_file,
                dwpose_infer_size,
                center_crop_before_resize=center_crop_before_resize,
            )
            with _cuda_guard(_dwpose_uses_cuda()):
                keypoint_cache = infer_keypoints_video(
                    _get_dwpose_detector(), pose_frames)
            save_keypoint_cache(keypoint_path, keypoint_cache)
            _log('saved DWPose keypoint cache:', keypoint_path)

        score_threshold = float(
            os.environ.get('SIGNGEN_DWPOSE_SCORE_THRESHOLD', '0.3'))
        return render_pose_video(
            keypoint_cache,
            height=image_size,
            width=image_size,
            score_threshold=score_threshold,
        )

    pose_cache_root = os.environ.get('SIGNGEN_POSE_CACHE_ROOT', 'cache_pose')
    if os.path.isabs(pose_cache_root):
        parent_dir = os.path.join(pose_cache_root, split)
    else:
        parent_dir = os.path.join(current_dir, pose_cache_root, split)
    filename = _cache_key(video_file)
    dwpose_infer_size = _read_int_env('SIGNGEN_DWPOSE_INFER_SIZE', image_size)
    dwpose_infer_size = max(1, int(dwpose_infer_size))
    pose_cache_suffix = os.environ.get('SIGNGEN_POSE_CACHE_SUFFIX')
    if pose_cache_suffix is None:
        pose_cache_suffix = (
            '.npy'
            if dwpose_infer_size == image_size
            else f'_dwpose{dwpose_infer_size}_to{image_size}.npy'
        )
    if not pose_cache_suffix.endswith('.npy'):
        raise ValueError("SIGNGEN_POSE_CACHE_SUFFIX must end with '.npy'.")
    path = os.path.join(parent_dir, filename + pose_cache_suffix)
    if os.path.exists(path):
        data = _load_npy_cache_or_delete(path, 'pose')
        if data is not None:
            print(f"  [cache] pose   {os.path.basename(video_file)} ({pose_cache_suffix})")
            return data

    _log('missing pose cache, generating with DWPose:', path)
    if dwpose_infer_size == image_size:
        if frames is None:
            frames = read_video(
                video_file,
                image_size,
                center_crop_before_resize=center_crop_before_resize,
            )
        pose_frames = frames
    else:
        pose_frames = read_video(
            video_file,
            dwpose_infer_size,
            center_crop_before_resize=center_crop_before_resize,
        )
    with _cuda_guard(_dwpose_uses_cuda()):
        data = _generate_dwpose_video(pose_frames)
    data = _resize_pose_video(data, image_size)
    os.makedirs(parent_dir, exist_ok=True)
    np.save(path, data)
    _log('saved pose cache:', path, data.shape)
    return data

def read_video_motion(video_file, image_size=None, frames=None):
    split = str(args.split)
    motion_cache_root = os.environ.get('SIGNGEN_MOTION_CACHE_ROOT', 'cache_motion')
    if os.path.isabs(motion_cache_root):
        parent_dir = os.path.join(motion_cache_root, split)
    else:
        parent_dir = os.path.join(current_dir, motion_cache_root, split)
    filename = _cache_key(video_file)
    motion_cache_suffix = os.environ.get('SIGNGEN_MOTION_CACHE_SUFFIX', '_fvd.npy')
    if not motion_cache_suffix.endswith('.npy'):
        raise ValueError("SIGNGEN_MOTION_CACHE_SUFFIX must end with '.npy'.")
    fvd_path = os.path.join(parent_dir, filename + motion_cache_suffix)

    if os.path.exists(fvd_path):
        data = _load_npy_cache_or_delete(fvd_path, 'FVD motion')
        if data is not None:
            print(f"  [cache] motion {os.path.basename(video_file)} ({motion_cache_suffix})")
            return data.astype(np.float32, copy=False)

    if motion_cache_suffix != '_fvd.npy':
        raise FileNotFoundError(
            f"Required custom motion cache is missing: {fvd_path}. "
            "Custom Fig. 3/I3D features must be precomputed; refusing to silently "
            "replace them with the legacy FVD proxy."
        )

    _log('missing motion cache, generating with FVD:', fvd_path)
    if frames is None:
        if image_size is None:
            raise ValueError("image_size is required when frames is None.")
        frames = read_video(video_file, image_size)

    _configure_torch_runtime()
    with _cuda_guard(required=True):
        data = compute_framewise_fvd_motion(frames)
    os.makedirs(parent_dir, exist_ok=True)
    np.save(fvd_path, data.astype(np.float32, copy=False))
    _log('saved FVD motion cache:', fvd_path, data.shape)
    # # f c h w
    return data

def read_video_depth(video_file, image_size=None, frames=None):
    split = str(args.split)
    depth_cache_root = os.environ.get('SIGNGEN_DEPTH_CACHE_ROOT', 'cache_depth')
    if os.path.isabs(depth_cache_root):
        parent_dir = os.path.join(depth_cache_root, split)
    else:
        parent_dir = os.path.join(current_dir, depth_cache_root, split)
    filename = _cache_key(video_file)
    path = os.path.join(parent_dir, filename + '.npy')
    if os.path.exists(path):
        data = _load_npy_cache_or_delete(path, 'depth')
        if data is not None:
            print(f"  [cache] depth  {os.path.basename(video_file)}")
            return data.astype(np.float32, copy=False)

    _log('missing depth cache, generating with DPT:', path)
    if frames is None:
        if image_size is None:
            raise ValueError("image_size is required when frames is None.")
        frames = read_video(video_file, image_size)

    _configure_torch_runtime()

    estimator = _get_depth_estimator()
    with _cuda_guard(required=True):
        data = estimator.predict_many(frames)
    os.makedirs(parent_dir, exist_ok=True)
    np.save(path, data.astype(np.float32, copy=False))
    _log('saved depth cache:', path, data.shape)
    return data

def read_pictures(video_file, image_size=128):
    frames = []
    # reader = imageio.get_reader(video_file)
    filenames = gfile.Glob(os.path.join('%s' % (video_file), '*'))

    fi = gfile.Glob(os.path.join('path' , '*'))
    h, w = image_size, image_size
    new_h = image_size
    new_w = int(new_h / h * w)
    for path in filenames:
        # img_cc = center_crop(img)
        pil_im = Image.open(path)
        # pil_im = Image.fromarray(img)
        pil_im_rsz = pil_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(np.array(pil_im_rsz))
        # frames.append(np.array(img))
    return np.stack(frames)

def process_pictures(video_file, image_size):
    frames = []
    print('Reading video frames from', video_file)
    try:
        frames = read_pictures(video_file, image_size)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        return "break"
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    return frames

def process_video(video_file, image_size, center_crop_before_resize=False):
    print("video_file: ", video_file)
    start_time = time.time()
    step_time = start_time
    _configure_torch_runtime()
    print(f"  runtime config: {time.time() - step_time:.2f}s")

    step_time = time.time()
    frames = np.ascontiguousarray(
        read_video(
            video_file,
            image_size,
            center_crop_before_resize=center_crop_before_resize,
        ),
        dtype=np.uint8,
    )
    print(f"  read video: {time.time() - step_time:.2f}s, shape={frames.shape}")

    step_time = time.time()
    handpose_video = read_video_handpose(
        video_file,
        image_size,
        frames=frames,
        center_crop_before_resize=center_crop_before_resize,
    )
    print(f"  handpose: {time.time() - step_time:.2f}s, shape={handpose_video.shape}")

    step_time = time.time()
    motion = read_video_motion(video_file, image_size=image_size, frames=frames)
    print(f"  motion: {time.time() - step_time:.2f}s, shape={motion.shape}")

    step_time = time.time()
    depth = read_video_depth(video_file, image_size, frames=frames)
    print(f"  depth: {time.time() - step_time:.2f}s, shape={depth.shape}")
    print(f"processed video in {time.time() - start_time:.2f} seconds")
    return frames, handpose_video, motion, depth


def validate_processed_sample(video, pose_video, motion, depth, image_size, sample_id):
    expected_frames = len(video)
    expected_hw = (int(image_size), int(image_size))
    shapes = {
        'video': tuple(video.shape),
        'pose_video': tuple(pose_video.shape),
        'motion': tuple(motion.shape),
        'depth': tuple(depth.shape),
    }
    lengths = {
        'video': len(video),
        'pose_video': len(pose_video),
        'motion': len(motion),
        'depth': len(depth),
    }
    if expected_frames <= 0 or any(length != expected_frames for length in lengths.values()):
        raise ValueError(
            f"Sample {sample_id!r} has unaligned modality lengths: {lengths}"
        )
    if tuple(video.shape[1:]) != (*expected_hw, 3):
        raise ValueError(f"Sample {sample_id!r} has invalid RGB shape: {shapes['video']}")
    if tuple(pose_video.shape[1:]) != (*expected_hw, 3):
        raise ValueError(
            f"Sample {sample_id!r} has invalid pose shape: {shapes['pose_video']}"
        )
    if tuple(depth.shape[1:]) != expected_hw:
        raise ValueError(f"Sample {sample_id!r} has invalid depth shape: {shapes['depth']}")
    if motion.ndim != 4 or tuple(motion.shape[-2:]) != expected_hw:
        raise ValueError(f"Sample {sample_id!r} has invalid motion shape: {shapes['motion']}")
    for name, array in (
        ('video', video),
        ('pose_video', pose_video),
        ('motion', motion),
        ('depth', depth),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"Sample {sample_id!r} has non-finite values in {name}")


def iter_processed_videos(filenames, image_size, num_workers=1, center_crop_before_resize=False):
    num_workers = max(1, int(num_workers))
    if num_workers == 1:
        for filename in filenames:
            yield filename, process_video(
                filename,
                image_size,
                center_crop_before_resize=center_crop_before_resize,
            )
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending = deque()
        filename_iter = iter(filenames)

        for _ in range(num_workers):
            try:
                filename = next(filename_iter)
            except StopIteration:
                break
            pending.append((
                filename,
                executor.submit(
                    process_video,
                    filename,
                    image_size,
                    center_crop_before_resize=center_crop_before_resize,
                ),
            ))

        while pending:
            filename, future = pending.popleft()
            yield filename, future.result()

            try:
                next_filename = next(filename_iter)
            except StopIteration:
                continue

            pending.append(
                (
                    next_filename,
                    executor.submit(
                        process_video,
                        next_filename,
                        image_size,
                        center_crop_before_resize=center_crop_before_resize,
                    ),
                )
            )


def find_sample_path(input_dir, sample_name):
    frame_dir = os.path.join(input_dir, sample_name)
    if os.path.isdir(frame_dir):
        return frame_dir

    for extension in sorted(_VIDEO_EXTENSIONS):
        video_path = os.path.join(input_dir, sample_name + extension)
        if os.path.isfile(video_path):
            return video_path

    matches = [
        path for path in gfile.Glob(os.path.join(input_dir, sample_name + ".*"))
        if _is_video_file(path)
    ]
    if matches:
        return sorted(matches)[0]

    return None


def _sample_name_variants(name):
    basename = os.path.basename(os.path.normpath(str(name)))
    stem, extension = os.path.splitext(basename)
    variants = {str(name), basename}
    if extension.lower() in _VIDEO_EXTENSIONS:
        variants.add(stem)
    return variants


def index_sample_paths(input_dir):
    sample_paths = {}

    for path in sorted(gfile.Glob(os.path.join(input_dir, "*"))):
        if os.path.isdir(path):
            image_files = [
                frame_path
                for frame_path in gfile.Glob(os.path.join(path, "*"))
                if os.path.splitext(frame_path)[1].lower() in _IMAGE_EXTENSIONS
            ]
            if not image_files:
                continue
        elif not _is_video_file(path):
            continue

        for key in _sample_name_variants(path):
            sample_paths.setdefault(key, path)

    return sample_paths


def resolve_split_dir(sl_dir, split):
    sl_dir = os.path.abspath(os.path.expanduser(sl_dir))
    split = str(split).strip() if split else ""

    split_dir = os.path.join(sl_dir, split) if split else None
    if split_dir and os.path.isdir(split_dir):
        return split_dir
    if os.path.isdir(sl_dir):
        return sl_dir

    raise FileNotFoundError(
        f"Could not resolve an input directory from sl_dir='{sl_dir}' and split='{split}'."
    )

#phoenix-2014/2014T
def make_text_dic(
    dir,
    text_encoder_backend="open_clip",
    clip_model="ViT-H-14",
    clip_pretrained="laion2b_s32b_b79k",
    normalize_clip_text=False,
    text_batch_size=128,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dictionary = {}
    ori_dictionary = {}
    print("dir: ",dir)
    model_clip, tokenizer = load_text_encoder(
        device=device,
        backend=text_encoder_backend,
        model_name=clip_model,
        pretrained=clip_pretrained,
    )
    print(
        "text encoder:",
        text_encoder_backend,
        clip_model,
        clip_pretrained if text_encoder_backend == "open_clip" else "",
    )
    cleaned_records = []
    with open(dir, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        required_columns = {"name", "orth", "translation"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Text annotation file '{dir}' is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            key = row["name"].strip()
            original_text = row["translation"].strip()
            value = original_text
            if not key or not value:
                raise ValueError(f"Empty name/translation in text annotation row: {row}")
            cleaned_records.append((key, original_text, value))

    unique_prompts = len({value for _, _, value in cleaned_records})
    if unique_prompts < max(2, len(cleaned_records) // 100):
        raise ValueError(
            f"Only {unique_prompts} unique prompts were parsed from {len(cleaned_records)} rows. "
            "Check the annotation delimiter and selected text column."
        )

    text_batch_size = max(1, int(text_batch_size))
    for start in tqdm(range(0, len(cleaned_records), text_batch_size), desc="encoding text"):
        batch = cleaned_records[start:start + text_batch_size]
        values = [item[2] for item in batch]
        clip = clip_text_encoder_batch(
            values,
            model_clip,
            device,
            tokenizer=tokenizer,
            normalize=normalize_clip_text,
        )
        clip_info = clip.detach().cpu().numpy()
        for (key, ori_value, _), embedding in zip(batch, clip_info):
            ori_dictionary[key] = ori_value
            dictionary[key] = embedding[None, :]
    return dictionary, ori_dictionary

#CS-daily
def make_csd_text_dic(dir):
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    dictionary = {}
    ori_dictionary = {}
    with open(dir, 'rb') as f:
            data = pickle.load(f)
            lens = len(data['info'])
            print(lens)

            for i in tqdm(range(lens)):
                key = data['info'][i]['name']
                value = ''.join(data['info'][i]['label_char'])
                bert = textToBert_dim96_csd(value).to(device=device).detach()
                bert_info = bert.cpu().numpy()
                ori_dictionary[key] = value
                dictionary[key] = bert_info


    return dictionary,ori_dictionary


_MANIFEST_FILE = 'done_samples.txt'


def _read_manifest(out_dir):
    """Đọc manifest — nguồn resume ưu tiên 1, nhanh và không phụ thuộc H5 parse."""
    path = os.path.join(out_dir, _MANIFEST_FILE)
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        keys = {line.strip() for line in f if line.strip()}
    print(f"[MANIFEST] {os.path.basename(out_dir)}: đọc {len(keys)} mẫu đã xong từ {_MANIFEST_FILE}")
    if keys:
        sample = list(keys)[:2]
        print(f"  ví dụ: {sample}")
    return keys


def _append_manifest(out_dir, sample_id):
    """Append một sample_id vào manifest sau khi ghi H5 thành công."""
    path = os.path.join(out_dir, _MANIFEST_FILE)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(sample_id + '\n')


def _scan_existing_hdf5(out_dir):
    """Quét tất cả shard HDF5 trong out_dir.
    Trả về (done_keys: set[str], max_shard_number: int).
    done_keys = tập hợp text_key của các mẫu đã được ghi.
    max_shard_number = số thứ tự shard lớn nhất hiện có (để tiếp tục từ shard kế tiếp).
    """
    import glob as _glob
    import re as _re
    import h5py as _h5py

    shard_paths = sorted(_glob.glob(os.path.join(out_dir, '*.hdf5')) +
                         _glob.glob(os.path.join(out_dir, '*.h5')))
    print(f"[SCAN] {out_dir}: tìm thấy {len(shard_paths)} shard file(s)")
    if not shard_paths:
        return set(), 0

    done_keys = set()
    max_shard_number = 0

    for path in shard_paths:
        m = _re.search(r'shard_(\d+)', os.path.basename(path))
        if m:
            max_shard_number = max(max_shard_number, int(m.group(1)))

        try:
            with _h5py.File(path, 'r') as f:
                top_keys = list(f.keys())
                if 'text_key' not in f:
                    n = len(f['len']) if 'len' in f else '?'
                    print(f"  [SCAN] {os.path.basename(path)}: KHÔNG có group 'text_key' "
                          f"(groups: {top_keys}, ~{n} mẫu) — không thể resume shard này")
                    continue
                key_group = f['text_key']
                n_before = len(done_keys)
                for idx in key_group.keys():
                    raw = key_group[idx][()]
                    if isinstance(raw, (bytes, np.bytes_)):
                        raw = raw.decode('utf-8', errors='replace')
                    elif isinstance(raw, np.ndarray):
                        raw = raw.item()
                        if isinstance(raw, bytes):
                            raw = raw.decode('utf-8', errors='replace')
                    done_keys.add(str(raw).strip())
                n_added = len(done_keys) - n_before
                sample_key = next(iter(key_group.keys()), None)
                sample_val = repr(key_group[sample_key][()]) if sample_key else 'N/A'
                print(f"  [SCAN] {os.path.basename(path)}: {n_added} text_key, "
                      f"ví dụ key[{sample_key}]={sample_val}")
        except Exception as exc:
            print(f"  [SCAN] WARNING: không đọc được '{path}': {exc}")

    print(f"[SCAN] tổng done_keys={len(done_keys)}, max_shard={max_shard_number}")
    if done_keys:
        sample = list(done_keys)[:3]
        print(f"  [SCAN] ví dụ done_keys: {[repr(k) for k in sample]}")
    return done_keys, max_shard_number


class _ResumableSignLanguageHDF5Maker(SignLanguage_HDF5Maker):
    """Tiếp tục ghi HDF5 từ shard tiếp theo mà không xóa shard đã có."""

    def __init__(
        self,
        out_dir,
        resume_from_shard,
        num_per_shard=100000,
        compression="lzf",
        pipeline_metadata=None,
    ):
        # Bypass HDF5Maker.__init__ hoàn toàn — tự setup state
        self.out_path = out_dir
        self.out_dir = out_dir
        self.num_per_shard = num_per_shard
        self.max_shards = None
        self.name = None
        self.name_fmt = 'shard_{:04d}.hdf5'
        self.force = False
        self.video = True
        self.compression = _resolve_h5_compression(compression)
        self.pipeline_metadata = dict(pipeline_metadata or {})

        self.writer = None
        self.shard_paths = []
        # shard_number sẽ được tăng lên 1 trong create_new_shard()
        # nên set về resume_from_shard để shard tiếp theo = resume_from_shard + 1
        self.shard_number = resume_from_shard

        os.makedirs(self.out_dir, exist_ok=True)
        self.create_new_shard()
        self.add_video_info()


def make_h5_from_sl(
    sl_dir,
    text_path,
    split='train',
    out_dir='./h5_ds',
    image_size=128,
    vids_per_shard=100000,
    force_h5=False,
    num_workers=1,
    h5_compression="lzf",
    text_encoder_backend="open_clip",
    clip_model="ViT-H-14",
    clip_pretrained="laion2b_s32b_b79k",
    normalize_clip_text=False,
    text_batch_size=128,
    center_crop_before_resize=False,
):
    # ── Resume: ưu tiên 1=manifest, 2=H5 scan, 3=chạy lại từ đầu ──────────────
    _done_keys = set()
    _resume_shard_number = 0
    if force_h5:
        manifest_path = os.path.join(out_dir, _MANIFEST_FILE)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print(f"[FORCE] xóa manifest cũ: {manifest_path}")
    else:
        os.makedirs(out_dir, exist_ok=True)
        # Ưu tiên 1: manifest (nhanh, đáng tin cậy)
        _done_keys = _read_manifest(out_dir)
        # Luôn quét H5 để lấy max_shard_number (cần cho _ResumableSignLanguageHDF5Maker)
        _done_keys_h5, _resume_shard_number = _scan_existing_hdf5(out_dir)
        # HDF5 is the source of truth. A manifest append may survive a crash even
        # when the corresponding HDF5 write did not, so never skip from manifest
        # alone after the shards have already been scanned.
        if _done_keys != _done_keys_h5:
            print(
                "[RESUME] manifest/HDF5 mismatch: "
                f"manifest={len(_done_keys)}, hdf5={len(_done_keys_h5)}; "
                "using HDF5 keys"
            )
        _done_keys = _done_keys_h5
        if _done_keys:
            print(f"[RESUME] tổng {len(_done_keys)} mẫu đã xong — "
                  f"tiếp tục từ shard_{_resume_shard_number + 1:04d}")

    #text dictionary
    #phoenix
    dic, ori_dictionary = make_text_dic(
        text_path,
        text_encoder_backend=text_encoder_backend,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        normalize_clip_text=normalize_clip_text,
        text_batch_size=text_batch_size,
    )
    print('dic done')
    #csl-daily
    # dic, ori_dictionary = make_csd_text_dic(text_path)
    # print(dic.get('01December_2011_Thursday_tagesschau_default-12'))
    input_dir = resolve_split_dir(sl_dir, split)
    print("input_dir: ", input_dir)
    sample_path_index = index_sample_paths(input_dir)
    print("indexed samples: ", len(sample_path_index))
    filenames = []
    missing = []
    for sample_name in dic.keys():
        sample_path = sample_path_index.get(sample_name)
        if sample_path is None:
            for candidate in _sample_name_variants(sample_name):
                sample_path = sample_path_index.get(candidate)
                if sample_path is not None:
                    break
        if sample_path is None:
            sample_path = find_sample_path(input_dir, sample_name)
        if sample_path is None:
            missing.append(sample_name)
            continue
        filenames.append(sample_path)
    print("filenames: ", len(filenames))
    if filenames:
        print(filenames[0])
    if missing:
        print(f"missing samples skipped: {len(missing)}")
        print("first missing sample: ", missing[0])
    if len(filenames) == 0:
        raise FileNotFoundError(
            f"No samples were found under '{input_dir}'. "
            f"Check --sl_dir and --split before creating HDF5 shards."
        )

    # ── Lọc mẫu đã xử lý rồi (resume) ──────────────────────────────────────
    if _done_keys:
        # Debug: kiểm tra xem sample_id của filenames có khớp với done_keys không
        if filenames:
            sample_f = filenames[0]
            sample_id = _sample_id(sample_f)
            in_done = sample_id in _done_keys
            print(f"[RESUME DEBUG] ví dụ filename[0]: {repr(sample_f)}")
            print(f"[RESUME DEBUG] _sample_id → {repr(sample_id)}, in done_keys={in_done}")
        filenames_before = len(filenames)
        filenames = [f for f in filenames if _sample_id(f) not in _done_keys]
        print(f"[RESUME] bỏ qua {filenames_before - len(filenames)} mẫu đã có, "
              f"còn {len(filenames)} mẫu cần xử lý")
        if len(filenames) == 0:
            print(f"[SKIP] {out_dir}: tất cả {filenames_before} mẫu đã hoàn thành.")
            return
    else:
        print(f"[RESUME] done_keys rỗng — không tìm thấy mẫu nào trong H5, bắt đầu từ đầu")

    # H5 maker:
    #   force_h5=True  → xóa toàn bộ shard cũ, bắt đầu lại (SignLanguage_HDF5Maker force=True)
    #   force_h5=False → _ResumableSignLanguageHDF5Maker luôn luôn, bất kể có shard cũ hay không
    #                    (resume_from_shard=0 → tạo shard_0001.hdf5 mới;
    #                     resume_from_shard=N → tiếp tục từ shard_N+1)
    #   NOTE: KHÔNG dùng SignLanguage_HDF5Maker(force=False) khi out_dir đã tồn tại
    #         vì HDF5Maker.__init__ sẽ raise ValueError: "... already exists."
    pipeline_metadata = {
        "schema_version": 2,
        "image_size": int(image_size),
        "video_preprocessing": (
            "center_crop_then_resize"
            if center_crop_before_resize else "resize_only"
        ),
        "video_layout": "THWC",
        "text_encoder_backend": str(text_encoder_backend),
        "text_encoder_model": str(clip_model),
        "text_encoder_pretrained": str(clip_pretrained),
        "text_embedding_normalized": bool(normalize_clip_text),
        "pose_extractor": "DWPose",
        "pose_inference_size": int(
            _read_int_env('SIGNGEN_DWPOSE_INFER_SIZE', image_size)
        ),
        "pose_representation": (
            "normalized_keypoints_rendered_at_output_size"
            if os.environ.get('SIGNGEN_POSE_KEYPOINT_CACHE_ROOT')
            else "pose_rgb_render"
        ),
        "pose_layout": "THWC",
        "depth_extractor": "DPT",
        "depth_layout": "THW",
        "legacy_motion_group": (
            "FVD proxy; use optical_flow_tvl1 for dense optical flow"
        ),
    }

    if force_h5:
        h5_maker = SignLanguage_HDF5Maker(
            out_dir,
            num_per_shard=vids_per_shard,
            force=True,
            video=True,
            compression=h5_compression,
            pipeline_metadata=pipeline_metadata,
        )
    else:
        h5_maker = _ResumableSignLanguageHDF5Maker(
            out_dir,
            resume_from_shard=_resume_shard_number,
            num_per_shard=vids_per_shard,
            compression=h5_compression,
            pipeline_metadata=pipeline_metadata,
        )
    written = 0
    compression_name = h5_compression if h5_compression is not None else "none"
    print(
        f'processing {len(filenames)} samples with {max(1, int(num_workers))} worker thread(s), '
        f'h5 compression={compression_name}'
    )
    print(
        "video preprocessing: "
        f"{'center-crop -> resize' if center_crop_before_resize else 'resize only'} "
        f"to {image_size}x{image_size}"
    )

    for video_path, processed in tqdm(
        iter_processed_videos(
            filenames,
            image_size,
            num_workers=num_workers,
            center_crop_before_resize=center_crop_before_resize,
        ),
        total=len(filenames),
    ):
        # try:
        if processed == "break":
            break
        if processed is None:
            continue

        video,handpose_video,motion,depth = processed

        text_key = _sample_id(video_path)
        validate_processed_sample(
            video,
            handpose_video,
            motion,
            depth,
            image_size,
            text_key,
        )
        text_value = dic[text_key]
        origin_text = ori_dictionary[text_key]
        if text_value.shape != (1, 1024) or not np.isfinite(text_value).all():
            raise ValueError(
                f"Sample {text_key!r} has invalid text embedding: "
                f"shape={text_value.shape}, finite={np.isfinite(text_value).all()}"
            )

        h5_maker.add_video_data_all(video, handpose_video, motion, text_value, text_key, depth)
        _append_manifest(out_dir, text_key)
        written += 1

        # except StopIteration:
        #     break

        # except (KeyboardInterrupt, SystemExit):
        #     print("Ctrl+C!!")
        #     break

        # except:
        #     e = sys.exc_info()[0]
        #     print("ERROR:", e)
            

    h5_maker.close()
    if written == 0:
        raise RuntimeError(
            f"No valid samples were written to '{out_dir}'. "
            f"Check whether the required pose/motion/depth side inputs exist for the selected split."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--sl_dir', type=str, help="Directory with videos")
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)
    parser.add_argument('--split', type=str, default=False)
    parser.add_argument('--image_size', type=eval,  default=False)
    parser.add_argument('--text_path', type=str, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--h5_compression', type=str, default='lzf')
    parser.add_argument('--text_encoder_backend', type=str, default='open_clip',
                        choices=['open_clip', 'clip'])
    parser.add_argument('--clip_model', type=str, default='ViT-H-14')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s32b_b79k')
    parser.add_argument('--normalize_clip_text', action='store_true')
    parser.add_argument('--text_batch_size', type=int,
                        default=_read_int_env('SIGNGEN_TEXT_BATCH_SIZE', 128))
    parser.add_argument('--center_crop_before_resize', type=eval, default=False,
                        help="If True, center-crop each RGB frame to a square before resizing. "
                             "This matches the SignGen supplemental preprocessing.")
    parser.add_argument('--eager_init_features', action='store_true',
                        help="Preload FVD/DPT models before processing. By default, feature models are loaded lazily.")

    args = parser.parse_args()

    _configure_torch_runtime()
    if args.eager_init_features:
        initialize_feature_models()

    make_h5_from_sl(out_dir=os.path.join(args.out_dir),
                    sl_dir=args.sl_dir,
                    split=args.split,
                    image_size=args.image_size, vids_per_shard=args.vids_per_shard,
                    force_h5=args.force_h5,
                    text_path = args.text_path,
                    num_workers=args.num_workers,
                    h5_compression=_resolve_h5_compression(args.h5_compression),
                    text_encoder_backend=args.text_encoder_backend,
                    clip_model=args.clip_model,
                    clip_pretrained=args.clip_pretrained,
                    normalize_clip_text=args.normalize_clip_text,
                    text_batch_size=args.text_batch_size,
                    center_crop_before_resize=args.center_crop_before_resize)
    print('done')
