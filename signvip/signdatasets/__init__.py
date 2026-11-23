"""Dataset exports, imported lazily to avoid loading unused optional stacks."""

from importlib import import_module


_EXPORTS = {
    "SignCondDataset": (".sign_cond", "SignCondDataset"),
    "SignLangVideoDataset": (".sign_lang_video", "SignLangVideoDataset"),
    "HDF5SignLangVideoDataset": (
        ".hdf5_sign_lang_video",
        "HDF5SignLangVideoDataset",
    ),
    "HDF5VQSignTextDataset": (
        ".hdf5_vq_sign_text",
        "HDF5VQSignTextDataset",
    ),
    "VideoLengthDataset": (".video_length", "VideoLengthDataset"),
    "SignTextDataset": (".sign_text", "SignTextDataset"),
    "VQSignTextDataset": (".vq_sign_text", "VQSignTextDataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
