"""Import this BEFORE anything from nbjw_calib.model.transforms / transforms_l.

The NBJW plugin's transforms files do `import torchvision.transforms.v2 as v2`, which does not exist
in torchvision 0.14 (pinned by this project, torch 1.13.1). v2 is only used there for v2.Compose,
and the classic torchvision.transforms.Compose behaves the same for these dict samples.

    import tools.nbjw_compat  # noqa: F401
    from nbjw_calib.model.transforms import transforms, no_transforms
"""
import sys

import torchvision.transforms as _T

if "torchvision.transforms.v2" not in sys.modules:
    try:
        import torchvision.transforms.v2  # noqa: F401  (newer torchvision: nothing to do)
    except ImportError:
        sys.modules["torchvision.transforms.v2"] = _T
