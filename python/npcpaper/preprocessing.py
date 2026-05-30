from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def read_image(path: str | Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_image(img: sitk.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def n4_bias_correct(img: sitk.Image, shrink_factor: int = 2, iterations: tuple[int, ...] = (50, 50, 30, 20)) -> sitk.Image:
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img_float, 0, 1, 200)
    if shrink_factor > 1:
        img_small = sitk.Shrink(img_float, [shrink_factor] * img.GetDimension())
        mask_small = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())
    else:
        img_small, mask_small = img_float, mask
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrected_small = corrector.Execute(img_small, mask_small)
    log_field = corrector.GetLogBiasFieldAsImage(img_float)
    corrected = img_float / sitk.Exp(log_field)
    corrected.CopyInformation(img)
    return corrected


def resample_image(
    img: sitk.Image,
    reference: sitk.Image | None = None,
    spacing: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkBSpline,
    default_value: float = 0.0,
) -> sitk.Image:
    if reference is not None:
        return sitk.Resample(img, reference, sitk.Transform(), interpolator, default_value, img.GetPixelID())
    spacing = tuple(float(s) for s in spacing)
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = [int(round(old_size[i] * old_spacing[i] / spacing[i])) for i in range(3)]
    return sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        default_value,
        img.GetPixelID(),
    )


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> tuple[sitk.Image, sitk.Transform]:
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, 1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    transform = reg.Execute(fixed_f, moving_f)
    moved = sitk.Resample(moving, fixed, transform, sitk.sitkBSpline, 0.0, moving.GetPixelID())
    return moved, transform


def zscore_normalize(img: sitk.Image, mask: sitk.Image | None = None) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if mask is not None:
        m = sitk.GetArrayFromImage(mask) > 0
        vals = arr[m]
    else:
        vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        vals = arr.reshape(-1)
    mean = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd == 0 or not np.isfinite(sd):
        sd = 1.0
    out = (arr - mean) / sd
    res = sitk.GetImageFromArray(out.astype(np.float32))
    res.CopyInformation(img)
    return res


def preprocess_patient(row: pd.Series, sequences: list[str], reference_sequence: str, outdir: str | Path, spacing: list[float], do_n4: bool = True) -> dict[str, str]:
    pid = str(row["patient_id"])
    outdir = Path(outdir) / pid
    outdir.mkdir(parents=True, exist_ok=True)
    imgs = {seq: read_image(row[f"{seq}_path"]) for seq in sequences}
    if do_n4:
        imgs = {seq: n4_bias_correct(img) for seq, img in imgs.items()}
    ref = imgs[reference_sequence]
    ref_iso = resample_image(ref, spacing=spacing, interpolator=sitk.sitkBSpline)
    paths: dict[str, str] = {"patient_id": pid}
    transforms = {}
    for seq in sequences:
        if seq == reference_sequence:
            img = ref_iso
        else:
            moved, tx = rigid_register(ref, imgs[seq])
            transforms[seq] = tx
            img = resample_image(moved, reference=ref_iso, interpolator=sitk.sitkBSpline)
        img = zscore_normalize(img)
        out_path = outdir / f"{pid}_{seq}_preprocessed.nii.gz"
        write_image(img, out_path)
        paths[f"{seq}_preprocessed_path"] = str(out_path)
    if "mask_path" in row and isinstance(row["mask_path"], str) and Path(row["mask_path"]).exists():
        mask = read_image(row["mask_path"])
        mask_iso = resample_image(mask, reference=ref_iso, interpolator=sitk.sitkNearestNeighbor)
        mask_path = outdir / f"{pid}_mask_preprocessed.nii.gz"
        write_image(mask_iso, mask_path)
        paths["mask_preprocessed_path"] = str(mask_path)
    return paths
