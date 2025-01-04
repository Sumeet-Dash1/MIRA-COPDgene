import os
from pathlib import Path
import subprocess
import shutil
from typing import List, Union


class ElastixTransformix:
    def __init__(self, verbose: bool = False, elastix_path: str = None):
        """
        Class to wrap Elastix and Transformix functionalities.

        Args:
            verbose (bool): Whether to print Elastix/Transformix logs. Defaults to False.
            elastix_path (str): Path to Elastix binaries. Defaults to None.
        """
        self.verbose = verbose
        self.elastix_path = elastix_path

    def run_elastix(
        self,
        fix_img_path: Union[str, Path],
        mov_img_path: Union[str, Path],
        res_path: Union[str, Path],
        parameters_paths: List[Union[str, Path]],
        fix_mask_path: Union[str, Path] = None,
        mov_mask_path: Union[str, Path] = None
    ) -> Path:
        """
        Run Elastix for image registration.

        Args:
            fix_img_path (Union[str, Path]): Path to the fixed image.
            mov_img_path (Union[str, Path]): Path to the moving image.
            res_path (Union[str, Path]): Path where results are stored.
            parameters_paths (List[Union[str, Path]]): List of Elastix parameter files.
            fix_mask_path (Union[str, Path], optional): Path to fixed image mask. Defaults to None.
            mov_mask_path (Union[str, Path], optional): Path to moving image mask. Defaults to None.

        Returns:
            Path: Path to the transformation file.
        """
        # Convert paths to Path objects
        os.makedirs(res_path, exist_ok=True)
        fix_img_path = Path(fix_img_path)
        mov_img_path = Path(mov_img_path)
        res_path = Path(res_path)
        parameters_paths = [Path(p) for p in parameters_paths]
        fix_mask_path = Path(fix_mask_path) if fix_mask_path else None
        mov_mask_path = Path(mov_mask_path) if mov_mask_path else None

        # Prepare command
        command = [f'{self.elastix_path}/elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m', str(mov_img_path)]
        if fix_mask_path:
            command.extend(['-fMask', str(fix_mask_path)])
        if mov_mask_path:
            command.extend(['-mMask', str(mov_mask_path)])
        for param_path in parameters_paths:
            command.extend(['-p', str(param_path)])

        # Execute command
        if self.verbose:
            print("Running command:", " ".join(command))
            subprocess.call(command)
        else:
            subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Process results
        transformation_file = res_path / 'TransformParameters.0.txt'

        transformation_filename = f'TransformParameters_{mov_img_path.stem}.txt'
        transformation_file.rename(res_path / transformation_filename)

        return res_path / transformation_filename

    def run_transformix(
        self,
        mov_img_path: Union[str, Path],
        res_path: Union[str, Path],
        transformation_path: Union[str, Path],
        points: bool = False
    ) -> Path:
        """
        Run Transformix for applying transformations.

        Args:
            mov_img_path (Union[str, Path]): Path to the moving image or points file.
            res_path (Union[str, Path]): Path to store transformed result.
            transformation_path (Union[str, Path]): Path to the transformation file.
            points (bool, optional): Whether to transform points instead of an image. Defaults to False.

        Returns:
            Path: Path to the transformed result.
        """
        # Convert paths to Path objects
        mov_img_path = Path(mov_img_path)
        res_path = Path(res_path)
        transformation_path = Path(transformation_path)

        # Prepare command
        command = [f'{self.elastix_path}/transformix', '-out', str(res_path), '-tp', str(transformation_path)]
        if points:
            command.extend(['-def', str(mov_img_path)])
        else:
            command.extend(['-in', str(mov_img_path)])

        # Execute command
        if self.verbose:
            print("Running command:", " ".join(command))
            subprocess.call(command)
        else:
            subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Process results
        if points:
            result_file = res_path / 'outputpoints.txt'
            res_filename = f'{mov_img_path.stem}_points.txt'
        else:
            result_file = res_path / 'result.nii.gz'
            res_filename = f'{mov_img_path.stem}.nii.gz'

        if not result_file.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_file}")

        result_file.rename(res_path / res_filename)
        return res_path / res_filename