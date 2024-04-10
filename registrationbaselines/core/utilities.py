from pathlib import Path
import datetime

from typing import Tuple


def create_result_paths(directory: Path,
                        name_fixed: str,
                        name_moving: str,
                        method: str,
                        extension_image: str = ".nii",
                        extension_transformatoin: str = ".tfm") -> Tuple[Path, Path]:
    
    """
    Create the paths for the result files (warped image and transformatoin)."""
    
    date_time = datetime.datetime.now()

    result_transformed_image_path = directory / f"{name_moving}_warped_on_{name_fixed}_{method}_image_{date_time}{extension_image}"
    result_transformation_path = directory / f"{name_moving}_warped_on_{name_fixed}_{method}_transform_{date_time}{extension_transformatoin}"

    # replace spaces with underscores
    result_transformed_image_path = result_transformed_image_path.resolve().as_posix().replace(" ", "_")
    result_transformation_path = result_transformation_path.resolve().as_posix().replace(" ", "_")

    return Path(result_transformed_image_path), Path(result_transformation_path)
