import warnings
from typing import List, Type, Optional, Tuple, Union

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json

# import nnunet
from nnunet.configuration import default_num_processes
from nnunet.fingerprint_extractor import DatasetFingerprintExtractor
from nnunet.verify_dataset_integrity import verify_dataset_integrity
from nnunet.paths import nnUNet_raw, nnUNet_preprocessed
from nnunet.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunet.utilities.find_class_by_name import recursive_find_python_class



def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = DatasetFingerprintExtractor(dataset_id, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)


