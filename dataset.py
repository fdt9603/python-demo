# 电路板缺陷公开数据集
# 1. PCB Dataset (Kaggle)
# 2. DeepPCB Dataset
# 3. PKU-Market-PCB

# 此文件已迁移到 data_loader.py
# 请使用 data_loader.py 中的完整功能

from data_loader import (
    load_pcb_dataset,
    augment_defects,
    load_huggingface_pcb_dataset,
    create_sample_labels_json
)

# 向后兼容
__all__ = [
    'load_pcb_dataset',
    'augment_defects',
    'load_huggingface_pcb_dataset',
    'create_sample_labels_json'
]