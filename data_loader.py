"""
Day 0: 电路板缺陷数据集加载器
支持公开数据集和自定义数据集两种方式
"""
import json
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Dict, Any, Generator
from datasets import Dataset, Features, Image as HFImage, Value, Sequence
import os


def augment_defects(batch: Dict[str, List]) -> Dict[str, List]:
    """
    实时数据增强：旋转、翻转、亮度调整
    缺陷样本增强10倍，解决类别不平衡问题
    """
    images = []
    labels = []
    
    for img, label in zip(batch.get("image", []), batch.get("defect_type", [])):
        # 原图
        images.append(img)
        labels.append(label)
        
        # 如果是缺陷样本（非"normal"），增强10倍
        if label != "normal":
            for _ in range(10):
                # 随机旋转±15度
                angle = np.random.randint(-15, 15)
                img_aug = img.rotate(angle, fillcolor=(255, 255, 255))
                
                # 随机亮度调整
                enhancer = ImageEnhance.Brightness(img_aug)
                img_aug = enhancer.enhance(np.random.uniform(0.8, 1.2))
                
                # 随机对比度调整
                contrast_enhancer = ImageEnhance.Contrast(img_aug)
                img_aug = contrast_enhancer.enhance(np.random.uniform(0.9, 1.1))
                
                images.append(img_aug)
                labels.append(label)
    
    return {"images": images, "defect_type": labels}


def load_pcb_dataset(data_dir: str, augment: bool = True) -> Dataset:
    """
    加载自定义PCB缺陷数据集
    
    Args:
        data_dir: 数据集根目录，包含 images/ 和 labels.json
        augment: 是否进行数据增强
    
    labels.json 格式：
    [
        {
            "image": "xxx.jpg",
            "defects": [
                {"type": "short", "bbox": [x, y, w, h], "repair": "清理焊锡桥接"},
                {"type": "open", "bbox": [x2, y2, w2, h2], "repair": "补焊"}
            ]
        },
        ...
    ]
    """
    labels_path = os.path.join(data_dir, "labels.json")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    def gen() -> Generator[Dict[str, Any], None, None]:
        for item in labels_data:
            img_path = os.path.join(data_dir, "images", item['image'])
            
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在 {img_path}，跳过")
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                # 调整图像大小（Qwen3-VL最优尺寸为448x448）
                if img.size != (448, 448):
                    img = img.resize((448, 448), Image.Resampling.LANCZOS)
                
                # 生成问答格式（符合MLLM输入）
                question = "检测这张电路板的所有缺陷，返回JSON格式：[{'defect': '类型', 'bbox': [x,y,w,h], 'repair': '维修建议'}]"
                
                # 将bbox转换为文本描述（训练用）
                defects_text = json.dumps(item["defects"], ensure_ascii=False)
                
                yield {
                    "image": img,
                    "question": question,
                    "answer": defects_text,  # 目标输出
                    "defect_type": item["defects"][0]["type"] if item["defects"] else "normal"
                }
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue
    
    dataset = Dataset.from_generator(
        gen,
        features=Features({
            "image": HFImage(),
            "question": Value("string"),
            "answer": Value("string"),
            "defect_type": Value("string")
        })
    )
    
    # 如果需要增强，应用增强函数
    if augment:
        dataset = dataset.map(
            augment_defects,
            batched=True,
            batch_size=1,
            remove_columns=dataset.column_names,
            desc="应用数据增强"
        )
    
    return dataset


def load_huggingface_pcb_dataset(dataset_name: str = "hf-internal-testing/pcb-defects", 
                                 split: str = "train", 
                                 streaming: bool = True):
    """
    从HuggingFace加载PCB数据集（如果可用）
    
    Args:
        dataset_name: HuggingFace数据集名称
        split: 数据集分割
        streaming: 是否使用流式加载
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        return dataset
    except Exception as e:
        print(f"无法从HuggingFace加载数据集: {e}")
        print("请使用自定义数据集或手动下载数据集")
        return None


def create_sample_labels_json(output_path: str, images_dir: str, num_samples: int = 10):
    """
    创建示例labels.json文件（用于测试）
    
    Args:
        output_path: 输出labels.json的路径
        images_dir: 图像目录
        num_samples: 生成的样本数量
    """
    sample_data = []
    
    defect_types = ["short", "open", "missing", "normal"]
    repair_suggestions = {
        "short": "清理焊锡桥接",
        "open": "补焊连接",
        "missing": "补装缺失元件",
        "normal": "无缺陷"
    }
    
    # 获取图像文件列表
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        image_files = [f"sample_{i}.jpg" for i in range(num_samples)]
    
    for i, img_file in enumerate(image_files[:num_samples]):
        defect_type = defect_types[i % len(defect_types)]
        
        if defect_type == "normal":
            defects = []
        else:
            # 生成随机bbox
            defects = [{
                "type": defect_type,
                "bbox": [
                    np.random.randint(50, 200),  # x
                    np.random.randint(50, 200),  # y
                    np.random.randint(20, 80),   # w
                    np.random.randint(20, 80)    # h
                ],
                "repair": repair_suggestions[defect_type]
            }]
        
        sample_data.append({
            "image": img_file,
            "defects": defects
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"示例labels.json已创建: {output_path}")
    print(f"包含 {len(sample_data)} 个样本")


if __name__ == "__main__":
    # 测试数据加载
    import tempfile
    
    # 创建临时目录结构
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        labels_path = os.path.join(tmpdir, "labels.json")
        create_sample_labels_json(labels_path, images_dir, num_samples=5)
        
        print(f"\n数据集结构:")
        print(f"  {tmpdir}/")
        print(f"    images/")
        print(f"    labels.json")
        print(f"\n请将您的电路板图像放入 images/ 目录，并更新 labels.json 文件")

