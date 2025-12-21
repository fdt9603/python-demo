# DeepPCB数据集转换指�?

本指南说明如何将DeepPCB数据集转换为项目所需的格式�?

## 📋 目录

1. [数据集下载](#数据集下�?
2. [格式说明](#格式说明)
3. [转换步骤](#转换步骤)
4. [验证转换结果](#验证转换结果)
5. [常见问题](#常见问题)

---

## 📥 数据集下�?

### 方法1: 从GitHub下载（推荐）

```bash
# 克隆DeepPCB仓库
git clone https://github.com/tangsanli5201/DeepPCB.git

# 或者如果下载的是zip文件，解压后会得�?DeepPCB-master 目录
# 数据集在 PCBData/ 目录�?
cd DeepPCB  # �?cd DeepPCB-master
ls PCBData/
```

### 方法2: 直接下载数据�?

访问 [DeepPCB GitHub仓库](https://github.com/tangsanli5201/DeepPCB) 下载 `PCBData` 文件夹�?

**数据集大�?*: �?500MB - 1.5GB

**目录结构**:
```
DeepPCB-master/
  PCBData/
    group00041/
      00041/
        00041000_test.jpg  # 测试图像（带缺陷�?
        00041000_temp.jpg  # 模板图像（无缺陷�?
        00041000.txt       # 标注文件
        ...
      00041_not/
        00041000.txt       # 标注文件（在_not子目录中�?
        ...
    group12000/
      12000/
        12000001_test.jpg
        12000001_temp.jpg
        12000001.txt
        ...
      12000_not/
        12000001.txt
        ...
    ...
```

**注意**: DeepPCB数据集使用分组目录结构，转换脚本会自动递归查找所�?`*_test.jpg` 文件，无需手动整理�?

---

## 📊 格式说明

### DeepPCB原始格式

- **图像文件**: 
  - `xxx_test.jpg` - 带缺陷的测试图像�?40x640像素�?
  - `xxx_temp.jpg` - 无缺陷的模板图像�?40x640像素�?
  
- **标注文件**: `xxx.txt`
  - 格式: `x1,y1,x2,y2,type`
  - `(x1,y1)`: 左上角坐�?
  - `(x2,y2)`: 右下角坐�?
  - `type`: 缺陷类型ID
    - 0: 背景（不使用�?
    - 1: open（断路）
    - 2: short（短路）
    - 3: mousebite（鼠咬）
    - 4: spur（毛刺）
    - 5: copper（多余铜�?
    - 6: pin-hole（针孔）

### 项目所需格式

- **图像**: 统一放在 `data/pcb_defects/images/` 目录
- **标签文件**: `data/pcb_defects/labels.json`

```json
[
  {
    "image": "deeppcb_000000.jpg",
    "defects": [
      {
        "type": "short",
        "bbox": [120, 350, 45, 12],
        "repair": "清理焊锡桥接，检查相邻焊�?
      }
    ]
  },
  {
    "image": "deeppcb_000001.jpg",
    "defects": []
  }
]
```

**格式差异**:
- �?bbox�?`(x1,y1,x2,y2)` 转换�?`[x,y,w,h]`
- �?type从数字ID转换为字符串�?short", "open", "missing"�?
- �?添加 `repair` 字段（维修建议）

---

## 🔄 转换步骤

### �?�? 准备数据�?

确保你已经下载了DeepPCB数据集，目录结构如下�?

```
DeepPCB/
  PCBData/
    00041000_test.jpg
    00041000_temp.jpg
    00041000.txt
    00041001_test.jpg
    ...
```

### �?�? 运行转换脚本

脚本会自动查�?`PCBData` 目录，你可以直接指定 `DeepPCB-master` 根目录或 `PCBData` 目录�?

```bash
# 方式1: 指定DeepPCB-master根目录（推荐�?
python tools/convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master

# 方式2: 直接指定PCBData目录
python tools/convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master/PCBData

# 指定输出目录
python tools/convert_deeppcb_dataset.py \
    --deeppcb_dir /path/to/DeepPCB-master \
    --output_dir ./data/pcb_defects

# 自定义训练集比例（默�?0%训练�?0%测试�?
python tools/convert_deeppcb_dataset.py \
    --deeppcb_dir /path/to/DeepPCB-master \
    --output_dir ./data/pcb_defects \
    --split_ratio 0.9

# 不打乱数据集（保持原始顺序）
python tools/convert_deeppcb_dataset.py \
    --deeppcb_dir /path/to/DeepPCB-master \
    --no-shuffle

# 自定义随机种�?
python tools/convert_deeppcb_dataset.py \
    --deeppcb_dir /path/to/DeepPCB-master \
    --seed 123
```

**Windows示例**:
```powershell
# 如果数据集在 D:\datasets\DeepPCB-master
python tools/convert_deeppcb_dataset.py --deeppcb_dir "D:\datasets\DeepPCB-master"

# 或者直接指定PCBData目录
python tools/convert_deeppcb_dataset.py --deeppcb_dir "D:\datasets\DeepPCB-master\PCBData"
```

**Linux/Mac示例**:
```bash
# 如果数据集在 ~/datasets/DeepPCB-master
python tools/convert_deeppcb_dataset.py --deeppcb_dir ~/datasets/DeepPCB-master

# 或者直接指定PCBData目录
python tools/convert_deeppcb_dataset.py --deeppcb_dir ~/datasets/DeepPCB-master/PCBData
```

### �?�? 查看转换结果

转换完成后，输出目录结构�?

```
data/
  pcb_defects/
    images/
      deeppcb_000000.jpg
      deeppcb_000001.jpg
      ...
    labels.json          # 训练集标签（80%�?
    labels_test.json     # 测试集标签（20%�?
    image_mapping.json   # 原始文件名映�?
```

### �?�? 验证数据�?

```bash
# 使用快速检查脚本验�?
python tools/quick_start.py

# 或者手动检�?
python -c "from data_loader import load_pcb_dataset; d=load_pcb_dataset('data/pcb_defects'); print(f'数据集大�? {len(d)}')"
```

---

## �?验证转换结果

### 检查统计信�?

转换脚本会显示：
- �?转换的图像数�?
- �?缺陷类型统计
- �?训练�?测试集分割信�?

### 验证JSON格式

```python
import json

# 检查labels.json格式
with open('data/pcb_defects/labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"训练集样本数: {len(data)}")
    print(f"第一个样�? {json.dumps(data[0], ensure_ascii=False, indent=2)}")
```

### 可视化检�?

```python
from PIL import Image
import json

# 加载标签
with open('data/pcb_defects/labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查第一张图�?
sample = data[0]
img_path = f"data/pcb_defects/images/{sample['image']}"
img = Image.open(img_path)
print(f"图像尺寸: {img.size}")
print(f"缺陷数量: {len(sample['defects'])}")
for defect in sample['defects']:
    print(f"  类型: {defect['type']}, bbox: {defect['bbox']}")
```

---

## 🔍 缺陷类型映射

DeepPCB�?种缺陷类型会映射到项目的4种类型：

| DeepPCB类型 | DeepPCB ID | 项目类型 | 说明 |
|------------|-----------|---------|------|
| open | 1 | `open` | 断路 |
| short | 2 | `short` | 短路 |
| mousebite | 3 | `open` | 鼠咬（映射为断路�?|
| spur | 4 | `short` | 毛刺（映射为短路�?|
| copper | 5 | `missing` | 多余铜（映射为缺件） |
| pin-hole | 6 | `missing` | 针孔（映射为缺件�?|

如果需要保留所�?种类型，可以修改 `convert_deeppcb_dataset.py` 中的 `DEEPPCB_TYPE_MAP`�?

---

## ⚠️ 常见问题

### Q1: 找不到数据集目录

**错误信息**: `未找到DeepPCB数据集目录` �?`未找到任�?*_test.jpg 图像文件`

**解决方案**:

1. **检查目录结�?*:
   ```bash
   # Windows
   dir D:\datasets\DeepPCB-master\PCBData
   
   # Linux/Mac
   ls /path/to/DeepPCB-master/PCBData/
   ```

2. **确认包含_test.jpg文件**:
   ```bash
   # Windows PowerShell
   Get-ChildItem "D:\datasets\DeepPCB-master\PCBData" -Filter "*_test.jpg" | Select-Object -First 5
   
   # Linux/Mac
   ls /path/to/DeepPCB-master/PCBData/*_test.jpg | head -5
   ```

3. **尝试直接指定PCBData目录**:
   ```bash
   python tools/convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master/PCBData
   ```

4. **检查路径中的中文或特殊字符**:
   如果路径包含中文或特殊字符，确保使用正确的引号：
   ```powershell
   # Windows - 使用双引�?
   python tools/convert_deeppcb_dataset.py --deeppcb_dir "D:\数据集\DeepPCB-master"
   ```

### Q2: 标注文件格式错误

**错误信息**: `解析标注文件 xxx �?X 行失败` �?`无法读取标注文件`

**解决方案**:

1. **检查标注文件格�?*:
   确保格式�? `x1,y1,x2,y2,type`（逗号分隔，无空格�?
   ```bash
   # 查看一个标注文件示�?
   head -5 /path/to/DeepPCB-master/PCBData/00041000.txt
   ```

2. **检查文件编�?*:
   脚本会自动尝试UTF-8、GBK和Latin-1编码。如果仍有问题，可以手动转换�?
   ```bash
   # Linux/Mac
   iconv -f gbk -t utf-8 input.txt > output.txt
   ```

3. **查看详细错误信息**:
   脚本会输出具体的行号和错误原因，根据提示修复

4. **验证标注文件完整�?*:
   ```bash
   # 检查标注文件数量是否匹配图像数�?
   ls PCBData/*_test.jpg | wc -l
   ls PCBData/*.txt | wc -l
   ```

### Q3: 图像复制失败

**错误信息**: `复制图像失败`

**解决方案**:
- 检查图像文件是否损�?
- 确认有足够的磁盘空间
- 检查文件权�?

### Q4: 转换后数据集为空

**可能原因**:
1. 所有标注文件都为空或格式错�?
2. 图像文件路径不正�?

**解决方案**:
```bash
# 检查原始数�?
ls /path/to/DeepPCB/PCBData/ | head -10

# 检查是否有标注文件
ls /path/to/DeepPCB/PCBData/*.txt | wc -l
```

### Q5: bbox坐标超出图像范围

**说明**: DeepPCB的标注可能包含超出图像范围的坐标

**处理**: 转换脚本会自动：
- 读取实际图像尺寸进行验证
- 自动裁剪超出范围的bbox
- 跳过无效的bbox（宽度或高度<=0�?

**查看处理结果**:
脚本会输出警告信息，如果有很多bbox被裁剪，可能需要检查原始标注文件�?

### Q6: 转换后图像数量不�?

**可能原因**:
1. 某些图像文件损坏无法读取
2. 某些标注文件格式错误导致跳过
3. 文件权限问题

**解决方案**:
```bash
# 检查原始数据数�?
ls PCBData/*_test.jpg | wc -l  # Linux/Mac
dir PCBData\*_test.jpg | find /c ".jpg"  # Windows

# 检查转换后的数�?
ls data/pcb_defects/images/ | wc -l  # Linux/Mac
dir data\pcb_defects\images | find /c ".jpg"  # Windows

# 查看转换日志中的错误和警告信�?
```

---

## 📝 自定义转�?

如果需要自定义转换逻辑，可以修�?`convert_deeppcb_dataset.py`:

### 修改缺陷类型映射

```python
# �?convert_deeppcb_dataset.py �?
DEEPPCB_TYPE_MAP = {
    1: "open",
    2: "short",
    3: "open",      # 改为其他类型
    4: "short",     # 改为其他类型
    5: "missing",
    6: "missing",
}
```

### 添加新的维修建议

```python
REPAIR_SUGGESTIONS = {
    "open": "你的自定义建�?,
    # ...
}
```

### 修改图像处理

�?`convert_deeppcb_dataset()` 函数中修改图像处理逻辑�?

```python
# 例如：调整图像尺�?
img = img.resize((448, 448), Image.Resampling.LANCZOS)
img.save(new_img_path, 'JPEG', quality=95)
```

---

## 🚀 转换后的下一�?

转换完成后，你可以：

1. **验证数据�?*:
   ```bash
   python tools/quick_start.py
   ```

2. **开始训�?*:
   ```bash
   python pcb_train.py --data_dir ./data/pcb_defects
   ```

3. **测试数据加载**:
   ```python
   from data_loader import load_pcb_dataset
   dataset = load_pcb_dataset('./data/pcb_defects')
   print(f"数据集大�? {len(dataset)}")
   ```

---

## 📚 相关文档

- [README.md](../README.md) - 项目总览
- [RUN_GUIDE.md](RUN_GUIDE.md) - 运行指南
- [DeepPCB GitHub](https://github.com/tangsanli5201/DeepPCB) - 原始数据�?

## 💡 使用提示

1. **首次转换建议**: 先在小范围测试（可以手动选择几组文件测试�?

2. **检查转换结�?*: 转换后务必使�?`python tools/quick_start.py` 验证数据集格�?

3. **数据集大�?*: DeepPCB包含1500对图像，转换后应该有1500张测试图�?

4. **训练集分�?*: 默认80%训练�?0%测试�?200张训练，300张测试）

5. **随机打乱**: 默认会随机打乱数据集，使用固定的随机种子�?2）确保可复现

---

**需要帮助？** 如果遇到问题，请检查常见问题部分或提交Issue�?

---

**需要帮助？** 如果遇到问题，请检查常见问题部分或提交Issue�?

