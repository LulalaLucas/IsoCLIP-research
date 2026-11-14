# 准备原始数据，在本机生成特征与中间文件

仓库包含生成代码与划分指纹，图像、模型、特征和实验结果保存在使用者自己的机器上。完整流程为：

**原图与固定划分 → 投影前CLIP特征缓存 → IsoCLIP的prepared中间文件 → 策略拟合／训练与评测。**

## 1. 准备原图和固定划分

使用[CUB官方数据](https://www.vision.caltech.edu/datasets/cub_200_2011/)和[CoOp的数据准备说明](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)。CoOp页面提供其他10个数据集的目录布局及固定`split_zhou_*.json`下载链接；Aircraft使用自己的文本标注文件。请使用固定划分，不要让旧加载器重新随机生成划分。

下表相对于数据根目录`dataroot`。图片目录、元数据都需要准备好；不要求下载本项目的`.pt`或`.pkl`。

| 配置中的数据集名称 | 图片目录 | 划分与标签文件 |
| --- | --- | --- |
| `cub2011` | `CUB_200_2011/images/` | `CUB_200_2011/`中的`images.txt`、`image_class_labels.txt`、`train_test_split.txt` |
| `oxford_pets` | `oxford_pets/images/` | `oxford_pets/split_zhou_OxfordPets.json` |
| `stanford_cars` | `stanford_cars/cars_train/`与`cars_test/` | `stanford_cars/split_zhou_StanfordCars.json` |
| `fgvc_aircraft` | `fgvc_aircraft/images/` | `fgvc_aircraft/variants.txt`、`images_variant_train.txt`、`images_variant_test.txt` |
| `caltech101` | `caltech-101/101_ObjectCategories/` | `caltech-101/split_zhou_Caltech101.json` |
| `eurosat` | `eurosat/2750/` | `eurosat/split_zhou_EuroSAT.json` |
| `oxford_flowers` | `oxford_flowers/jpg/` | `oxford_flowers/split_zhou_OxfordFlowers.json` |
| `dtd` | `dtd/images/` | `dtd/split_zhou_DescribableTextures.json` |
| `ucf101` | `ucf101/UCF-101-midframes/` | `ucf101/split_zhou_UCF101.json` |
| `food101` | `food-101/images/` | `food-101/split_zhou_Food101.json` |
| `sun397` | `sun397/SUN397/` | `sun397/split_zhou_SUN397.json` |

在`experiments/iso_b32/configs/prepare.json`中设置`dataroot`，例如`"/mnt/datasets"`。配置内相对路径都从仓库根目录解析。已有数据可以用软链接组织，避免重复下载。

## 2. 自动提取和保存特征

在已安装完整环境且具有CUDA GPU的机器上，从仓库根目录运行：

```bash
python -m isoclip.extract --config experiments/iso_b32/configs/prepare.json
```

首次编码会通过OpenAI的CLIP加载器下载ViT-B/32到`cache/clip/ViT-B-32.pt`，并校验固定SHA-256。已有权重可提前放在该路径；如需换位置，修改配置的`checkpoint`，文件名保持`ViT-B-32.pt`。编码使用冻结的FP32骨干，取视觉投影前的768维特征，批大小32。

默认保存到：

```text
data/image_noproj_features/<数据集>/ViT-B32/<划分>/
  image_noproj_features.pt
  image_noproj_names.pkl
```

CUB编码`all`，其余数据集分别编码`train`和`test`。再次运行会检查名单并复用已存在的缓存；缓存格式也与作者入口兼容。

## 3. 自动生成prepared

```bash
python -m isoclip.prepare --config experiments/iso_b32/configs/prepare.json
```

`prepared`是本地生成的实验中间目录，默认`runs/iso_b32/prepared/`。准备程序从原始标注读取样本名单和标签，校验11个数据集的样本数、名单顺序及标签指纹，构造原Iso投影与谱，并生成：

- `spectrum.pt`、`energy_basis.pt`：投影与子空间信息。
- `<数据集>.pt`：查询／图库特征、标签、名单、训练校准子集及记录。
- `mixed/`、`energy/`：明确配置使用的补充对照输入。
- `manifest.json`：本次准备配置与各数据集的划分指纹。

准备程序直接使用原始标注，不读取旧`prepared`或历史研究结果。所有校准样本都来自训练图库，测试查询不参与拟合。CUB保留原实验的确定性训练顺序，仍包含全部5994张官方训练图像；其余数据集保持固定划分文件中的顺序。校准集按文件名SHA-256排序选择至多4096张。

如果出现`annotations differ from the frozen PPT protocol`，表示当前划分、顺序或标签与实验约定不一致，应核对固定划分文件。

后续运行见[探索实验说明](../experiments/iso_b32/README.md)。`data/`、`datasets/`、`cache/`和`runs/`均被Git排除，生成后直接保存在复现者本机。
