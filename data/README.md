# 在自己的机器上生成并保存缓存

本仓库不上传特征数据。复现者先准备原始数据集、对应训练／测试划分文件与模型权重，首次运行作者复现入口时，`src/utils.py::get_features`会编码并把特征保存到本地`data/`。再次运行相同骨干、数据集和划分时，会直接读取已有缓存。

目录结构为：

```text
data/
  image_features/<数据集>/<骨干>/<划分>/
  image_noproj_features/<数据集>/<骨干>/<划分>/
  text_features/<数据集>/<骨干>/<划分>/
  text_noproj_features/<数据集>/<骨干>/<划分>/
```

每个目录保存特征`.pt`与对齐名称名单`.pkl`；Iso入口请求投影前特征，因此使用`*_noproj_features`。这些文件由代码自动写入，无需上传GitHub，也无需下载本项目作者的缓存。

运行原版复现可使用`src/retrieval.py`或`reproduction/programs/run_retrieval.py`配合相应配置，先把配置中的数据路径与输出目录改为自己的位置。

## 探索入口的额外准备

`isoclip.evaluate`与`isoclip.train`读取配置中`prepared`指向的中间文件。它们还包括标签、查询／图库身份、校准名单和谱；当前`isoclip.prepare`会读取已有的划分身份参考。这个准备步骤与首次提取原始特征分别完成，当前代码并非从空目录一条命令自动创建全部探索输入。

首次原图编码需要时间与计算，后续策略实验复用已经保存的特征即可。`data/`下生成的缓存由`.gitignore`排除。
