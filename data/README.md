# 特征缓存数据

完整的`data/`缓存放在同一私有仓库的[数据Release](https://github.com/LulalaLucas/IsoCLIP-research/releases/tag/data-20261114)，包含216个文件、B/16与B/32两种骨干的四类缓存：图像特征、投影前图像特征、文本特征、投影前文本特征，以及各自名称名单。每个包解压后直接恢复`data/<类型>/...`结构。

下载需要你的GitHub账号已获得本私有仓库访问权限。在仓库根目录执行：

```bash
gh auth login
gh release download data-20261114 --repo LulalaLucas/IsoCLIP-research --pattern '*.tar.gz' --pattern SHA256SUMS --pattern DATA_MANIFEST.json --dir .
sha256sum -c SHA256SUMS
tar -xzf image_features.tar.gz
tar -xzf image_noproj_features.tar.gz
tar -xzf text_features.tar.gz
tar -xzf text_noproj_features.tar.gz
```

单文件路径、大小与SHA-256见`manifest.json`，附件校验见Release中的`SHA256SUMS`。这些数据是已经编码的特征缓存；原始图像数据集和模型权重不在此包中。运行需要相应数据集元数据与配置中约定的输入，具体见源码及实验README。
