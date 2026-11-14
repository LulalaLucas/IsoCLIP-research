# 真实实验环境

记录来自实际使用的`isoclip`环境。Python3.10.14、PyTorch2.1.1+cu121、torchvision0.16.1+cu121，PyTorch CUDA运行时12.1。具体GPU与平台见`runtime.json`。

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate isoclip
python -m pip install -r environment/requirements.lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

`requirements.lock.txt`来自当前完整pip冻结结果；原来CLIP／Dassl的本地安装路径已转换为其Git提交固定的公开源码地址。当前项目单独editable安装，避免保留原服务器路径。CUDA版本指PyTorch运行时，不要求安装与驱动显示完全相同版本的独立CUDA工具包。

这里上传的是可重建的环境定义与完整依赖，不是带绝对路径的整个虚拟环境目录。
