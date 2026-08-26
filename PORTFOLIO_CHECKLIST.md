# 项目发布检查清单

## 必须保留

- `README.md`
- `app/pipeline.py`
- `app/gradio_app.py`
- `models/qwen_diagnoser.py`
- 数据构建、训练和评测脚本
- `tools/video_denoiser.py`
- 两张经过筛选的定性结果图
- 数据划分与实验指标说明

## 不要提交

- 原始数据集
- `.pth`、`.pt`、`.safetensors`等模型权重
- Hugging Face缓存
- QLoRA checkpoint目录
- 全量复原输出
- Gradio运行目录
- Token、密码、代理地址和其他私密配置

## 发布前检查

```bash
git status --short
git check-ignore -v outputs results data
rg -n "HF_TOKEN|api[_-]?key|password|secret" app models scripts tools
rg -n "/home/xyl|C:\\\\Users" app models scripts tools README.md
```

绝对路径检查命中README示例并不一定是问题；代码中若出现个人绝对路径，应改成命令行参数或配置项。

## 推荐提交内容

```text
multimodal-restoration/
├── README.md
├── .gitignore
├── assets/
├── app/
├── models/
├── scripts/
└── tools/
```

## 推荐提交信息

```text
feat: add Qwen2.5-VL video restoration agent and evaluation pipeline
```
