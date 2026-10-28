# 猫狗图像分类实验：从搭建模型到部署上线

以猫狗二分类为例，完整走一遍 **搭模型 → 训练 → 测试 → 算指标 → 保存权重 → 封装成 Web 应用** 的全流程。
数据集：`datasets/` 下 100 张图片（猫 50 / 狗 50），标签藏在文件名里。

---

## 1. 快速开始

```bash
# ① 安装依赖（CPU 版 PyTorch）
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# ② 一键跑完第一~第四步所有实验（CPU 约 5~10 分钟）
python scripts/run_all.py

# ②b 获取部署模型：仓库为轻量化不含大模型权重，请从 GitHub Releases（v1.0）
#     下载 step4_mobilenet_v3_small.pth（约 6MB）放到 outputs/checkpoints/，
#     这样无需训练即可直接启动应用；或跳过下载自己训练（仅几秒）：
#     python scripts/train.py --arch mobilenet_v3_small --epochs 5

# ③ 启动 Web 应用，浏览器打开 http://127.0.0.1:8000
python backend/server.py

# ④ （可选）手机连同一 Wi-Fi 拍照测试
python backend/server.py --lan      # 然后手机访问 http://<本机IP>:8000
```

Windows 用户本机也可以直接双击目录里的 `一键训练.bat` / `启动服务.bat`（这两个是**本地快捷脚本**，硬编码了本机 Python 环境路径，未纳入仓库、已写在 `.gitignore` 里；克隆到别的机器请用上面的 `python` 命令）。

只想最快看到效果？两步就够：

```bash
python scripts/train.py --arch resnet18 --epochs 5     # 约 40 秒
python backend/server.py
```

---

## 2. 项目结构

```
catdog-lab/
├── datasets/                 100 张图片（cat.x.jpg / dog.x.jpg）
├── src/catdog/               核心库（分层，无脚本里堆代码）
│   ├── config.py             全局配置：路径、超参、类别、归一化系数
│   ├── data.py               数据集 / 分层划分 / 变换 / DataLoader
│   ├── models.py             手写 AlexNet + torchvision 模型工厂
│   ├── metrics.py            准确率 / 精确率 / 召回率 / F1 / 混淆矩阵 / AUC
│   ├── engine.py             训练循环、评估、最佳权重保存、延迟测量
│   ├── inference.py          Predictor：面向应用的单图推理
│   ├── pipeline.py           实验流水线（脚本与后端共用同一入口）
│   └── utils.py              随机种子、设备、计时、日志
├── scripts/                  命令行实验脚本
│   ├── step1_scratch.py      第一步：手写 AlexNet 从零训练
│   ├── step2_pretrained.py   第二步：加载 ImageNet 预训练权重（冻结 / 微调对比）
│   ├── step3_augmentation.py 第三步：数据增强三组对照
│   ├── step4_model_zoo.py    第四步：torchvision 模型横向对比
│   ├── step5_predict.py      第五步：单张 / 批量图片推理测试
│   ├── train.py              通用训练入口
│   └── run_all.py            一键跑完 1~4 步
├── backend/                  FastAPI 后端
│   ├── server.py             HTTP 路由
│   ├── service.py            模型仓库 + 训练任务管理
│   └── schemas.py            请求 / 响应数据模型
├── web/                      原生前端（无框架、无 CDN、可离线）
│   ├── index.html
│   └── assets/{style.css,app.js,favicon.svg}
├── outputs/                  实验产物（自动生成）
│   ├── checkpoints/          权重（不随仓库；见 GitHub Releases v1.0，或自行训练生成）
│   │                         + 同名 .meta.json 元信息（轻量，已随仓库）
│   ├── metrics/*.json        每组实验的指标与逐轮曲线
│   └── logs/*.log            运行日志
└── docs/实验报告.md           含真实数据的实验报告
```

**为什么这么分层**：CLI 脚本和 Web 后端都调用 `pipeline.run_experiment()`，
所以页面上训练出来的模型，和命令行跑出来的必然是同一套代码路径下的产物，不会出现"两份实现各跑各的"。

---

## 3. 六步实验内容

| 步骤 | 做什么 | 入口 |
|---|---|---|
| 第一步 | 手写 AlexNet，随机初始化，从零训练，建立基线 | `scripts/step1_scratch.py` |
| 第二步 | 灌入 ImageNet 预训练权重；冻结卷积 vs 全量微调 | `scripts/step2_pretrained.py --finetune` |
| 第三步 | 数据增强三组对照（从零/预训练 × 有无增强） | `scripts/step3_augmentation.py` |
| 第四步 | resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0 横向对比 | `scripts/step4_model_zoo.py` |
| 第五步 | 保存最佳权重（只存 best）+ 单图 / 批量推理 | `scripts/step5_predict.py` |
| 第六步 | 封装成**完整前后端**应用（FastAPI + 原生前端） | `backend/server.py` |

> 第六步在指导书的 Gradio 方案上做了升级：Gradio 是"给模型套壳"，
> 本项目给出的是**真正的前后端分离**——后端提供 REST 接口，前端是独立页面，
> 换成微信小程序 / App / 其它语言前端时，只要调 `/api/predict` 即可，后端一行不用改。

### 控制变量（保证对比公平）

所有实验共用：①同一份数据划分（`seed=42`，分层抽样，80 训练 / 20 验证）；
②同一套评价指标（accuracy / precision / recall / F1 / AUC，macro 平均）；
③同一预处理（Resize 224 + ImageNet 归一化）。每次只改一个变量。

---

## 4. 实验结果（本机 CPU 实跑）

| 实验 | 架构 | 准确率 | 精确率 | 召回率 | F1 | AUC | 训练耗时 |
|---|---|---|---|---|---|---|---|
| step1 从零训练 | alexnet_scratch | 0.8500 | 0.8535 | 0.8500 | 0.8496 | 0.8000 | 71.6s |
| step2 预训练·冻结 | alexnet_manual | **0.9500** | 0.9545 | 0.9500 | 0.9499 | 1.0000 | 10.0s |
| step2 预训练·微调 | alexnet_manual | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9400 | 13.2s |
| step3a 从零·无增强 | resnet18 | 0.7500 | 0.7747 | 0.7500 | 0.7442 | 0.7500 | 29.4s |
| step3b 从零·有增强 | resnet18 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.7500 | 32.4s |
| step3c 预训练·有增强 | resnet18 | **0.9500** | 0.9545 | 0.9500 | 0.9499 | 0.9800 | 14.4s |
| step4 resnet18 | resnet18 | **0.9500** | 0.9545 | 0.9500 | 0.9499 | 0.9800 | 6.6s |
| step4 mobilenet_v3_small | mobilenet_v3_small | **0.9500** | 0.9545 | 0.9500 | 0.9499 | 0.9500 | 2.9s |
| step4 shufflenet_v2_x1_0 | shufflenet_v2_x1_0 | 0.8000 | 0.8571 | 0.8000 | 0.7917 | 0.9800 | 3.3s |

第四步的工程指标（选型关键）：

| 模型 | 参数量 | 权重体积 | 单图推理延迟 |
|---|---|---|---|
| resnet18 | 11,177,538 | 42.71 MB | 19.68 ms |
| **mobilenet_v3_small** | **1,519,906** | **5.93 MB** | **13.14 ms** |
| shufflenet_v2_x1_0 | 1,255,654 | 4.96 MB | 18.00 ms |

**结论**：准确率和 resnet18 打平（都是 0.95），但 mobilenet_v3_small 体积只有 1/7、推理快 33%，
是部署到手机 / 浏览器 / 小程序的最优选择。小数据集上"知识从哪来"比"训练多久"重要得多。

---

## 5. 后端接口

启动后访问 http://127.0.0.1:8000/api/docs 可看到自动生成的交互式文档。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务状态、设备、当前模型 |
| GET | `/api/models` | 权重清单（准确率 / 体积 / 创建时间） |
| POST | `/api/models/active` | 切换当前使用的模型 |
| POST | `/api/predict` | 上传图片识别（multipart/form-data） |
| POST | `/api/predict/base64` | base64 图片识别（摄像头拍照用） |
| GET | `/api/dataset/stats` | 数据集统计与样本列表 |
| GET | `/api/dataset/image/{name}` | 读取数据集图片 |
| GET | `/api/experiments` | 全部实验记录（指标 + 逐轮曲线） |
| POST | `/api/train` | 启动后台训练任务 |
| GET | `/api/train/status` | 查询训练进度（前端每 0.7s 轮询） |
| GET | `/api/archs` | 可用模型架构清单 |

调用示例：

```bash
curl -X POST -F "file=@datasets/cat.3.jpg" http://127.0.0.1:8000/api/predict
# {"label":"cat","label_cn":"猫","confidence":0.65,"probabilities":{"cat":0.65,"dog":0.35},"inference_ms":18.9,...}

curl -X POST http://127.0.0.1:8000/api/train -H "Content-Type: application/json" \
     -d '{"arch":"resnet18","epochs":5,"augment":true}'
```

改成微信小程序只需前端替换：`wx.uploadFile({ url: 'https://xxx/api/predict', filePath, name: 'file' })`。

---

## 6. 前端功能

| 页签 | 能力 |
|---|---|
| 🔍 识别 | 点击/拖放上传、摄像头拍照、数据集快速取样；概率条、置信度、推理耗时、与真实标签比对 |
| 🎯 训练 | 选架构/轮数/学习率/增强，实时进度条 + loss 与准确率曲线，结束后展示四项指标 |
| 📊 实验对比 | 所有实验的准确率条形图 + 完整指标表（自动读取 `outputs/metrics/*.json`） |
| 🖼️ 数据集 | 100 张缩略图，点击任意一张直接送模型识别并比对真实标签 |

前端为原生 HTML/CSS/JS，**不依赖任何 CDN**，断网也能用。

---

## 7. 常见问题

**Q：提示"尚未加载模型"？**
先训练一次：`python scripts/train.py --arch resnet18 --epochs 5`。

**Q：想让准确率更高？**
`python scripts/train.py --arch resnet18 --epochs 20 --augment --lr 5e-4`
（注意：验证集只有 20 张，涨 1 张就是 ±5%，不必过度调参。）

**Q：权重文件是什么格式？**
`outputs/checkpoints/*.pth` 是带元信息的字典：`{"arch", "classes", "state_dict", "config", "metrics", ...}`，
同目录还有一份 `.meta.json` 供服务端快速读取。用 `catdog.inference.load_predictor()` 一行加载即可。

**Q：摄像头打不开？**
浏览器要求 `localhost` 或 HTTPS 才能调用摄像头。用 `http://127.0.0.1:8000` 即可；手机访问需 HTTPS。

**Q：`outputs/` 占了 800MB，正常吗？**

正常。9 个权重里 AlexNet 的三个就有 615MB（它的两个 4096 维全连接层参数极多）。
真正要部署的是 `step4_mobilenet_v3_small.pth`（5.93 MB），已作为
[GitHub Release v1.0](https://github.com/jyz1111373/catdog-lab/releases/tag/v1.0) 附件提供；
克隆后下载放到 `outputs/checkpoints/` 即可直接启动应用（无需训练）。
若想瘦身，删掉下面三个即可，指标和日志都保留在 `outputs/metrics/` 与 `outputs/logs/` 里，不影响报告：

```bash
rm outputs/checkpoints/step1_scratch.pth
rm outputs/checkpoints/step2_pretrained_frozen.pth
rm outputs/checkpoints/step2_pretrained_finetune.pth
```

**Q：想换自己的数据？**
把图片放进 `datasets/`，文件名以 `cat`/`dog` 开头即可；改 `src/catdog/config.py` 里的 `CLASSES` 就能换类别。
