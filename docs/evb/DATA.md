# EV-B数据来源与保管

## 来源与署名

EnergyVille / KU Leuven发布的 **BEV energy dynamics dataset**：
- 数据DOI：[10.48804/8KPDTW](https://doi.org/10.48804/8KPDTW)。
- 原论文：[Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data](https://doi.org/10.1038/s41597-025-06148-5)。
- 作者：Mohamed Aboubacar Yasko、Adamou Moussa Issaka、Fanghao Tian、Hussain Syed Kazmi、Johan Driesen、Wilmar Martinez。
- 本项目2026-09-30获取的V2官方元数据记载CC BY 4.0；[原元数据](../../reports/public_dataset_audit_20260930/sources/energyville_metadata.json)及[原README](../../reports/public_dataset_audit_20260930/sources/energyville_readme.txt)保留来源与许可。
- 处理修改包括会话质量标记、温度截断、同步计数器净量、IV补边、因果历史、特征和实验划分，不能把派生标签描述为原作者认证组件真值。

数据许可属于原始数据及其派生材料，不能因此推定仓库全部自编代码也自动授予同一许可；本次没有额外选择全仓库开源许可证。

## 原始包与校验

官方V2文件名BEV energy dynamic data_V2.zip，官方file ID 272593，162,310,975字节。下载入口：

https://rdr.kuleuven.be/api/access/datafile/272593

本项目以energyville_V2.zip保存至：

    reports/public_dataset_audit_20260930/sources/energyville_V2.zip

校验：
- MD5：cc2d63758b411c0f8d4b61bdaa1fff1c
- SHA256：97bcf4b914cd7312a08be49d53ae704a2450d39faa64d0c94d1294cd8eb92016

原发布件是1秒均值CSV，不是全量原始CAN，仅有一个raw示例。公开发布原始值保留、缺口不臆造，首行缺包温不作为标签拒收条件。

## Git归档范围

| 材料 | Git | 说明 |
|---|---|---|
| 自编适配/训练/验收代码、测试、研究Markdown | 保存 | 可阅读、可维护 |
| 协议、schema、哈希、配置、指标和审计JSON | 保存 | 保持当时口径 |
| EV-B逐行预测、特征/历史来源、分层与图 | 保存 | 派生评测材料，引用以上原数据 |
| 原始V2 ZIP、重建信号ZIP/GZ、R05批量CSV | 本地/云端 | 获取原包后运行重建；不塞入Git大对象 |
| 模型joblib、训练日志及初版重放ZIP | 本地/云端 | 新clone需新目录重新拟合或另复制已验收权重 |
| 论文PDF/全文、第三方克隆库、云SSH配置/密钥 | 不上传 | 文献证据与来源链接保留 |

原manifest仍会列出本地权重/大数据的hash。它证明当时执行链路，不意味着这些文件全部在Git。不要修改manifest删掉未上传条目来制造“完整验收”。

## 标签边界与信息边界

净电量以counter放电−充电为主，source边缘用有观测IV补边。EnergyVille放电电流方向为正；不得套旧VED的负号。标签资格、起点信息资格和业务时点分别管理。

1272为首温后预算候选，不是1272辆车。源记录不保证真实ON/OFF，GPS不是真正出发导航快照，两辆车不支持车队普适结论。计数器/IV内部一致性不构成独立电表标定。

EV-B02输出延迟至cut+1秒、状态仍截至cut；125校准终局量尚未用于模型、历史或打分。开发193条已看过，不重命名成新的独立测试。

## 重建说明

流程与命令见[复现指南](REPRODUCTION.md)，字段与诊断见[R05处理说明](../../reports/energyville_label_rebuild_20260930/data_pipeline/数据处理说明.md)、[标签合同](../../reports/energyville_label_rebuild_20260930/label_audit/推荐标签契约.md)。
