1. The main code of IFDPC is in package "IFDPC" in IFDPC5, "DPC" is the original algorithm DPC.
2. The datasets and procession used in IFDPC is in package "data_process".
3. The experimental code is in package "IFDPC".
4. 将IFDPC5.py注释掉的内容加上（归一化部分），以及experiment.py的 1 改为 注释掉的指标计算。
5. 本文使用过两个聚类质量评估指标，DB指数与CH指数，都在compute_SC函数中用sklearn库实现。
