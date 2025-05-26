1. The main code of IFDPC is in package "IFDPC" in IFDPC5, "DPC" is the original algorithm DPC.
2. The datasets and procession used in IFDPC is in package "data_process".
3. The experimental code is in package "IFDPC".
4. 将IFDPC5.py注释掉的内容加上（归一化部分），以及experiment.py的 1 改为 注释掉的指标计算。
5. We employs two cluster validity indices, both implemented via scikit-learn in the compute_SC function:
  Davies-Bouldin Index (DB)
    Direction: Lower values indicate better clustering.
    Implementation: sklearn.metrics.davies_bouldin_score
  Calinski-Harabasz Index (CH)
    Direction: Higher values indicate better clustering.
    Implementation: sklearn.metrics.calinski_harabasz_score
6. Boundary Point Similarity Calculation (SPC Function). The experimental function: "def compute_boundary_point_similarity(model, K):"
