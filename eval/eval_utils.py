# -*- coding: utf-8 -*-
import math
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, List

import os
import gc
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_json_to_graph(data: Dict[str, Any]) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
    G = nx.Graph()
    pos = {}
    for node in data.get("nodes", []):
        node_id = str(node["id"])
        G.add_node(node_id)
        pos[node_id] = np.array([float(node["x"]), float(node["y"])])
    for edge in data.get("edges", []):
        u, v = str(edge["source"]), str(edge["target"])
        weight = float(edge.get("weight", 1.0))
        if u in pos and v in pos:
            G.add_edge(u, v, weight=weight)
    return G, pos

def calculate_stress(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    if G.number_of_nodes() < 2:
        return 0.0

    path_lengths = dict(nx.shortest_path_length(G))
    nodes = list(G.nodes())
    n = len(nodes)
    
    numerator = 0.0
    denominator = 0.0
    pairs_data = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            if u not in path_lengths or v not in path_lengths[u]:
                continue
            d_ij = float(path_lengths[u][v])
            if d_ij == 0:
                continue
                
            dist_euclidean = np.linalg.norm(pos[u] - pos[v])
            w_ij = 1.0 / (d_ij ** 2) 
            
            numerator += w_ij * d_ij * dist_euclidean
            denominator += w_ij * (dist_euclidean ** 2)
            pairs_data.append((d_ij, dist_euclidean, w_ij))
            
    s = (numerator / denominator) if denominator > 0 else 1.0
    
    stress_loss = 0.0
    for d_ij, dist_euclidean, w_ij in pairs_data:
        scaled_dist = s * dist_euclidean
        stress_loss += w_ij * ((scaled_dist - d_ij) ** 2)
            
    return stress_loss / (n * (n - 1) / 2)

#def calculate_vertex_resolution(G: nx.Graph, pos: dict) -> float:
#
#    nodes = list(G.nodes())
#    n = len(nodes)
#    if n < 2: 
#        return 0.0
#        
#    coords = np.array([pos[node] for node in nodes])
#    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
#    dist_sq = np.sum(diff ** 2, axis=-1)
#    dist = np.sqrt(np.maximum(dist_sq, 0.0))
#    
#    np.fill_diagonal(dist, np.inf)
#    min_dist = np.min(dist)
#    dist_matrix_no_inf = np.where(dist == np.inf, 0, dist)
#    d_max = np.max(dist_matrix_no_inf)
#    
#    if d_max < 1e-6: 
#        return 1.0
#        
#    r = 1.0 / math.sqrt(n)
#    q_vr = min_dist / (r * d_max)
#    
#    vr_score = min(1.0, max(0.0, q_vr))
#    
#    return 1.0 - vr_score



def calculate_vertex_resolution(G: nx.Graph, pos: dict) -> float:
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2: 
        return 0.0
        
    coords = np.array([pos[node] for node in nodes])
    
    # 利用 NumPy 广播机制计算全矩阵欧氏距离
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    
    # 提取非对角线元素（即 i != j 的所有情况）
    # 创建一个布尔掩码，对角线为 False，其他为 True
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dist[mask]
    
    # 计算所有非对角线节点对的欧氏距离平均值
    avg_dist = np.mean(valid_dists)
    
    # 避免所有节点重叠导致平均距离为 0 的除零错误
    if avg_dist < 1e-8:
        # 如果距离全为0，e^(-0) = 1，共 n*(n-1) 个项
        return float(n * (n - 1))
        
    # 计算归一化系数 \beta
    beta = 1.0 / avg_dist
    
    # 计算指数惩罚 e^{-\beta * ||xi - xj||} 并求和
    m_no = np.sum(np.exp(-beta * valid_dists))/(n * (n - 1))
    
    return float(m_no)

#def calculate_angular_resolution(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
#    if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
#        return 0.0 
#        
#    degrees = dict(G.degree())
#    d_max = max(degrees.values())
#    if d_max < 2:
#        return 0.0
#        
#    min_angle_global = float('inf')
#    
#    for node in G.nodes():
#        neighbors = list(G.neighbors(node))
#        if len(neighbors) < 2:
#            continue
#            
#        angles = []
#        center_pos = pos[node]
#        for neighbor in neighbors:
#            vec = pos[neighbor] - center_pos
#            if np.linalg.norm(vec) == 0:
#                continue
#            angle = math.atan2(vec[1], vec[0])
#            angles.append(angle)
#            
#        if len(angles) < 2:
#            continue
#            
#        angles.sort()
#        min_angle_local = min(angles[i] - angles[i-1] for i in range(1, len(angles)))
#        min_angle_local = min(min_angle_local, angles[0] - angles[-1] + 2 * math.pi)
#        
#        min_angle_global = min(min_angle_global, min_angle_local)
#        
#    if min_angle_global == float('inf'):
#        return 0.0
#        
#    ideal_angle = 2 * math.pi / d_max
#    q_anr = min_angle_global / ideal_angle if ideal_angle > 0 else 0.0
#    q_anr = min(1.0, max(0.0, q_anr))
#    
#    return 1.0 - q_anr

def calculate_angular_resolution(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0
        
    penalty_sum = 0.0
    
    for node in nodes:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        
        # 如果度数小于2，不存在邻接角，此时误差为0，不做累加
        if degree < 2:
            continue
            
        angles = []
        center_pos = pos[node]
        for neighbor in neighbors:
            vec = pos[neighbor] - center_pos
            if np.linalg.norm(vec) == 0:
                continue
            # 计算以该节点为中心的边向量的角度 [-pi, pi]
            angle = math.atan2(vec[1], vec[0])
            angles.append(angle)
            
        if len(angles) < 2:
            continue
            
        # 对角度排序以计算相邻边夹角
        angles.sort()
        
        # 计算所有相邻角度差，取最小值
        min_angle_local = min(angles[i] - angles[i-1] for i in range(1, len(angles)))
        # 计算首尾两条边的夹角 (跨越 360度的夹角)
        min_angle_local = min(min_angle_local, angles[0] - angles[-1] + 2 * math.pi)
        
        theta_i = min_angle_local
        # 局部理想邻接角：360度平分给该节点的所有边
        theta_min = 2 * math.pi / degree
        
        penalty = abs((theta_i - theta_min) / theta_min)
        
        penalty_sum += penalty
        
    # 根据公式计算最终得分
    m_a = 1.0 - (penalty_sum / n)
    
    return float(m_a)

def _do_segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def calculate_crossing_number(G: nx.Graph, pos: Dict[str, np.ndarray]) -> int:
    edges = list(G.edges())
    cross_count = 0
    num_edges = len(edges)
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            u, v = edges[i]
            x, y = edges[j]
            if len({u, v, x, y}) < 4:
                continue
            if _do_segments_intersect(pos[u], pos[v], pos[x], pos[y]):
                cross_count += 1
    return cross_count
    
def calculate_crossing_metric(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    edges = list(G.edges())
    num_edges = len(edges)
    
    # 边界情况：少于2条边不可能有交叉
    if num_edges < 2:
        return 1.0
        
    # 1. 计算实际边交叉数量 c (保留你原有的逻辑)
    c = 0
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            u, v = edges[i]
            x, y = edges[j]
            # 排除共用顶点的边对
            if len({u, v, x, y}) < 4:
                continue
            if _do_segments_intersect(pos[u], pos[v], pos[x], pos[y]):
                c += 1
                
    # 计算所有边对的总组合数: |E|(|E|-1)/2
    total_edge_pairs = (num_edges * (num_edges - 1)) / 2.0
    
    # 计算所有共用顶点的边对组合数
    shared_vertex_pairs = 0.0
    for node in G.nodes():
        degree = G.degree(node)
        if degree >= 2:
            shared_vertex_pairs += (degree * (degree - 1))
    shared_vertex_pairs = shared_vertex_pairs / 2.0
    
    # 相减得到最大理论交叉数
    c_max = total_edge_pairs - shared_vertex_pairs
    
    # 3. 计算最终的归一化指标 m_c (根据公式 2.1)
    if c_max > 0:
        m_c = 1.0 - (c / c_max)
    else:
        m_c = 1.0
        
    return float(m_c)



def calculate_neighborhood_preservation(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2 or G.number_of_edges() == 0:
        return 0.0
        
    # 提取坐标矩阵
    X = np.array([pos[node] for node in nodes])
    
    # 【性能优化】一次性利用矩阵广播计算全局距离矩阵
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist_matrix = np.sqrt(np.maximum(dist_sq, 0.0))
    
    # 将自身到自身的距离设为无穷大，确保在寻找最近邻时不会选中自己
    np.fill_diagonal(dist_matrix, np.inf)
    
    jaccard_sum = 0.0
    valid_nodes = 0
    
    for i in range(n):
        node_i = nodes[i]
        degree_i = G.degree(node_i)
        
        # 排除孤立节点
        if degree_i == 0:
            continue
            
        # NG(i): 图空间中的真实邻居
        adj_neighbors = set(G.neighbors(node_i))
        
        # NL(i): 布局空间中的 degree_i 个最近邻
        # argsort 返回从小到大的索引，直接切片取前 degree_i 个
        closest_indices = np.argsort(dist_matrix[i])[:degree_i]
        spatial_neighbors = {nodes[idx] for idx in closest_indices}
        
        intersection = len(adj_neighbors.intersection(spatial_neighbors))
        union = len(adj_neighbors.union(spatial_neighbors))
        
        if union > 0:
            jaccard_sum += intersection / union
            
        valid_nodes += 1
        
    if valid_nodes == 0:
        return 0.0
        
    # 计算平均杰卡德相似度 m_np
    avg_jaccard = jaccard_sum / valid_nodes
    
    # 修正：直接返回相似度得分，不再使用 1.0 - 
    return float(avg_jaccard)

def calculate_edge_length_distribution(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    #边长一致性 
    edges = list(G.edges())
    num_edges = len(edges)
    
    # 边界条件：若边数少于2，无法比较分布差异
    if num_edges < 2:
        return 0.0
        
    # 【性能优化】：利用 NumPy 向量化计算所有边的欧氏距离，替代 for 循环
    # 将边列表转换为 NumPy 数组以便于切片
    edge_array = np.array(edges)
    
    # 批量提取起点和终点的坐标矩阵
    # 假设 pos 字典中存储的坐标是 np.ndarray 类型
    pos_u = np.array([pos[u] for u in edge_array[:, 0]])
    pos_v = np.array([pos[v] for v in edge_array[:, 1]])
    
    # 沿 axis=1 批量计算所有边向量的欧氏距离
    edge_lengths = np.linalg.norm(pos_u - pos_v, axis=1)
    
    # 计算平均边长 (l_mu) - 对应公式中的均值
    l_mu = np.mean(edge_lengths)
    
    # 边界条件：如果平均边长为0，变异系数无意义
    if l_mu == 0:
        return 1.0  
        
    # 计算边长的标准差
    l_std = np.std(edge_lengths)
    
    # 计算变异系数 l_cv
    l_cv = l_std / l_mu
    
    # 归一化处理得到最终指标 m_e
    m_e = l_cv / np.sqrt(num_edges - 1)
    
    # 截断处理以防浮点数精度误差溢出
    return float(np.clip(m_e, 0.0, 1.0))

      
    

def convert_case_format(case_data: Dict[str, Any], graph_name: str = "converted_case_graph") -> Dict[str, Any]:
    standard_data = {
        "graph_id": graph_name,
        "nodes": [],
        "edges": []
    }
    for node in case_data.get("nodes", []):
        node_id = str(node.get("id", node.get("name", "")))
        if node_id:
            standard_data["nodes"].append({"id": node_id})
    for link in case_data.get("links", []):
        if "source" in link and "target" in link:
            weight = float(link.get("weight", 1.0))
            standard_data["edges"].append({
                "source": str(link["source"]),
                "target": str(link["target"]),
                "weight": weight
            })
    return standard_data

def convert_graphml_format(filepath: str, graph_name: str = "graphml_case") -> dict:
    try:
        G_raw = nx.read_graphml(filepath)
    except Exception as e:
        raise ValueError(f"Failed to parse GraphML {filepath}: {e}")

    standard_data = {
        "graph_id": graph_name,
        "nodes": [],
        "edges": []
    }
    
    for node in G_raw.nodes():
        standard_data["nodes"].append({"id": str(node)})
        
    for u, v, edge_data in G_raw.edges(data=True):
        weight = float(edge_data.get("weight", 1.0))
        standard_data["edges"].append({
            "source": str(u),
            "target": str(v),
            "weight": weight
        })
        
    return standard_data
def _get_hubs(G: nx.Graph) -> list:
    nodes = list(G.nodes())
    if not nodes:
        return []
        
    degrees = np.array([G.degree(n) for n in nodes])
    
    if len(nodes) < 2:
        return nodes
        
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    threshold = mean_deg + std_deg
    
    # 按照统计学定义提取 Hubs
    hubs = [nodes[i] for i in range(len(nodes)) if degrees[i] > threshold]
    
    # 绝对防错：如果阈值筛选为空集，直接取度数最大的一批点
    if not hubs:
        max_deg = np.max(degrees)
        hubs = [nodes[i] for i in range(len(nodes)) if degrees[i] == max_deg]
        
    return hubs
def calculate_LAR(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    hubs = _get_hubs(G)
    if not hubs:
        return 0.0
        
    penalty_sum = 0.0
    valid_hubs = 0
    
    for node in hubs:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        
        if degree < 2:
            continue
            
        angles = []
        center_pos = pos[node]
        for neighbor in neighbors:
            vec = pos[neighbor] - center_pos
            if np.linalg.norm(vec) == 0:
                continue
            angle = math.atan2(vec[1], vec[0])
            angles.append(angle)
            
        if len(angles) < 2:
            continue
            
        angles.sort()
        min_angle_local = min(angles[i] - angles[i-1] for i in range(1, len(angles)))
        min_angle_local = min(min_angle_local, angles[0] - angles[-1] + 2 * math.pi)
        
        theta_i = min_angle_local
        theta_min = 2 * math.pi / degree
        
        # 采用严谨的理论误差相对值，避免边重合时的无穷大
        penalty = abs((theta_i - theta_min) / theta_min)
        penalty_sum += penalty
        valid_hubs += 1
        
    if valid_hubs == 0:
        return 1.0 # 如果所有hub度数都小于2，无角度偏差
        
    m_lar = 1.0 - (penalty_sum / valid_hubs)
    return float(m_lar)


def calculate_LEU(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    hubs = set(_get_hubs(G))
    if not hubs:
        return 0.0
        
    # 提取所有至少有一个端点是高度节点的边
    local_edges = [(u, v) for u, v in G.edges() if u in hubs or v in hubs]
    num_edges = len(local_edges)
    
    if num_edges < 2:
        return 0.0
        
    edge_array = np.array(local_edges)
    pos_u = np.array([pos[u] for u in edge_array[:, 0]])
    pos_v = np.array([pos[v] for v in edge_array[:, 1]])
    
    edge_lengths = np.linalg.norm(pos_u - pos_v, axis=1)
    
    l_mu = np.mean(edge_lengths)
    if l_mu == 0:
        return 1.0 
        
    l_std = np.std(edge_lengths)
    l_cv = l_std / l_mu
    
    m_leu = l_cv / np.sqrt(num_edges - 1)
    return float(np.clip(m_leu, 0.0, 1.0))


def calculate_LNP(G: nx.Graph, pos: Dict[str, np.ndarray]) -> float:
    hubs = _get_hubs(G)
    nodes = list(G.nodes())
    if not hubs or len(nodes) < 2:
        return 0.0
        
    # 预先提取全图坐标并建立索引，用于高效计算空间距离
    X_all = np.array([pos[node] for node in nodes])
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    jaccard_sum = 0.0
    valid_nodes = 0
    
    for hub in hubs:
        degree_i = G.degree(hub)
        if degree_i == 0:
            continue
            
        adj_neighbors = set(G.neighbors(hub))
        hub_idx = node_to_idx[hub]
        
        # 仅计算当前 hub 到全图所有节点的距离，复杂度 O(N) 而非 O(N^2)
        distances = np.linalg.norm(X_all - X_all[hub_idx], axis=1)
        # 将自身距离设为无穷大
        distances[hub_idx] = np.inf
        
        closest_indices = np.argsort(distances)[:degree_i]
        spatial_neighbors = {nodes[idx] for idx in closest_indices}
        
        intersection = len(adj_neighbors.intersection(spatial_neighbors))
        union = len(adj_neighbors.union(spatial_neighbors))
        
        if union > 0:
            jaccard_sum += intersection / union
            
        valid_nodes += 1
        
    if valid_nodes == 0:
        return 0.0
        
    m_lnp = jaccard_sum / valid_nodes
    return float(m_lnp)
# ==========================================
# Dynamic Export and Visualization Tools
# ==========================================

def export_evaluation_csv(records: list, output_path: str, metrics_cols: list, group_by_col: str = "Algorithm", aggregate: bool = False):
    """
    Exports evaluation records to a CSV file dynamically.
    
    :param records: List of flat dictionaries containing evaluation data.
    :param output_path: Path to save the CSV.
    :param metrics_cols: List of string keys representing the metrics to output/aggregate.
    :param group_by_col: The column name used to group data (e.g., "Algorithm" or "Model_Name").
    :param aggregate: If False, exports raw data. If True, exports Mean +/- Std grouped by group_by_col.
    """
    if not records:
        print("[Warning] No records to export.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    
    if not aggregate:
        # Raw export (Like Script 1)
        columns_to_export = ["Graph_ID", group_by_col] + metrics_cols
        # Ensure only existing columns are exported to prevent KeyError
        columns_to_export = [col for col in columns_to_export if col in df.columns]
        
        if "Graph_ID" in df.columns and group_by_col in df.columns:
            df = df.sort_values(by=["Graph_ID", group_by_col])
            
        df[columns_to_export].to_csv(output_path, index=False, encoding='utf-8')
    else:
        # Aggregated Academic Export (Like Script 2)
        if group_by_col not in df.columns:
            print(f"[Error] Grouping column '{group_by_col}' not found in records.")
            return
            
        valid_metrics = [col for col in metrics_cols if col in df.columns]
        grouped = df.groupby(group_by_col)[valid_metrics].agg(['mean', 'std'])
        report_df = pd.DataFrame(index=grouped.index)
        
        for col in valid_metrics:
            report_df[col] = grouped[col].apply(
                lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}" if pd.notnull(row['std']) else f"{row['mean']:.4f} +/- 0.0000",
                axis=1
            )
        report_df.to_csv(output_path, encoding='utf-8')


#def generate_comparison_png(layouts_info: list, output_path: str, metrics_config: list):
#    """
#    Generates a grid PNG comparing multiple graph layouts with dynamic metrics.
#    Includes continuous colormap based on node degrees.
#    """
#    import os
#    import gc
#    import math
#    import numpy as np
#    import networkx as nx
#    import matplotlib.pyplot as plt
#
#    num_algos = len(layouts_info)
#    if num_algos == 0:
#        return
#        
#    os.makedirs(os.path.dirname(output_path), exist_ok=True)
#        
#    cols = min(4, num_algos)
#    rows = math.ceil(num_algos / cols)
#    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
#    
#    if num_algos == 1:
#        axes = [axes]
#    else:
#        axes = axes.flatten()
#        
#    for i, info in enumerate(layouts_info):
#        ax = axes[i]
#        algo_name = info.get("name", f"Layout {i}")
#        G = info.get("G")
#        pos = info.get("pos")
#        metrics = info.get("metrics", {})
#        
#        ax.set_title(algo_name, fontsize=14, fontweight='bold')
#        
#        if G and pos and G.number_of_nodes() > 0:
#            # ==========================================
#            # Continuous Colormap based on Degree
#            # ==========================================
#            # 提取度数列表，必须与 G.nodes() 的遍历顺序绝对一致
#            degrees = [G.degree(n) for n in G.nodes()]
#            
#            # 先画边，放在最底层
#            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', alpha=0.6)
#            
#            # 再画节点，映射 Viridis 色图
#            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, 
#                                   node_color=degrees, cmap=plt.cm.viridis, alpha=0.9)
#            
#            # ==========================================
#            # Skybox Padding (Anti-overlap)
#            # ==========================================
#            y_coords = [coords[1] for coords in pos.values()]
#            x_coords = [coords[0] for coords in pos.values()]
#            
#            if y_coords and x_coords:
#                y_min, y_max = min(y_coords), max(y_coords)
#                x_min, x_max = min(x_coords), max(x_coords)
#                
#                h = y_max - y_min
#                w = x_max - x_min
#                if h == 0: h = 1.0
#                if w == 0: w = 1.0
#                
#                # Add 35% empty space at the top of the Y-axis for the text box
#                ax.set_ylim(y_min - 0.05 * h, y_max + 0.35 * h)
#                ax.set_xlim(x_min - 0.05 * w, x_max + 0.05 * w)
#        
#        # ==========================================
#        # Dynamic Text Box Rendering
#        # ==========================================
#        if metrics and metrics_config:
#            text_lines = []
#            for key, label, fmt in metrics_config:
#                val = metrics.get(key)
#                if val is not None:
#                    formatted_val = fmt.format(val)
#                    text_lines.append(f"{label}: {formatted_val}")
#                    
#            if text_lines:
#                textstr = '\n'.join(text_lines)
#                props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
#                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
#                        verticalalignment='top', bbox=props, family='monospace')
#                        
#        ax.axis('off')
#
#    # Hide unused subplots
#    for j in range(num_algos, len(axes)):
#        axes[j].axis('off')
#
#    plt.tight_layout()
#    plt.savefig(output_path, dpi=300, bbox_inches='tight')
#    
#    # Strict Memory Management
#    fig.clf()
#    plt.clf()
#    plt.close('all')
#    gc.collect()


def generate_comparison_png(layouts_info: list, output_path: str, metrics_config: list):
    """
    Generates a grid PNG comparing multiple graph layouts with dynamic metrics.
    Separates Hubs (Red) and Non-Hubs (Blue) to avoid muddy color transitions.
    Node size is dynamically adjusted based on the total number of nodes.
    """
    import os
    import gc
    import math
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    num_algos = len(layouts_info)
    if num_algos == 0:
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    cols = min(4, num_algos)
    rows = math.ceil(num_algos / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    
    if num_algos == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, info in enumerate(layouts_info):
        ax = axes[i]
        algo_name = info.get("name", f"Layout {i}")
        G = info.get("G")
        pos = info.get("pos")
        metrics = info.get("metrics", {})
        
        ax.set_title(algo_name, fontsize=14, fontweight='bold')
        
        if G and pos and G.number_of_nodes() > 0:
            
            # ==========================================
            # 0. 动态计算节点大小
            # ==========================================
            num_nodes = G.number_of_nodes()
            if num_nodes < 50:
                dynamic_node_size = 200
            elif num_nodes <= 100:
                dynamic_node_size = 100
            else:
                dynamic_node_size = 50

            # ==========================================
            # 1. 拆分高度节点与低度节点
            # ==========================================
            hubs = _get_hubs(G)
            hubs_set = set(hubs)
            non_hubs = [n for n in G.nodes() if n not in hubs_set]
            
            hub_degrees = [G.degree(n) for n in hubs]
            non_hub_degrees = [G.degree(n) for n in non_hubs]
            
            # ==========================================
            # 2. 绘制边线 (底层)
            # ==========================================
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666666', width=1.5, alpha=0.8)
            
            # ==========================================
            # 3. 绘制低度节点 (Non-Hubs): 蓝 -> 浅蓝
            # ==========================================
            if non_hubs:
                cmap_blue = mcolors.LinearSegmentedColormap.from_list("blues", ["#3498db", "#85c1e9"])
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=non_hubs, node_size=dynamic_node_size,
                                       node_color=non_hub_degrees, cmap=cmap_blue, alpha=0.9)
                                       
            # ==========================================
            # 4. 绘制高度节点 (Hubs): 浅红 -> 深红
            # ==========================================
            if hubs:
                cmap_red = mcolors.LinearSegmentedColormap.from_list("reds", ["#f1948a", "#e74c3c", "#8b0000"])
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=hubs, node_size=dynamic_node_size,
                                       node_color=hub_degrees, cmap=cmap_red, alpha=0.9)
            
            # ==========================================
            # Skybox Padding (Anti-overlap)
            # ==========================================
            y_coords = [coords[1] for coords in pos.values()]
            x_coords = [coords[0] for coords in pos.values()]
            
            if y_coords and x_coords:
                y_min, y_max = min(y_coords), max(y_coords)
                x_min, x_max = min(x_coords), max(x_coords)
                
                h = y_max - y_min
                w = x_max - x_min
                if h == 0: h = 1.0
                if w == 0: w = 1.0
                
                ax.set_ylim(y_min - 0.05 * h, y_max + 0.35 * h)
                ax.set_xlim(x_min - 0.05 * w, x_max + 0.05 * w)
        
        # ==========================================
        # Dynamic Text Box Rendering
        # ==========================================
        if metrics and metrics_config:
            text_lines = []
            for key, label, fmt in metrics_config:
                val = metrics.get(key)
                if val is not None:
                    formatted_val = fmt.format(val)
                    text_lines.append(f"{label}: {formatted_val}")
                    
            if text_lines:
                textstr = '\n'.join(text_lines)
                props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=props, family='monospace')
                        
        ax.axis('off')

    for j in range(num_algos, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    fig.clf()
    plt.clf()
    plt.close('all')
    gc.collect()