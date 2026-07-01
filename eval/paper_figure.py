# paper_figures.py
# Mimics batch_benchmark_pipeline_v2 patterns. User configures INPUT_DIR, OUTPUT_DIR,
# and DEEPCS_EXPERIMENTS below. Baselines (FR, KK, MDS, FA2, D3) are auto-loaded.

import os, sys, json, math, gc, glob, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import eval_utils as utils
from layout import LayoutGenerator

warnings.filterwarnings("ignore")

# ==================== USER CONFIG ====================
INPUT_DIR = "./figure_input"
OUTPUT_DIR = "/root/autodl-tmp/paper_figure"

DEEPCS_EXPERIMENTS = [
    "ablation_exp0_baseline",
    "kl_only", "lnp_only", "ar_only_gamma1.5", "eu_only",
    "ablation_exp4_gcn", "ablation_exp5_gin",
]

# =====================================================
# All graphs in INPUT_DIR will be processed. No hand-picked lists.
# =====================================================

BASELINE_ALGOS = ["fr", "kk", "mds", "fa2", "d3"]
BASELINE_DISPLAY = {"fr": "FR", "kk": "KK", "mds": "MDS",
                    "fa2": "ForceAtlas2", "d3": "D3-Force"}
DPI = 300


class PaperFigureGenerator:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.loaded_components = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def load_models(self):
        print(f"[INFO] Loading {len(DEEPCS_EXPERIMENTS)} DeepCS models...")
        project_root = os.path.dirname(current_dir)
        for exp_name in DEEPCS_EXPERIMENTS:
            cfg_path = os.path.join(project_root, "logs", exp_name, "config.yaml")
            ckpt_path = os.path.join(project_root, "logs", exp_name, "best_model.pth")
            if os.path.exists(cfg_path) and os.path.exists(ckpt_path):
                try:
                    LayoutGenerator.load_deepcs_model(cfg_path, ckpt_path, model_name=exp_name)
                    self.loaded_components[exp_name] = LayoutGenerator.deepcs_components
                    print(f"  [OK] {exp_name}")
                except Exception as e:
                    print(f"  [FAIL] {exp_name}: {e}")
            else:
                print(f"  [MISSING] {exp_name}")

    def _find_graph_file(self, graph_name):
        for ext in ['.json', '.graphml']:
            candidate = os.path.join(self.input_dir, graph_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_graph_data(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        graph_name = os.path.splitext(os.path.basename(filepath))[0]
        if ext == ".json":
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='gbk') as f:
                    raw_data = json.load(f)
            return utils.convert_case_format(raw_data, graph_name=graph_name)
        elif ext == ".graphml":
            return utils.convert_graphml_format(filepath, graph_name=graph_name)
        raise ValueError(f"Unknown format: {ext}")

    @staticmethod
    def _build_graph_ref(graph_data):
        G = nx.Graph()
        for node in graph_data.get("nodes", []):
            G.add_node(str(node["id"]))
        for edge in graph_data.get("edges", []):
            G.add_edge(str(edge["source"]), str(edge["target"]))
        return G

    def _generate_baseline(self, layout_gen, algo):
        func = getattr(layout_gen, f"generate_{algo}")
        layout_data = func()
        _, pos = utils.parse_json_to_graph(layout_data)
        return pos

    def _generate_deepcs(self, layout_gen, exp_name):
        LayoutGenerator.deepcs_components = self.loaded_components[exp_name]
        layout_data = layout_gen.generate_deepcs()
        _, pos = utils.parse_json_to_graph(layout_data)
        return pos

    def _render_grid(self, layout_infos, output_path, graph_name):
        """layout_infos: list of {"name": str, "G": nx.Graph, "pos": dict}"""
        n = len(layout_infos)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4.2))
        if n == 1:
            axes = [axes]

        for i, info in enumerate(layout_infos):
            ax = axes[i]
            ax.set_title(info["name"], fontsize=11, fontweight='bold')
            G = info["G"]
            pos = info["pos"]
            if G is None or pos is None or G.number_of_nodes() == 0:
                ax.axis('off')
                continue

            n_nodes = G.number_of_nodes()
            sz = 200 if n_nodes < 50 else (100 if n_nodes <= 100 else 40)
            hubs = set(utils._get_hubs(G))
            non_hubs = [n for n in G.nodes() if n not in hubs]

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666666',
                                   width=1.2, alpha=0.7)
            if non_hubs:
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=non_hubs,
                                       node_size=sz, node_color='#3498db', alpha=0.85)
            if hubs:
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=hubs,
                                       node_size=int(sz * 1.6),
                                       node_color='#e74c3c', alpha=0.9)

            x_vals = [c[0] for c in pos.values()]
            y_vals = [c[1] for c in pos.values()]
            if x_vals and y_vals:
                w, h = max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)
                ax.set_xlim(min(x_vals) - 0.08 * max(w, 1),
                            max(x_vals) + 0.08 * max(w, 1))
                ax.set_ylim(min(y_vals) - 0.08 * max(h, 1),
                            max(y_vals) + 0.08 * max(h, 1))
            ax.axis('off')

        for j in range(n, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  -> {output_path}")

    def _all_graphs(self):
        files = glob.glob(os.path.join(self.input_dir, "*.json"))
        files += glob.glob(os.path.join(self.input_dir, "*.graphml"))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def generate_fig51(self):
        graphs = self._all_graphs()
        print(f"\n=== Figure 5-1: Qualitative Comparison ({len(graphs)} graphs x 6 algorithms) ===")
        for gname in sorted(graphs):
            fpath = self._find_graph_file(gname)
            if not fpath:
                continue
            print(f"\n  {gname}")
            graph_data = self._load_graph_data(fpath)
            G_ref = self._build_graph_ref(graph_data)
            layout_gen = LayoutGenerator(graph_data)

            infos = []
            for algo in BASELINE_ALGOS:
                try:
                    pos = self._generate_baseline(layout_gen, algo)
                    infos.append({"name": BASELINE_DISPLAY[algo], "G": G_ref, "pos": pos})
                except Exception as e:
                    print(f"    [WARN] {algo}: {e}")

            try:
                pos = self._generate_deepcs(layout_gen, "ablation_exp0_baseline")
                infos.append({"name": "VRRGL (Ours)", "G": G_ref, "pos": pos})
            except Exception as e:
                print(f"    [WARN] VRRGL: {e}")

            out_path = os.path.join(self.output_dir, f"fig5-1_{gname}.png")
            self._render_grid(infos, out_path, gname)
            gc.collect()

    def generate_fig52(self):
        graphs = self._all_graphs()
        print(f"\n=== Figure 5-2: Loss Ablation ({len(graphs)} graphs x 5 configs) ===")
        ablation_names = ["kl_only", "lnp_only", "ar_only_gamma1.5", "eu_only",
                          "ablation_exp0_baseline"]
        ablation_display = {
            "kl_only": "KL only", "lnp_only": "LNP only",
            "ar_only_gamma1.5": "AR only", "eu_only": "EU only",
            "ablation_exp0_baseline": "Full (VRRGL)"
        }

        for gname in sorted(graphs):
            fpath = self._find_graph_file(gname)
            if not fpath:
                continue
            print(f"\n  {gname}")
            graph_data = self._load_graph_data(fpath)
            G_ref = self._build_graph_ref(graph_data)
            layout_gen = LayoutGenerator(graph_data)

            infos = []
            for exp_name in ablation_names:
                try:
                    pos = self._generate_deepcs(layout_gen, exp_name)
                    infos.append({"name": ablation_display[exp_name], "G": G_ref, "pos": pos})
                except Exception as e:
                    print(f"    [WARN] {exp_name}: {e}")

            out_path = os.path.join(self.output_dir, f"fig5-2_{gname}.png")
            self._render_grid(infos, out_path, gname)
            gc.collect()

    def generate_fig53(self):
        graphs = self._all_graphs()
        print(f"\n=== Figure 5-3: GNN Backbone Ablation ({len(graphs)} graphs x 3 GNNs) ===")
        gnn_names = ["ablation_exp0_baseline", "ablation_exp4_gcn", "ablation_exp5_gin"]
        gnn_display = {"ablation_exp0_baseline": "GATv2",
                       "ablation_exp4_gcn": "GCN",
                       "ablation_exp5_gin": "GIN"}

        for gname in sorted(graphs):
            fpath = self._find_graph_file(gname)
            if not fpath:
                continue
            print(f"\n  {gname}")
            graph_data = self._load_graph_data(fpath)
            G_ref = self._build_graph_ref(graph_data)
            layout_gen = LayoutGenerator(graph_data)

            infos = []
            for exp_name in gnn_names:
                try:
                    pos = self._generate_deepcs(layout_gen, exp_name)
                    infos.append({"name": gnn_display[exp_name], "G": G_ref, "pos": pos})
                except Exception as e:
                    print(f"    [WARN] {exp_name}: {e}")

            out_path = os.path.join(self.output_dir, f"fig5-3_{gname}.png")
            self._render_grid(infos, out_path, gname)
            gc.collect()

    def run(self):
        self.load_models()
        self.generate_fig51()
        self.generate_fig52()
        self.generate_fig53()
        print(f"\nDone. Figures saved to {self.output_dir}")


if __name__ == "__main__":
    gen = PaperFigureGenerator(INPUT_DIR, OUTPUT_DIR)
    gen.run()
