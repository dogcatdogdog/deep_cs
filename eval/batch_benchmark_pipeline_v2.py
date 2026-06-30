# -*- coding: utf-8 -*-
import os
import glob
import json
import math
import warnings
import gc
import pandas as pd
import torch
from tqdm import tqdm
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import eval_utils as utils
from layout import LayoutGenerator

warnings.filterwarnings("ignore")

class BatchBenchmarkPipeline:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Sub-directories for organized outputs
        self.details_dir = os.path.join(self.output_dir, "graph_details")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.csv_dir = os.path.join(self.output_dir, "csv_raw")
        self.summary_csv_path = os.path.join(self.output_dir, "academic_benchmark_summary.csv")
        
        self.baseline_algos = ["fr", "kk", "mds", "fa2", "d3"]
        self.deepcs_experiments = [
#            "v18_metrics_phase2_joint_balanced",
#            "hpo_trial_017",
#            "hpo_trial_019",
#            "hpo_trial_051",
#            "hpo_6d_nocoh_trial_020",
#            "hpo_6d_nocoh_trial_022",
#            "hpo_6d_nocoh_trial_028",
#            "hpo_6d_nocoh_trial_019"
#            "hpo_pure_global_trial_007",
#            "hpo_v18_canonical_trial_002",
#            "hpo_v23_final_trial_000",
#            "hpo_rank_loss_trial_008",
            "ablation_exp0_baseline",
            "ablation_exp1_nodetach",
            "ablation_exp2_zeropadding",
            "ablation_exp3_widening",
            "ablation_exp4_gcn",
            "ablation_exp5_gin",
#            "loss_abl_kl_ar",
#            "loss_abl_kl_lnp",
#            "loss_abl_kl_eu",
            "ar_only_gamma1.5",
            "eu_only",
            "kl_only",
            "lnp_only",
#            "loss_abl_eu",
#            "loss_abl_ar",
#            "loss_abl_kl",
#            "loss_abl_lnp",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_000",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_001",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_002",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_003",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_004",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_007",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_008",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_009",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_010",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_011",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_012",
#            "hpo_v10_sinar/hpo_v10_sinar_trial_013",
        ]
        self.algorithms = self.baseline_algos.copy()
        
        self.all_records = []
        self.loaded_deepcs_components = {}
        
        # Define the metrics we want to extract and export
        self.metrics_keys = [
            "Time",
            "Stress", "VertexRes(VR)", "CrossMetric(EC)", 
            "NeighPreserv(NP)", "AngularRes(AR)", "EdgeLengthDistrib(EU)", 
            "LocalNeighPerserv(LNP)", "LocalAngularRes(LAR)", "LocalEdgeLengthDistrib(LEU)"
        ]
        
        self.png_display_config = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.details_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def load_all_deepcs_models(self):
        print(f"[INFO] Preparing to load {len(self.deepcs_experiments)} DeepCS model variants...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        for exp_name in self.deepcs_experiments:
            
            cfg_path = os.path.join(project_root, "logs", exp_name, "config.yaml")
            ckpt_path = os.path.join(project_root, "logs", exp_name, "best_model.pth")
            
            if os.path.exists(cfg_path) and os.path.exists(ckpt_path):
                try:
                    LayoutGenerator.load_deepcs_model(cfg_path, ckpt_path, model_name=exp_name)
                    self.loaded_deepcs_components[exp_name] = LayoutGenerator.deepcs_components
                    self.algorithms.append(exp_name)
                except Exception as e:
                    print(f"[ERROR] Failed to load model {exp_name}: {e}")
            else:
                print(f"[WARNING] Config or weights not found for {exp_name}, skipping this model.")


    def run(self):
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        graphml_files = glob.glob(os.path.join(self.input_dir, "*.graphml"))
        all_files = json_files + graphml_files
        
        if not all_files:
            print(f"[Warning] No .json or .graphml files found in '{self.input_dir}'.")
            return

        print(f"Starting batch evaluation for {len(all_files)} graph datasets...")

        self.load_all_deepcs_models()
        print(f"[INFO] Algorithms actually participating in the evaluation pipeline: {self.algorithms}")
        
        for filepath in tqdm(all_files, desc="Processing Graphs", unit="graph"):
            graph_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1].lower()
            clean_data = None
            
            graph_records = []       # For CSV
            graph_json_layouts = []  # For JSON
            png_layouts_info = []    # For PNG
            
            try:
                if ext == ".json":
                    # [Modified] Add encoding fallback mechanism (UTF-8 -> GBK -> Latin-1)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            raw_data = json.load(f)
                    except UnicodeDecodeError:
                        tqdm.write(f"[WARNING] Non-UTF-8 encoded file found: {graph_name}.json, attempting to read with GBK fallback...")
                        try:
                            with open(filepath, 'r', encoding='gbk') as f:
                                raw_data = json.load(f)
                        except UnicodeDecodeError:
                            with open(filepath, 'r', encoding='latin-1') as f:
                                raw_data = json.load(f)
                                
                    clean_data = utils.convert_case_format(raw_data, graph_name=graph_name)
                    
                elif ext == ".graphml":
                    clean_data = utils.convert_graphml_format(filepath, graph_name=graph_name)
                else:
                    continue
                    
                layout_gen = LayoutGenerator(clean_data)
                
                algo_map = {
                    "fr": layout_gen.generate_fr,
                    "kk": layout_gen.generate_kk,
                    "mds": layout_gen.generate_mds,
                    "fa2": layout_gen.generate_fa2,
                    "neato": getattr(layout_gen, "generate_neato", None),
                    "d3": layout_gen.generate_d3,
                }
                
                def make_deepcs_generator(name):
                    def generator():
                        LayoutGenerator.deepcs_components = self.loaded_deepcs_components[name]
                        layout_result = layout_gen.generate_deepcs()
                        layout_result["algorithm"] = name 
                        return layout_result
                    return generator

                for exp_name in self.loaded_deepcs_components.keys():
                    algo_map[exp_name] = make_deepcs_generator(exp_name)

                
                for algo in self.algorithms:
                    algo_func = algo_map.get(algo)
                    if not algo_func: 
                        continue
                        
                    try:
                        start_time = time.time()
                        layout_data = algo_func()
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        algo_name_full = layout_data.get("algorithm", algo.upper())
                        G_layout, pos = utils.parse_json_to_graph(layout_data)
                        
                        raw_metrics = {
                            "Time": execution_time,
                            "Stress": utils.calculate_stress(G_layout, pos),
                            "VertexRes(VR)": utils.calculate_vertex_resolution(G_layout, pos),
                            "CrossMetric(EC)": utils.calculate_crossing_metric(G_layout, pos),
                            "NeighPreserv(NP)": utils.calculate_neighborhood_preservation(G_layout, pos),
                            "AngularRes(AR)": utils.calculate_angular_resolution(G_layout, pos),
                            "EdgeLengthDistrib(EU)": utils.calculate_edge_length_distribution(G_layout, pos),
                            "LocalNeighPerserv(LNP)":utils.calculate_LNP(G_layout, pos),
                            "LocalAngularRes(LAR)":utils.calculate_LAR(G_layout, pos),
                            "LocalEdgeLengthDistrib(LEU)":utils.calculate_LEU(G_layout, pos)
                        }
                        
                        metrics = {k: float(v) if v is not None else None for k, v in raw_metrics.items()}
                        
                        record = {
                            "Graph_ID": graph_name, 
                            "Algorithm": algo_name_full
                        }
                        record.update(metrics)
                        graph_records.append(record)
                        self.all_records.append(record)
                        
                        graph_json_layouts.append({
                            "algorithm": algo_name_full,
                            "metrics": metrics,
                            "layout_data": layout_data
                        })
                        
                        png_layouts_info.append({
                            "name": algo_name_full,
                            "G": G_layout,
                            "pos": pos,
                            "metrics": metrics
                        })
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            tqdm.write(f"{graph_name} in {algo.upper()}(OOM)")
                            if torch.cuda.is_available(): torch.cuda.empty_cache() 
                            continue
                        else:
                            tqdm.write(f"[Error] Graph {graph_name} crashed on {algo.upper()}: {e}")
                            continue    
                    except Exception as e:
                        tqdm.write(f"[Error] Graph {graph_name} crashed on {algo.upper()}: {e}")
                        continue
                if graph_records:
                    # Output A: Single Graph JSON (存入 graph_details 文件夹)
                    json_path = os.path.join(self.details_dir, f"{graph_name}_layouts.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump({"graph_id": graph_name, "evaluations": graph_json_layouts}, f, indent=2)
                    
                    # Output B: Single Graph Raw CSV (存入 csv_raw 文件夹)
                    csv_path = os.path.join(self.csv_dir, f"{graph_name}_metrics.csv")
                    utils.export_evaluation_csv(
                        records=graph_records, 
                        output_path=csv_path, 
                        metrics_cols=self.metrics_keys, 
                        aggregate=False
                    )
                    
                    # Output C: Single Graph PNG (存入 images 文件夹)
                    img_path = os.path.join(self.images_dir, f"{graph_name}_comparison.png")
                    utils.generate_comparison_png(
                        layouts_info=png_layouts_info, 
                        output_path=img_path, 
                        metrics_config=self.png_display_config
                    )
                # =========================================================        
            except Exception as e:
                import traceback
                tqdm.write(f"[Error] Failed to process file {filepath}: {e}")
                traceback.print_exc()
            
            finally:
                if 'layout_gen' in locals():
                    del layout_gen
                del clean_data
                del graph_records
                del graph_json_layouts
                del png_layouts_info
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._aggregate_and_export()

    def _aggregate_and_export(self):
        if not self.all_records:
            print("[Error] No evaluation records were successfully generated.")
            return
            
        utils.export_evaluation_csv(
            records=self.all_records,
            output_path=self.summary_csv_path,
            metrics_cols=self.metrics_keys,
            group_by_col="Algorithm",
            aggregate=True
        )
        
        print(f"\n[Success] Batch benchmark completed!")
        print(f" -> Master summary report saved to: {self.summary_csv_path}")
        print(f" -> Detailed JSONs, CSVs, and PNGs saved in: {self.output_dir}/\n")

if __name__ == "__main__":
#    INPUT_DIRECTORY = "./rome_1000"
    INPUT_DIRECTORY = "./benchmark_test_0312"

#    OUTPUT_DIRECTORY = "./bench_pipeline_out_0612_rome1000_11"
    OUTPUT_DIRECTORY = "/root/autodl-tmp/bench_pipeline_out_0612_ablation_2"
    
    pipeline = BatchBenchmarkPipeline(
        input_dir=INPUT_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY
    )
    pipeline.run()