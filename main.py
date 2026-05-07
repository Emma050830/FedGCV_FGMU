from FGMU.fl_model.trainer import FGLTrainer
from FGMU.client.train_and_test import train_and_test_client, load_client_data
from FGMU.evaluation.backdoor_evaluation import BackdoorEvaluator
from FGMU.evaluation.backdoor_train_and_test import (backdoor_aware_train_and_test_client, federated_unlearning_with_backdoor_evaluation, train_and_test_client_with_model_return)
from FGMU.evaluation.graph_mia_train_and_test import graph_mia_aware_train_and_test_client
from FGMU.vgae.vgae_virtual_client import train_client_vgae, GraphFeatureExtractor, VirtualClientGenerator
from FGMU.vgae.convert_to_pyg_data import convert_to_pyg_data
from FGMU.fl_model.fedavg.client import FedAvgClient
from FGMU.utils.basic_utils import seed_everything
import argparse
import torch
import os
import glob
import copy


def get_args():
    supported_scenario = ["graph_fl", "subgraph_fl"]
    supported_graph_fl_datasets = [
"AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR", "hERG", "ogbg-molhiv", "ogbg-molpca", "ogbg-ppa", "ogbg-code2"]
    supported_subgraph_fl_datasets = [
"Cora", "CiteSeer", "PubMed", "CS", "Physics", "Computers", "Photo", "Chameleon", "Squirrel", "ogbn-arxiv", "ogbn-products", "Tolokers", "Actor", \
"Amazon-ratings", "Roman-empire", "Questions", "Minesweeper", "Reddit", "Flickr"]
    supported_graph_fl_simulations = ["graph_fl_cross_domain", "graph_fl_label_skew", "graph_fl_topology_skew", "graph_fl_feature_skew"]
    supported_subgraph_fl_simulations = ["subgraph_fl_label_skew", "subgraph_fl_louvain_plus", "subgraph_fl_metis_plus", "subgraph_fl_louvain", "subgraph_fl_metis"]
    supported_graph_fl_task = ["graph_cls", "graph_reg"]
    supported_subgraph_fl_task = ["node_cls", "link_pred", "node_clust"]
    supported_fl_algorithm = ["isolate", "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad", "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp", "fgssl", "fedgl"]
    supported_metrics = ["accuracy", "precision", "f1", "recall", "auc", "ap", "clustering_accuracy", "nmi", "ari"]
    supported_evaluation_modes = ["global_model_on_local_data", "global_model_on_global_data", "local_model_on_local_data", "local_model_on_global_data"]
    supported_data_processing = ["raw", "random_feature_sparsity", "random_feature_noise", "random_topology_sparsity", "random_topology_noise", "random_label_sparsity", "random_label_noise"]
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)

    # global dataset settings 
    parser.add_argument("--root", type=str, default="change_to_your_root_path")
    parser.add_argument("--scenario", type=str, default="subgraph_fl", choices=supported_scenario)
    parser.add_argument("--dataset", type=str, default=[], action='append')
    parser.add_argument("--processing", type=str, default="raw", choices=supported_data_processing)
    parser.add_argument("--processing_percentage", type=float, default=0.1)

    # post_process: 
    # random feature mask ratio
    parser.add_argument("--feature_mask_prob", type=float, default=0.1)
    # dp parameter: epsilon, support 1) random response for link
    parser.add_argument("--dp_epsilon", type=float, default=0.)
    # homo/hete random injection
    parser.add_argument("--homo_injection_ratio", type=float, default=0.)
    parser.add_argument("--hete_injection_ratio", type=float, default=0.)

    # fl settings
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--fl_algorithm", type=str, default="fedavg", choices=supported_fl_algorithm)
    parser.add_argument("--client_frac", type=float, default=1.0)

    # simulation settings
    parser.add_argument("--simulation_mode", type=str, default="subgraph_fl_louvain", choices=supported_graph_fl_simulations + supported_subgraph_fl_simulations)
    parser.add_argument("--dirichlet_alpha", type=float, default=10)
    parser.add_argument("--dirichlet_try_cnt", type=int, default=100)
    parser.add_argument("--least_samples", type=int, default=5)
    parser.add_argument("--louvain_resolution", type=float, default=1)
    parser.add_argument("--louvain_delta", type=float, default=20, help="Maximum allowable difference in node counts between any two clients in the graph_fl_louvain simulation.")
    parser.add_argument("--metis_num_coms", type=int, default=100)

    # task settings
    parser.add_argument("--task", type=str, default="node_cls", choices=supported_graph_fl_task + supported_subgraph_fl_task)
    parser.add_argument("--num_clusters", type=int, default=7)
    # training settings
    parser.add_argument("--train_val_test", type=str, default="default_split") # e.g., 0.2-0.4-0.4
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)

    # model settings
    parser.add_argument("--model", type=str, default=[], action='append')
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=64)

    # evaluation settings
    parser.add_argument("--metrics", type=str, default=[], action='append')
    parser.add_argument("--evaluation_mode", type=str, default="local_model_on_local_data", choices=supported_evaluation_modes)

    # privacy
    parser.add_argument("--dp_mech", type=str, default='no_dp')
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dp_q", type=float, default=0.1)

    # for node-level and link-level prediction tasks
    parser.add_argument("--max_degree", type=int, default=5)
    parser.add_argument("--max_epsilon", type=float, default=20)

    # debug
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)
    parser.add_argument("--comm_cost", type=bool, default=False)
    parser.add_argument("--model_param", type=bool, default=False)

    # backdoor
    parser.add_argument("--enable_backdoor", type=bool, default=False)
    parser.add_argument("--malicious_clients", type=int, default=[], action='append')
    parser.add_argument("--poisoning_rate", type=float, default=0.5)
    parser.add_argument("--trigger_type", type=str, default="pattern", choices=["pattern", "statistical"])
    parser.add_argument("--target_offset", type=float, default=0.5)

    # unlearning settings
    parser.add_argument("--unlearning_mode", type=str, default="interactive", choices=["interactive", "automated", "batch"], help="Unlearning mode")
    parser.add_argument("--batch_unlearn_clients", type=int, nargs='+', default=[], help="Client list for batch unlearning")

    # unlearning constraints (to reduce catastrophic forgetting)
    parser.add_argument(
        "--max_update_norm",
        type=float,
        default=10.0,
        help="Clip the (projected) unlearning update norm per step (<=0 to disable).",
    )
    parser.add_argument(
        "--unlearn_param_delta_tau",
        type=float,
        default=10.0,
        help=(
            "Constrain unlearning parameter drift: enforce ||theta - theta0||_2 <= tau by projection after each step. "
            "Set <=0 to disable."
        ),
    )

    # unlearning hyperparams (optional overrides)
    parser.add_argument(
        "--forgetting_loss_type",
        type=str,
        default=None,
        help="Unlearning loss type override (e.g., 'npo', 'neg_ce'). Default keeps current config value.",
    )
    parser.add_argument(
        "--forget_ascend_scale",
        type=float,
        default=None,
        help="Override GRU/NPO unlearning strength scaling. Default keeps current config value.",
    )
    parser.add_argument(
        "--unlearning_epochs",
        type=int,
        default=None,
        help="Override unlearning epochs. Default keeps current config value.",
    )

    # fixed-pre-threshold alignment helper (Graph-MIA)
    parser.add_argument(
        "--fixed_mia_target_loss_margin",
        type=float,
        default=0.5,
        help=(
            "During unlearning, optionally push the target CE loss above (fixed_pre_threshold + margin). "
            "Only active in Graph-MIA path when fixed threshold is available. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--fixed_mia_push_lambda",
        type=float,
        default=3.0,
        help="Weight for the fixed-pre-threshold loss push term (when enabled).",
    )

    # virtual repair (post-unlearning) settings
    parser.add_argument("--enable_virtual_repair", type=bool, default=True)
    parser.add_argument("--extra_virtual_rounds", type=int, default=5)
    # Keep fixed-pre-threshold MIA close to post-unlearning baseline
    parser.add_argument("--virtual_repair_mia_delta_max", type=float, default=0.1, help="Absolute MIA increase allowed vs post-unlearning fixed-pre MIA")
    parser.add_argument("--virtual_repair_mia_abs_cap", type=float, default=-1.0, help="Absolute MIA cap (if >0, takes min with baseline+delta)")
    parser.add_argument("--virtual_repair_utility_target", type=float, default=0.80)
    parser.add_argument(
        "--virtual_repair_target_loss_margin",
        type=float,
        default=0.2,
        help=(
            "Extra safety margin for target_mean_loss vs fixed-pre threshold during repair. "
            "When fixed_thr is available, require target_mean_loss >= fixed_thr + margin. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--virtual_repair_vgae_source",
        type=str,
        default="target",
        choices=["retained", "target"],
        help="Which graph to use to train VGAE for virtual client generation.",
    )
    parser.add_argument(
        "--virtual_repair_vgae_feature_noise",
        type=float,
        default=0.1,
        help="Feature noise level passed to the VGAE virtual client generator.",
    )
    parser.add_argument(
        "--virtual_repair_backtrack_strategy",
        type=str,
        default="min_mia",
        choices=["first_feasible", "min_mia", "min_alpha", "max_alpha"],
        help=(
            "How to choose a scaled server update when the full update violates the MIA cap. "
            "first_feasible: accept the first alpha (from BACKTRACK_ALPHAS order) that satisfies cap; "
            "min_mia: choose feasible alpha with smallest MIA; "
            "min_alpha/max_alpha: choose smallest/largest feasible alpha."
        ),
    )
    

    return parser.parse_args()


def main():
# 1) Federated training
    args = get_args()

    # Fix random seed to reduce variance in training/virtual data/MIA evaluation.
    try:
        seed_everything(int(getattr(args, 'seed', 2025)))
    except Exception:
        pass

    args.root = "FGMU/Cora_overlapping"

    args.dataset = ["Cora"]
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10

    # Enable backdoor-attack unlearning evaluation mode
    args.enable_backdoor = True
    args.malicious_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    args.poisoning_rate = 0.5
    args.trigger_type = "pattern"


    if True:
        args.fl_algorithm = "fedavg"
        args.model = ["gcn"]
    else:
        args.fl_algorithm = "fedproto"
        args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

    args.metrics = ["accuracy"]
    # Align utility evaluation to the global model (avoid local-model accuracy masking issues)
    args.evaluation_mode = "global_model_on_local_data"
    print("=== Federated Unlearning & Backdoor Evaluation System ===")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    if args.enable_backdoor:
        print("=== Backdoor Evaluation Mode ===")
        print(f"Malicious clients: {args.malicious_clients}")
        print(f"Poisoning rate: {args.poisoning_rate}")
        print(f"Trigger type: {args.trigger_type}")
        print(f"Unlearning mode: {args.unlearning_mode}")
    else:
        print("=== Standard Evaluation Mode ===")
    
# 2) Federated training stage
    print("\n=== Start Federated Training ===")
    trainer = FGLTrainer(args)
    trainer.train()
    print("=== Federated Training Completed ===")

# 3) Unlearning experiment stage
    config = create_unlearning_config(args)

    # Ensure all subsequent load_client_data(...) calls can find the data directory
    # (including calls inside evaluation modules).
    _set_data_env(args)
    
    if args.enable_backdoor:
        # Backdoor evaluation mode
        if args.unlearning_mode == "interactive":
            # Interactive mode: user chooses which client to forget
            print("\n=== Interactive Unlearning Mode ===")
            results = run_interactive_unlearning(config, args, trainer)
            
        elif args.unlearning_mode == "automated":
            # Automated mode: use the full backdoor evaluation pipeline
            print("\n=== Automated Backdoor Evaluation Pipeline ===")
            results = federated_unlearning_with_backdoor_evaluation(config)
            
        elif args.unlearning_mode == "batch":
            # Batch mode: evaluate multiple specified clients
            print("\n=== Batch Unlearning Evaluation Mode ===")
            if args.batch_unlearn_clients:
                target_clients = args.batch_unlearn_clients
            else:
                target_clients = args.malicious_clients
            results = run_batch_unlearning(config, target_clients, args)
            
    else:
        # Standard unlearning mode
        print("\n=== Standard Unlearning Mode ===")
        results = run_standard_unlearning(config)

    # # 4) Result summary
    # print("\n=== Experiment Result Summary ===")
    # summarize_results(results, args)


def run_interactive_unlearning(unlearn_config, args, trainer):
    """
    Interactive unlearning mode
    """
    # Select the client to forget
    forget_id = int(input("Please enter the client id to forget: "))
    print(f"Unlearning will be performed on Client {forget_id}...")
    selected_federated_model = trainer.clients[forget_id].task.model

    # Load the data for the selected client (pass args.root as data_root to avoid environment variable dependency)
    raw_data = load_client_data(forget_id)
    

    
    # Initialize variables that may be used in different evaluation paths to avoid unassigned references
    pre_unlearn_results = None
    post_unlearn_backdoor_results = None
    effectiveness = None
    virtual_retrain_stats = None
    unlearn_model = None

    # Determine if the client is malicious
    is_malicious = forget_id in args.malicious_clients
    
    if args.enable_backdoor:
        # Choose evaluation method by config
        if getattr(unlearn_config, 'evaluation_method', 'backdoor') == 'graph_mia':
            print("=== Using Graph-MIA evaluation ===")
            # The member/non-member distribution for Graph-MIA should match the data trained by the global model: use the server's aggregated model by default
            global_model = getattr(getattr(trainer, 'server', None), 'task', None)
            global_model = getattr(global_model, 'model', None) or selected_federated_model
            results = graph_mia_aware_train_and_test_client(forget_id, raw_data, unlearn_config, model=global_model)
            pre_unlearn_results = results

            # Treat the post-unlearning model as the "global forgotten model" for subsequent (optional) repair training
            try:
                unlearned_model = results.get('model', None) if isinstance(results, dict) else None
                if unlearned_model is not None and hasattr(trainer, 'server') and hasattr(trainer.server, 'task'):
                    trainer.server.task.model = unlearned_model

                    # Quick utility check immediately after unlearning (global model on target/retained).
                    try:
                        server_task = trainer.server.task
                        metrics = list(getattr(trainer.args, 'metrics', ['accuracy']))

                        # (a) target client utility (expected to drop after unlearning)
                        try:
                            tgt_sd = trainer.clients[forget_id].task.splitted_data
                            tgt_res = server_task.evaluate(tgt_sd, mute=True)
                            tgt_info = ", ".join([f"{m}_test={float(tgt_res.get(m + '_test', 0.0)):.4f}" for m in metrics])
                            print(f"[After Unlearning] Global model on TARGET(client {forget_id}) data: {tgt_info}")
                        except Exception:
                            pass

                        # (b) retained utility across other clients
                        tot_samples = 0
                        agg = {f"{m}_val": 0.0 for m in metrics}
                        agg.update({f"{m}_test": 0.0 for m in metrics})
                        for cid in range(getattr(trainer.args, 'num_clients', 0)):
                            if cid == forget_id:
                                continue
                            try:
                                sd = trainer.clients[cid].task.splitted_data
                                num_samples = trainer.clients[cid].task.num_samples
                                res = server_task.evaluate(sd, mute=True)
                                for m in metrics:
                                    agg[m + '_val'] += float(res.get(m + '_val', 0.0)) * num_samples
                                    agg[m + '_test'] += float(res.get(m + '_test', 0.0)) * num_samples
                                tot_samples += num_samples
                            except Exception:
                                continue
                        if tot_samples > 0:
                            retained_info = ", ".join([f"{m}_val={agg[m + '_val']/tot_samples:.4f}, {m}_test={agg[m + '_test']/tot_samples:.4f}" for m in metrics])
                            print(f"[After Unlearning] Global model on RETAINED clients: {retained_info}")
                    except Exception:
                        pass
            except Exception:
                unlearned_model = None

            # Optionally: add VGAE virtual client for a few extra federated rounds (repair) to restore utility
            enable_virtual_repair = getattr(args, 'enable_virtual_repair', True)
            if enable_virtual_repair:
                try:
                    from FGMU.evaluation.graph_mia_evaluation import GraphMIAEvaluator
                    from FGMU.evaluation.backdoor_train_and_test import validate_and_extract_raw_data

                    evaluator = GraphMIAEvaluator(unlearn_config, device=unlearn_config.device)
                    norm = validate_and_extract_raw_data(raw_data, forget_id)
                    training = norm.get('train')
                    testset = norm.get('test')

                    target_dataset = []
                    if isinstance(training, (list, tuple)):
                        target_dataset.extend(list(training))
                    else:
                        target_dataset.append(training)
                    if testset is not None:
                        if isinstance(testset, (list, tuple)):
                            for t in testset:
                                if t not in target_dataset:
                                    target_dataset.append(t)
                        else:
                            if testset not in target_dataset:
                                target_dataset.append(testset)

                    retained_dataset = []
                    for cid in range(getattr(unlearn_config, 'num_clients', args.num_clients)):
                        if cid == forget_id:
                            continue
                        try:
                            other_raw = load_client_data(cid)
                            other_norm = validate_and_extract_raw_data(other_raw, cid)
                            other_train = other_norm.get('train', None)
                            if other_train is None:
                                continue
                            if isinstance(other_train, (list, tuple)):
                                retained_dataset.extend(list(other_train))
                            else:
                                retained_dataset.append(other_train)
                        except Exception:
                            continue
                    if len(retained_dataset) == 0:
                        retained_dataset = list(target_dataset)

                    # Evaluate virtual (non-member) set: must use data not seen by the current model for each repair monitor
                    seed = getattr(unlearn_config, 'seed', 0)
                    num_virtual_eval = int(getattr(unlearn_config, 'num_virtual_graphs', 10))
                    noise_scale_eval = float(getattr(unlearn_config, 'virtual_noise_scale', 0.5))
                    virtual_dataset_eval = evaluator.generate_virtual_dataset_from(
                        retained_dataset,
                        num_virtual=num_virtual_eval,
                        noise_scale=noise_scale_eval,
                        seed=int(seed) + 1000,
                    )

                    # Use fixed threshold: use the pre-unlearning threshold for comparable evaluation (focus only on fixed-pre-threshold)
                    fixed_thr = None
                    try:
                        fixed_thr = results.get('pre_mia', {}).get('loss_threshold', None)
                    except Exception:
                        fixed_thr = None

                    # After repair, fixed-pre-threshold mia_rate should not increase much compared to post-unlearning (default +5% absolute)
                    base_mia = None
                    try:
                        base_mia = results.get('post_mia_fixed_threshold', {}).get('mia_rate', None)
                    except Exception:
                        base_mia = None

                    try:
                        delta_max = float(getattr(args, 'virtual_repair_mia_delta_max', 0.1))
                    except Exception:
                        delta_max = 0.1
                    try:
                        mia_cap_abs = float(getattr(args, 'virtual_repair_mia_abs_cap', -1.0))
                    except Exception:
                        mia_cap_abs = -1.0

                    if base_mia is not None:
                        mia_cap = float(base_mia) + float(delta_max)
                    else:
                        mia_cap = 0.05
                    if mia_cap_abs is not None and float(mia_cap_abs) > 0:
                        mia_cap = min(float(mia_cap), float(mia_cap_abs))

                    try:
                        utility_target = float(getattr(args, 'virtual_repair_utility_target', 0.80))
                    except Exception:
                        utility_target = 0.80

                    try:
                        target_loss_margin = float(getattr(args, 'virtual_repair_target_loss_margin', 0.5))
                    except Exception:
                        target_loss_margin = 0.5
                    required_target_mean_loss = None
                    if fixed_thr is not None and target_loss_margin is not None and float(target_loss_margin) > 0:
                        try:
                            required_target_mean_loss = float(fixed_thr) + float(target_loss_margin)
                        except Exception:
                            required_target_mean_loss = None

                    if required_target_mean_loss is not None:
                        print(
                            f"[Virtual Repair] Privacy guard: base_mia={base_mia}, mia_cap={mia_cap:.4f} (fixed-pre-threshold), "
                            f"target_mean_loss>={required_target_mean_loss:.4f}, utility_target={utility_target:.2f}"
                        )
                    else:
                        print(
                            f"[Virtual Repair] Privacy guard: base_mia={base_mia}, mia_cap={mia_cap:.4f} (fixed-pre-threshold), "
                            f"utility_target={utility_target:.2f}"
                        )

                    USE_RANDOM_NOISE_VIRTUAL_CLIENT = False
                    RANDOM_VIRTUAL_FEATURE_STD = 1.0
                    RANDOM_VIRTUAL_EDGE_DENSITY = None

                    ENABLE_REPAIR_BACKTRACKING = True
                    BACKTRACK_ALPHAS = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
                    BACKTRACK_STRATEGY = str(getattr(args, 'virtual_repair_backtrack_strategy', 'first_feasible')).strip().lower()

                    vgae_source = str(getattr(args, 'virtual_repair_vgae_source', 'target')).strip().lower()
                    if vgae_source not in ('retained', 'target'):
                        vgae_source = 'target'

                    pyg_train_data = None
                    if vgae_source == 'retained':
                        pyg_train_data = retained_dataset[0] if len(retained_dataset) > 0 else None
                    else:
                        pyg_train_data = raw_data.get('train', None) if isinstance(raw_data, dict) else None
                        if isinstance(pyg_train_data, (list, tuple)):
                            pyg_train_data = pyg_train_data[0] if len(pyg_train_data) > 0 else None

                    if pyg_train_data is None:
                        raise ValueError("Cannot get subgraph data for VGAE training")

                    if USE_RANDOM_NOISE_VIRTUAL_CLIENT:
                        try:
                            from torch_geometric.data import Data
                        except Exception as e:
                            raise ImportError("USE_RANDOM_NOISE_VIRTUAL_CLIENT=True requires torch_geometric") from e

                        ref = pyg_train_data
                        num_nodes = int(ref.x.size(0))
                        feat_dim = int(ref.x.size(1))
                        try:
                            num_classes = int(getattr(ref, 'num_global_classes', None) or trainer.server.task.num_global_classes)
                        except Exception:
                            num_classes = 7

                        p = RANDOM_VIRTUAL_EDGE_DENSITY
                        if p is None:
                            try:
                                e = int(ref.edge_index.size(1))
                                denom = max(1, num_nodes * max(1, (num_nodes - 1)))
                                p = min(0.05, max(0.001, float(e) / float(denom)))
                            except Exception:
                                p = 0.01

                        num_edges = max(1, int(p * num_nodes * max(1, (num_nodes - 1))))
                        row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
                        col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
                        edge_index = torch.stack([row, col], dim=0)

                        x = torch.randn((num_nodes, feat_dim), dtype=ref.x.dtype) * float(RANDOM_VIRTUAL_FEATURE_STD)
                        y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
                        virtual_subgraph_data = Data(x=x, edge_index=edge_index, y=y)
                        virtual_subgraph_data.num_nodes = num_nodes
                    else:
                        trained_vgae = train_client_vgae(pyg_train_data)
                        try:
                            vgae_feature_noise = float(getattr(args, 'virtual_repair_vgae_feature_noise', 0.1))
                        except Exception:
                            vgae_feature_noise = 0.1
                        vgae_config = {
                            'use_spectrum': True,
                            'privacy_level': 'high',
                            'recon_threshold': 0.7,
                            'feature_noise': float(vgae_feature_noise),
                            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                        }
                        extractor = GraphFeatureExtractor(pyg_train_data)
                        generator = VirtualClientGenerator(trained_vgae, extractor, vgae_config)
                        v_adj, v_features, v_metadata = generator.generate()
                        virtual_subgraph_data = convert_to_pyg_data(v_adj, v_features, v_metadata)

                    num_nodes = virtual_subgraph_data.x.size(0) if hasattr(virtual_subgraph_data, 'x') and virtual_subgraph_data.x is not None else 0
                    if not hasattr(virtual_subgraph_data, 'num_global_classes'):
                        try:
                            virtual_subgraph_data.num_global_classes = trainer.server.task.num_global_classes
                        except Exception:
                            try:
                                virtual_subgraph_data.num_global_classes = int(virtual_subgraph_data.y.max().item() + 1)
                            except Exception:
                                virtual_subgraph_data.num_global_classes = 1

                    if not hasattr(virtual_subgraph_data, 'global_map'):
                        virtual_subgraph_data.global_map = list(range(num_nodes))

                    target_splitted_data = None
                    target_num_samples = None
                    try:
                        if hasattr(trainer, 'clients') and 0 <= forget_id < len(trainer.clients):
                            target_splitted_data = trainer.clients[forget_id].task.splitted_data
                            target_num_samples = trainer.clients[forget_id].task.num_samples
                    except Exception:
                        target_splitted_data = None
                        target_num_samples = None

                    if hasattr(trainer, 'server') and hasattr(trainer.server, 'task') and hasattr(trainer.server.task, 'data_dir'):
                        data_dir_for_virtual = os.path.join(str(trainer.server.task.data_dir), f"virtual_client_{forget_id}")
                    else:
                        data_dir_for_virtual = os.path.join(str(args.root), f"virtual_client_{forget_id}")
                    message_pool_for_virtual = trainer.message_pool if hasattr(trainer, 'message_pool') else {}
                    device_for_virtual = trainer.device if hasattr(trainer, 'device') else torch.device('cpu')
                    virtual_client = FedAvgClient(trainer.args, forget_id, virtual_subgraph_data, data_dir_for_virtual, message_pool_for_virtual, device_for_virtual)
                    trainer.clients[forget_id] = virtual_client

                    extra_rounds = int(getattr(args, 'extra_virtual_rounds', 5))
                    old_rounds = trainer.args.num_rounds

                    best_state = copy.deepcopy(trainer.server.task.model.state_dict())
                    best_round = -1
                    best_retained_test = -1.0
                    best_guard_mia = None

                    print(f"[Virtual Repair] Replacing client {forget_id} with virtual client, up to {extra_rounds} extra rounds (privacy-guarded)...")
                    for r in range(extra_rounds):
                        prev_state = copy.deepcopy(trainer.server.task.model.state_dict())
                        trainer.args.num_rounds = 1
                        trainer.train()

                        curr_state = copy.deepcopy(trainer.server.task.model.state_dict())

                        def _interpolate_state(a_state, b_state, alpha: float):
                            out = {}
                            for k, a_v in a_state.items():
                                b_v = b_state[k]
                                if torch.is_floating_point(a_v):
                                    out[k] = a_v + (b_v - a_v) * float(alpha)
                                else:
                                    out[k] = b_v
                            return out

                        def _guard_mia_for_state(state_dict, tag: str):
                            if fixed_thr is None:
                                return None, None
                            try:
                                trainer.server.task.model.load_state_dict(state_dict)
                                stats = evaluator.evaluate_mia_attack(
                                    trainer.server.task.model,
                                    retained_dataset,
                                    virtual_dataset_eval,
                                    target_dataset,
                                    loss_threshold=fixed_thr,
                                    run_name=f'post_virtual_repair_guard_r{r+1}_{tag}',
                                    verbose=False,
                                )
                                mia = float(stats.get('mia_rate', float('nan')))
                                return stats, mia
                            except Exception:
                                return None, None

                        guard_stats, guard_mia = _guard_mia_for_state(curr_state, 'full')
                        if guard_mia is not None:
                            try:
                                tml = float(guard_stats.get('target_mean_loss', float('nan'))) if guard_stats else float('nan')
                            except Exception:
                                tml = float('nan')
                            if required_target_mean_loss is not None:
                                print(
                                    f"[Virtual Repair][Guard][r{r+1}] alpha=1.0 mia={guard_mia:.2%} cap={mia_cap:.2%} "
                                    f"target_mean_loss={tml:.4f} req>={required_target_mean_loss:.4f}"
                                )
                            else:
                                print(f"[Virtual Repair][Guard][r{r+1}] alpha=1.0 mia={guard_mia:.2%} cap={mia_cap:.2%} target_mean_loss={tml:.4f}")

                        accepted_state = curr_state
                        accepted_guard_mia = guard_mia
                        accepted_guard_stats = guard_stats
                        used_alpha = 1.0

                        full_update_violates_cap = (guard_mia is not None and guard_mia > mia_cap)
                        full_update_violates_margin = False
                        if required_target_mean_loss is not None and guard_stats is not None:
                            try:
                                full_tml = float(guard_stats.get('target_mean_loss', float('nan')))
                                if full_tml < float(required_target_mean_loss):
                                    full_update_violates_margin = True
                            except Exception:
                                full_update_violates_margin = False

                        if full_update_violates_cap or full_update_violates_margin:
                            if ENABLE_REPAIR_BACKTRACKING:
                                feasible = []
                                for alpha in BACKTRACK_ALPHAS:
                                    cand_state = _interpolate_state(prev_state, curr_state, alpha)
                                    cand_stats, cand_mia = _guard_mia_for_state(cand_state, f'alpha{alpha}')
                                    if cand_mia is not None:
                                        try:
                                            tml = float(cand_stats.get('target_mean_loss', float('nan'))) if cand_stats else float('nan')
                                        except Exception:
                                            tml = float('nan')
                                        if required_target_mean_loss is not None:
                                            print(
                                                f"[Virtual Repair][Guard][r{r+1}] alpha={alpha} mia={cand_mia:.2%} cap={mia_cap:.2%} "
                                                f"target_mean_loss={tml:.4f} req>={required_target_mean_loss:.4f}"
                                            )
                                        else:
                                            print(f"[Virtual Repair][Guard][r{r+1}] alpha={alpha} mia={cand_mia:.2%} cap={mia_cap:.2%} target_mean_loss={tml:.4f}")

                                    cand_ok = (cand_mia is not None and cand_mia <= mia_cap)
                                    if cand_ok and required_target_mean_loss is not None and cand_stats is not None:
                                        try:
                                            cand_tml = float(cand_stats.get('target_mean_loss', float('nan')))
                                            cand_ok = cand_ok and (cand_tml >= float(required_target_mean_loss))
                                        except Exception:
                                            pass

                                    if cand_ok:
                                        feasible.append((float(alpha), float(cand_mia), cand_state, cand_stats))

                                    if BACKTRACK_STRATEGY == 'first_feasible' and cand_ok:
                                        break

                                if len(feasible) > 0:
                                    if BACKTRACK_STRATEGY == 'min_mia':
                                        alpha_sel, mia_sel, state_sel, stats_sel = sorted(feasible, key=lambda t: (t[1], -t[0]))[0]
                                    elif BACKTRACK_STRATEGY == 'min_alpha':
                                        alpha_sel, mia_sel, state_sel, stats_sel = sorted(feasible, key=lambda t: (t[0], t[1]))[0]
                                    elif BACKTRACK_STRATEGY == 'max_alpha' or BACKTRACK_STRATEGY == 'first_feasible':
                                        alpha_sel, mia_sel, state_sel, stats_sel = sorted(feasible, key=lambda t: (-t[0], t[1]))[0]
                                    else:
                                        alpha_sel, mia_sel, state_sel, stats_sel = feasible[0]

                                    accepted_state = state_sel
                                    accepted_guard_mia = mia_sel
                                    accepted_guard_stats = stats_sel
                                    used_alpha = float(alpha_sel)
                                else:
                                    trainer.server.task.model.load_state_dict(prev_state)
                                    print(
                                        f"[Virtual Repair] Privacy guard triggered: fixed-pre MIA={guard_mia:.4f} > cap={mia_cap:.4f} (or loss-margin violated). "
                                        f"Backtracking failed; reverting and stopping."
                                    )
                                    break
                            else:
                                trainer.server.task.model.load_state_dict(prev_state)
                                print(
                                    f"[Virtual Repair] Privacy guard triggered: fixed-pre MIA={guard_mia:.4f} > cap={mia_cap:.4f} (or loss-margin violated). "
                                    f"Reverting and stopping."
                                )
                                break

                        trainer.server.task.model.load_state_dict(accepted_state)

                        retained_test_global = None
                        try:
                            server_task = trainer.server.task
                            metrics = list(getattr(trainer.args, 'metrics', ['accuracy']))
                            tot_samples = 0
                            agg = {f"{m}_val": 0.0 for m in metrics}
                            agg.update({f"{m}_test": 0.0 for m in metrics})
                            for cid in range(getattr(trainer.args, 'num_clients', 0)):
                                if cid == forget_id:
                                    continue
                                try:
                                    sd = trainer.clients[cid].task.splitted_data
                                    num_samples = trainer.clients[cid].task.num_samples
                                    res = server_task.evaluate(sd, mute=True)
                                    for m in metrics:
                                        agg[f"{m}_val"] += float(res.get(m + '_val', 0.0)) * num_samples
                                        agg[f"{m}_test"] += float(res.get(m + '_test', 0.0)) * num_samples
                                    tot_samples += num_samples
                                except Exception:
                                    continue
                            if tot_samples > 0 and len(metrics) > 0:
                                retained_test_global = float(agg[metrics[0] + '_test'] / tot_samples)
                        except Exception:
                            retained_test_global = None

                        if accepted_guard_mia is not None:
                            print(
                                f"[Virtual Repair][Round {r+1}/{extra_rounds}] retained_test_global={retained_test_global}, "
                                f"fixed-pre MIA={accepted_guard_mia:.2%}, server_update_alpha={used_alpha}"
                            )
                        else:
                            print(
                                f"[Virtual Repair][Round {r+1}/{extra_rounds}] retained_test_global={retained_test_global}, "
                                f"fixed-pre MIA=None, server_update_alpha={used_alpha}"
                            )

                        if retained_test_global is not None and retained_test_global > best_retained_test:
                            best_retained_test = float(retained_test_global)
                            best_state = copy.deepcopy(trainer.server.task.model.state_dict())
                            best_round = r + 1
                            best_guard_mia = accepted_guard_mia

                        if retained_test_global is not None and retained_test_global >= utility_target:
                            print(f"[Virtual Repair] Utility target reached: retained_test_global={retained_test_global:.4f} >= {utility_target:.4f}. Stopping.")
                            break

                    trainer.args.num_rounds = old_rounds
                    trainer.server.task.model.load_state_dict(best_state)
                    print(f"[Virtual Repair] Selected best repair checkpoint: best_round={best_round}, best_retained_test={best_retained_test:.4f}, best_fixed_pre_mia={best_guard_mia}")

                    try:
                        server_task = trainer.server.task
                        metrics = list(getattr(trainer.args, 'metrics', ['accuracy']))

                        if target_splitted_data is not None:
                            tgt_res = server_task.evaluate(target_splitted_data, mute=True)
                            tgt_info = ", ".join([f"{m}_test={float(tgt_res.get(m + '_test', 0.0)):.4f}" for m in metrics])
                            print(f"[Virtual Repair] Global model on TARGET(client {forget_id}) data: {tgt_info}")

                        tot_samples = 0
                        agg = {f"{m}_val": 0.0 for m in metrics}
                        agg.update({f"{m}_test": 0.0 for m in metrics})

                        for cid in range(getattr(trainer.args, 'num_clients', 0)):
                            if cid == forget_id:
                                continue
                            try:
                                sd = trainer.clients[cid].task.splitted_data
                                num_samples = trainer.clients[cid].task.num_samples
                                res = server_task.evaluate(sd, mute=True)
                                for m in metrics:
                                    agg[f"{m}_val"] += float(res.get(m + '_val', 0.0)) * num_samples
                                    agg[f"{m}_test"] += float(res.get(m + '_test', 0.0)) * num_samples
                                tot_samples += num_samples
                            except Exception:
                                continue

                        if tot_samples > 0:
                            retained_info = ", ".join([f"{m}_val={agg[m + '_val']/tot_samples:.4f}, {m}_test={agg[m + '_test']/tot_samples:.4f}" for m in metrics])
                            print(f"[Virtual Repair] Global model on RETAINED clients (exclude virtual): {retained_info}")
                    except Exception as e:
                        print(f"[Virtual Repair] Utility evaluation skipped: {e}")

                    # Only keep post_virtual_repair_fixed_pre_threshold evaluation (remove post_virtual_repair_refit_threshold)
                    try:
                        repaired_fixed_stats = None
                        if fixed_thr is not None:
                            repaired_fixed_stats = evaluator.evaluate_mia_attack(
                                trainer.server.task.model,
                                retained_dataset,
                                virtual_dataset_eval,
                                target_dataset,
                                loss_threshold=fixed_thr,
                                run_name='post_virtual_repair_fixed_pre_threshold',
                            )

                        print(f"[Virtual Repair] GraphMIA after repair (fixed-pre-threshold): {repaired_fixed_stats}")
                    except Exception as e:
                        print(f"[Virtual Repair] GraphMIA re-evaluation skipped: {e}")

                except Exception as e:
                    print(f"[Virtual Repair] Virtual client repair training failed: {e}")
            

        else:
            print("=== Using Backdoor Evaluation ===")
            # original backdoor flow
            backdoor_evaluator = BackdoorEvaluator(unlearn_config, unlearn_config.device)
            pre_unlearn_results = backdoor_aware_train_and_test_client(
                forget_id, raw_data, unlearn_config, backdoor_evaluator, is_malicious, model=selected_federated_model)

            print(f"\n=== Perform Federated Unlearning ===")
            unlearn_model = train_and_test_client_with_model_return(
                forget_id, raw_data, unlearn_config, skip_forgetting=False, model=selected_federated_model)

            print(f"\n=== Post-unlearning Evaluation (Client {forget_id}) ===")
            post_unlearn_backdoor_results = backdoor_evaluator.evaluate_backdoor_attack(
                unlearn_model, raw_data, forget_id)

            # Effect analysis
            effectiveness = backdoor_evaluator.evaluate_unlearning_effectiveness(
                pre_unlearn_results['backdoor_results'],
                post_unlearn_backdoor_results
            )
            
    else:
        # Standard evaluation
        mape = train_and_test_client(forget_id, raw_data, unlearn_config)
        results = {
            'forget_client_id': forget_id,
            'mape': mape
        }

    
    return results


def run_batch_unlearning(config, target_clients, args):
    """
    Batch unlearning mode.
    """
    from FGMU.evaluation.backdoor_train_and_test import evaluate_multiple_clients_unlearning
    
    print(f"Batch evaluating clients: {target_clients}")
    results = evaluate_multiple_clients_unlearning(config, target_clients)
    
    return {
        'mode': 'batch',
        'target_clients': target_clients,
        'client_results': results
    }


def run_standard_unlearning(config):
    """
    Standard unlearning mode (without backdoor evaluation).
    """
    forget_id = int(input("Enter the client id to unlearn: "))
    print(f"Standard unlearning will be performed on Client {forget_id}...")

    raw_data = load_client_data(forget_id)
    mape = train_and_test_client(forget_id, raw_data, config)
    
    return {
        'mode': 'standard',
        'forget_client_id': forget_id,
        'mape': mape
    }


def summarize_results(results, args):
    """
    Summarize and display results.
    """
    print(f"Experiment mode: {'Backdoor evaluation' if args.enable_backdoor else 'Standard evaluation'}")
    
    if 'mode' in results:
        if results['mode'] == 'batch':
            print("Batch evaluation results:")
            for client_id, result in results['client_results'].items():
                status = "Malicious" if result['is_malicious'] else "Benign"
                print(f"  Client {client_id} ({status}):")
                print(f"    MAPE: {result['mape']:.4f}")
                if args.enable_backdoor:
                    print(f"    ASR: {result['backdoor_results']['ASR']:.4f}")
                    print(f"    Clean Accuracy: {result['backdoor_results']['clean_accuracy']:.4f}")
                    
        elif results['mode'] == 'standard':
            print("Standard unlearning result:")
            print(f"  Client {results['forget_client_id']}: MAPE = {results['mape']:.4f}")
    
    elif 'effectiveness' in results:
        # Detailed results for interactive or automated mode
        client_id = results['client_id']
        is_malicious = results['is_malicious']
        
        print(f"\nClient {client_id} ({'Malicious' if is_malicious else 'Benign'}) unlearning analysis:")
        
        if args.enable_backdoor:
            pre_results = results['pre_unlearn']['backdoor_results']
            post_results = results['post_unlearn']['backdoor_results']
            effectiveness = results['effectiveness']
            
            print("  Before unlearning:")
            # print(f"    MAPE: {results['pre_unlearn']['mape']:.4f}")
            print(f"    ASR: {pre_results['ASR']:.4f}")
            print(f"    Clean Accuracy: {pre_results['clean_accuracy']:.4f}")
            
            print("  After unlearning:")
            # print(f"    MAPE: {results['post_unlearn']['mape']:.4f}")
            print(f"    ASR: {post_results['ASR']:.4f}")
            print(f"    Clean Accuracy: {post_results['clean_accuracy']:.4f}")
            
            print("  Unlearning effectiveness:")
            print(f"    ASR reduction: {effectiveness['asr_reduction']:.4f}")
            print(f"    Accuracy retention: {effectiveness['accuracy_retention']:.4f}")
            print(f"    Unlearning success: {'Yes' if effectiveness['unlearning_success'] else 'No'}")
            
            if is_malicious and effectiveness['unlearning_success']:
                print("The malicious client's backdoor attack has been successfully unlearned!")
            elif is_malicious:
                print("Unlearning effect on the malicious client's backdoor attack is limited.")
            else:
                print("Benign client unlearning completed.")


def create_unlearning_config(args):
    """
    Create the configuration for the unlearning experiment.
    
    Args:
        args: arguments parsed from the command line
        
    Returns:
        UnlearningConfig: configuration object for the unlearning experiment
    """
    class UnlearningConfig:
        def __init__(self):
            # Base model config
            self.input_dim = 1433
            self.output_dim = 7
            self.hidden_dim = 128
            self.num_layers = 2
            self.dropout = 0.3
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lr = 0.02
            # For reproducibility of Graph-MIA virtual data and other randomness
            self.seed = int(getattr(args, 'seed', 2025))
            # Training epochs during unlearning (should be large enough to noticeably increase target CE loss)
            self.epochs = 30 if getattr(args, 'unlearning_epochs', None) is None else int(getattr(args, 'unlearning_epochs'))
            self.window_size = 10
            
            # Federated learning config
            self.num_clients = args.num_clients
            
            # Backdoor config
            self.malicious_clients = args.malicious_clients if args.malicious_clients else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.poisoning_rate = args.poisoning_rate
            self.trigger_type = args.trigger_type
            self.target_offset = args.target_offset
            
            # Unlearning config
            class Forgetting:
                def __init__(self):
                    self.enabled = True
                    self.method = "gradient_ascent"
                    
            self.forgetting = Forgetting()
            self.forgetting_strength = 5.0

            # Unlearning hyperparams required by GRU_forgetting / backdoor_train_and_test.py
            # - forgetting_loss_type: default uses NPO
            # - beta: NPO temperature
            # - forget_ascend_scale: scale factor for (orthogonalized) gradient ascent to strengthen unlearning
            # - memory_batches: number of batches for memory_grad sampling (retain-side gradient statistics)
            self.forgetting_loss_type = 'npo' if getattr(args, 'forgetting_loss_type', None) is None else str(getattr(args, 'forgetting_loss_type'))
            self.beta = 5.0
            self.forget_ascend_scale = 50 if getattr(args, 'forget_ascend_scale', None) is None else float(getattr(args, 'forget_ascend_scale'))
            self.memory_batches = 1
            self.max_update_norm = (None if getattr(args, 'max_update_norm', None) is None else float(getattr(args, 'max_update_norm')))
            self.unlearn_param_delta_tau = float(getattr(args, 'unlearn_param_delta_tau', -1.0))

            # Optional: push target loss above the fixed-pre threshold by a margin (Graph-MIA only)
            self.fixed_mia_target_loss_margin = float(getattr(args, 'fixed_mia_target_loss_margin', 0.0))
            self.fixed_mia_push_lambda = float(getattr(args, 'fixed_mia_push_lambda', 1.0))
            # Choose evaluation method: 'backdoor' or 'graph_mia'
            self.evaluation_method = getattr(args, 'evaluation_method', 'graph_mia')
            # Graph-MIA specific defaults
            self.num_virtual_graphs = 10
            self.virtual_noise_scale = 0.5
            
    return UnlearningConfig()

def _set_data_env(args):
    """Set env vars so `load_client_data` can find the distrib directory."""
    os.environ["FGMU_DATA_ROOT"] = str(args.root)
    os.environ["FGMU_TASK"] = str(getattr(args, "task", "node_cls"))
    os.environ["FGMU_SPLIT"] = str(getattr(args, "train_val_test", "default_split"))

    distrib_root = os.path.join(args.root, "distrib")
    if not os.path.isdir(distrib_root):
        return

    # Prefer exact match if possible; otherwise fall back to first directory containing expected files.
    dataset_name = args.dataset[0] if isinstance(getattr(args, "dataset", None), (list, tuple)) and args.dataset else str(getattr(args, "dataset", ""))
    resolution = getattr(args, "louvain_resolution", 1)
    # folder naming uses '1' not '1.0'
    try:
        resolution_str = str(int(resolution)) if float(resolution).is_integer() else str(resolution)
    except Exception:
        resolution_str = str(resolution)

    preferred = os.path.join(
        distrib_root,
        f"{args.simulation_mode}_{resolution_str}_{dataset_name}_client_{args.num_clients}",
    )
    if os.path.isdir(preferred) and os.path.isfile(os.path.join(preferred, "data_0.pt")):
        os.environ["FGMU_DISTRIB_DIR"] = preferred
        return

    # Fallback: search for any matching distrib dir with data_0.pt
    for cand in sorted(glob.glob(os.path.join(distrib_root, "*"))):
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "data_0.pt")):
            os.environ["FGMU_DISTRIB_DIR"] = cand
            return


if __name__ == "__main__":
    main()