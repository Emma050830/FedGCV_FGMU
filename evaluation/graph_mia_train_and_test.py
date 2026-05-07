import copy
import torch
from FGMU.evaluation.graph_mia_evaluation import GraphMIAEvaluator
from FGMU.client.train_and_test import load_client_data
from FGMU.evaluation.backdoor_train_and_test import validate_and_extract_raw_data
from FGMU.evaluation.backdoor_train_and_test import train_and_test_client_with_model_return
from FGMU.fl_model.trainer import FGLTrainer

def graph_mia_aware_train_and_test_client(client_id, raw_data, config, model=None):
    """
    Graph-MIA-aware unlearning pipeline for a single client:
    - Evaluate a baseline model (pre-unlearning) for MIA.
    - Perform unlearning via `train_and_test_client_with_model_return`.
    - Re-evaluate MIA after unlearning.

    Note: We intentionally keep ONLY the fixed-pre-threshold evaluation (loss threshold fitted on the pre model)
    to make pre/post comparisons consistent. The refit-threshold evaluation is removed.
    """
    print(f"=== GraphMIA-aware training for Client {client_id} ===")
    # Validate and normalize the raw data into a unified train/test format
    normalized = validate_and_extract_raw_data(raw_data, client_id)
    training = normalized['train']
    test = normalized['test']

    # 1) Get the pre-unlearning model.
    # IMPORTANT: For Graph-MIA, the member/non-member split must match what the model has actually trained on.
    # - If a global/aggregated model is provided (model is not None), we use it as `pre_model` by default to avoid
    #   locally training on the target client's data, which would invalidate the retained/non-member distribution.
    # - If no model is provided, we fall back to the legacy behavior: train a local baseline model for this client.
    use_existing_model = (model is not None) and getattr(config, 'graph_mia_use_existing_model', True)
    if use_existing_model:
        pre_model = copy.deepcopy(model)
    else:
        pre_model = train_and_test_client_with_model_return(
            client_id,
            {'train': training, 'test': test},
            config,
            skip_forgetting=True,
            model=model,
        )

    model_for_unlearning = copy.deepcopy(pre_model)

    oracle_model = pre_model
    oracle_model.eval()
    for param in oracle_model.parameters():
        param.requires_grad = False

    # Build the evaluator (wrapper around GraphMIA)
    evaluator = GraphMIAEvaluator(config, device=config.device)

    # Prepare datasets for MIA:
    # - retained_dataset: member data (typically training data from all other clients)
    # - virtual_dataset: synthetic non-member data (for member vs non-member separation)
    # - target_dataset: the target client's data to be forgotten (train + test)
    # Build target_dataset: include all of this client's data (train + test)
    target_dataset = []
    if isinstance(training, (list, tuple)):
        target_dataset.extend(list(training))
    else:
        target_dataset.append(training)
    if test is not None:
        if isinstance(test, (list, tuple)):
            for t in test:
                if t not in target_dataset:
                    target_dataset.append(t)
        else:
            if test not in target_dataset:
                target_dataset.append(test)

    # Build retained_dataset: use other clients' training data (exclude current client)
    retained_dataset = []
    num_clients = getattr(config, 'num_clients', None)
    if isinstance(num_clients, int) and num_clients > 0:
        for cid in range(num_clients):
            if cid == client_id:
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
                # If one client fails to load, skip it without breaking the whole pipeline
                continue

    # If other clients' data cannot be collected, fall back to target_dataset (to keep the pipeline runnable)
    if len(retained_dataset) == 0:
        retained_dataset = list(target_dataset)

    virtual_dataset = evaluator.generate_virtual_dataset_from(
        retained_dataset,
        num_virtual=getattr(config, 'num_virtual_graphs', 10),
        noise_scale=getattr(config, 'virtual_noise_scale', 0.5)
    )

    # MIA evaluation on the pre-unlearning model (baseline)
    pre_stats = evaluator.evaluate_mia_attack(
        pre_model,
        retained_dataset,
        virtual_dataset,
        target_dataset,
        run_name='pre',
    )

    fixed_threshold = None
    if isinstance(pre_stats, dict):
        fixed_threshold = pre_stats.get('loss_threshold', None)
    try:
        setattr(config, 'fixed_mia_loss_threshold', fixed_threshold)
    except Exception:
        pass

    # 2) Perform unlearning (skip_forgetting=False).
    # Pass retained_dataset so the unlearning controller can compute memory gradients on retained (non-target) data.
    unlearn_model = train_and_test_client_with_model_return(
        client_id,
        {'train': training, 'test': test, 'retain': retained_dataset},
        config,
        skip_forgetting=False,
        model=model_for_unlearning,
        pre_model=oracle_model
    )

    # Post-unlearning MIA evaluation: ONLY fixed-pre-threshold (threshold from pre model)
    post_fixed_stats = None
    if fixed_threshold is not None and fixed_threshold == fixed_threshold:
        post_fixed_stats = evaluator.evaluate_mia_attack(
            unlearn_model,
            retained_dataset,
            virtual_dataset,
            target_dataset,
            loss_threshold=fixed_threshold,
            run_name='post_fixed_pre_threshold',
        )

    # Summarize unlearning effectiveness (based on fixed-pre-threshold only)
    effectiveness = None
    if isinstance(pre_stats, dict) and isinstance(post_fixed_stats, dict):
        effectiveness = evaluator.evaluate_unlearning_effectiveness(pre_stats, post_fixed_stats)

    results = {
        'client_id': client_id,
        'model': unlearn_model,
        'pre_mia': pre_stats,
        # NOTE: refit-threshold evaluation has been removed intentionally.
        'post_mia': None,
        'post_mia_fixed_threshold': post_fixed_stats,
        'fixed_threshold': fixed_threshold,
        'effectiveness': effectiveness
    }

    def _pct(x):
        try:
            return f"{float(x) * 100:.2f}%"
        except Exception:
            return "nan"

    def _f(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "nan"

    pre_mia = pre_stats.get('mia_rate') if isinstance(pre_stats, dict) else None
    post_fixed_mia = post_fixed_stats.get('mia_rate') if isinstance(post_fixed_stats, dict) else None

    print(
        f"[Client {client_id}] GraphMIA summary: "
        f"pre(mia={_pct(pre_mia)}, thr={_f(pre_stats.get('loss_threshold'))}, tgt_loss={_f(pre_stats.get('target_mean_loss'))}) | "
        f"post_fixed_pre(mia={_pct(post_fixed_mia)}, thr={_f(fixed_threshold)}, tgt_loss={_f((post_fixed_stats or {}).get('target_mean_loss'))}) | "
        f"Δ(pre→post_fixed)={_pct((pre_mia or 0.0) - (post_fixed_mia or 0.0))}"
    )

    if bool(getattr(config, 'debug', False)):
        print(
            f"[Client {client_id}] GraphMIA details: pre={pre_stats}, "
            f"post_fixed_pre={post_fixed_stats}, effectiveness={effectiveness}"
        )
    return results