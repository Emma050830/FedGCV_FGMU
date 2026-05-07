import torch
from FGMU.evaluation.Graph_MIA import GraphMIA

class GraphMIAEvaluator:
    """
    A thin wrapper around GraphMIA (FGMU/evaluation/Graph_MIA.py).

    Provides an interface similar to BackdoorEvaluator for easy integration in the main pipeline.
    """
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or getattr(config, 'device', torch.device('cpu'))
        self.device = torch.device(self.device if isinstance(self.device, str) else self.device)
        # GraphMIA expects a model; we create an instance on demand for evaluation
        # (GraphMIA does not keep additional state beyond the model).
    
    def evaluate_mia_attack(self, model, retained_dataset, virtual_dataset, target_dataset, loss_threshold=None, run_name=None, verbose: bool = True):
        """
        Run a loss-based MIA using GraphMIA.

        retained_dataset / virtual_dataset / target_dataset: iterables of PyG Data objects.
        Returns a dict containing basic statistics such as 'mia_rate'.
        """
        gm = GraphMIA(model, device=self.device)
        stats = gm.evaluate_unlearning(
            target_data=target_dataset,
            retained_data=retained_dataset,
            test_data=virtual_dataset,
            loss_threshold=loss_threshold,
            run_name=run_name,
            verbose=verbose,
        )
        # Backward compatibility: always return a dict with 'mia_rate'
        if isinstance(stats, dict) and 'mia_rate' in stats:
            return stats
        return {'mia_rate': float(stats)}
    
    def evaluate_unlearning_effectiveness(self, original_stats, post_unlearn_stats):
        """
        Generate a summary similar to BackdoorEvaluator.evaluate_unlearning_effectiveness.

        original_stats and post_unlearn_stats are dicts returned by evaluate_mia_attack.
        """
        res = {}
        orig = original_stats.get('mia_rate', 0.0)
        post = post_unlearn_stats.get('mia_rate', 0.0)
        res['mia_reduction'] = orig - post
        res['unlearning_success'] = (res['mia_reduction'] > 0.1)
        return res

    def generate_virtual_dataset_from(self, real_dataset, num_virtual=10, noise_scale=0.01, seed=None):
        """
        A simple placeholder to generate a "virtual" subgraph set for MIA evaluation.

        In practice, this should be replaced by a VGAE-based generator that produces synthetic
        subgraphs with similar characteristics but without leaking real data.

        This simple implementation clones real subgraphs and adds small feature noise.
        """
        # Reduce variance: add reproducible noise to virtual data (fixed seed).
        if seed is None:
            seed = getattr(self.config, 'seed', 0)

        vset = []
        for i, d in enumerate(real_dataset):
            if i >= num_virtual:
                break
            dcopy = d.clone().to(self.device)
            if hasattr(dcopy, 'x') and dcopy.x is not None:
                gen = torch.Generator(device=dcopy.x.device)
                gen.manual_seed(int(seed) + int(i))
                noise = torch.randn(dcopy.x.shape, device=dcopy.x.device, generator=gen) * float(noise_scale)
                dcopy.x = dcopy.x + noise
            vset.append(dcopy)
        return vset