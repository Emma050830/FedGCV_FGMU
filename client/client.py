from FGMU.client.GRU_forgetting import GRU_ForgettingController
import torch
import torch.nn as nn

def add_gaussian_noise(tensor, noise_scale):
    noise = torch.normal(mean=0, std=noise_scale, size=tensor.size()).to(tensor.device)
    return tensor + noise

class ClientClass:
    def __init__(self, config):
        # Initialize forgetting controller
        self.forget_controller = GRU_ForgettingController(config)
        self.forgetting_enabled = config.forgetting.enabled

    def local_train(self, model, data):  
        """Run local training with optional forgetting.

        This includes gradient bookkeeping for the forgetting mechanism (storing gradients,
        orthogonal projection, applying parameter masks, etc.).

        Steps:
        - Backup original parameters.
        - Standard training and store objective gradients (for forgetting).
        - Train again and store retain gradients.
        - If forgetting is enabled, compute forgetting scores/masks, orthogonalize gradients,
            restore parameters, apply the forgetting mask, and train again.
        """
        original_params = {n: p.clone() for n, p in model.named_parameters()}
                # Standard training
        loss_obj = self._train_epoch(model, data, return_tensor=True)  # return tensor

        # Store and flatten gradients
        self.forget_controller.store_grads(model, loss_obj, typ="objective")
        self.forget_controller.flatten_and_store_grads()
        model.zero_grad()  # clear grads and graph
        loss_obj = None  # detach graph
        del loss_obj

        # Forward again to get a fresh loss tensor for retain gradients
        loss_retain = self._train_epoch(model, data, return_tensor=True)
        self.forget_controller.store_grads(model, loss_retain, typ="retain")  # retain gradients
        self.forget_controller.flatten_and_store_grads()
        model.zero_grad()  # clear again
        loss_retain = None  # detach graph
        del loss_retain

        if self.forgetting_enabled:
            # Compute forgetting mask
            scores = self.forget_controller.compute_forgetting_scores(data)
            masks = self.forget_controller.generate_forgetting_mask(
                original_params, scores)

            # Orthogonalize gradients and restore (assumes memory_grad stores retain gradients)
            if self.forget_controller.flattened_memory_accumulation is not None and self.forget_controller.flattened_memory_accumulation.numel() > 0:
                ortho_grad = self.forget_controller.orthogonal_component_precise(
                    self.forget_controller.flattened_gradient,
                    self.forget_controller.flattened_memory_accumulation)
                self.forget_controller.flattened_gradient = ortho_grad
                self.forget_controller.restore_gradients_from_flat(model)

            # === Gradient-ascent forgetting ===
            # Maximize loss on the forget sample (i.e., backprop through -loss)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            inputs = data.unsqueeze(0).float()
            targets = data[-1].unsqueeze(0)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            # Gradient ascent: maximize the loss
            (-loss).backward()
            optimizer.step()

            # Optional: apply forgetting mask
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data = original_params[name] * masks[name]

            # Optional: return loss after unlearning
            loss_after_unlearn = self._train_epoch(model, data, return_tensor=True)
            return loss_after_unlearn

        # If forgetting is disabled, return the standard training loss
        return self._train_epoch(model, data, return_tensor=True)
    
    def _train_epoch(self, model, data, return_tensor=False):
        """Run a single training epoch."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        # data is assumed to be [seq_len, features]; build a batch
        inputs = data.unsqueeze(0)  # [1, seq_len, features]
        inputs = inputs.float()
        targets = data[-1].unsqueeze(0)  # [1, features]
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        if return_tensor:
            return loss  # tensor
        return loss.item()  # float
    
    def get_noisy_model_params(self, model, noise_scale=1e-2):
        state_dict = model.state_dict()
        noisy_state_dict = {}
        for k, v in state_dict.items():
            noisy_state_dict[k] = add_gaussian_noise(v, noise_scale)
        return noisy_state_dict

    def upload_model(self, model):
        # Add Gaussian noise before uploading model parameters
        noise_scale = self.config.dp_noise_scale if hasattr(self.config, "dp_noise_scale") else 1e-2
        noisy_params = self.get_noisy_model_params(model, noise_scale)
        # Upload noisy_params to the server
        return noisy_params