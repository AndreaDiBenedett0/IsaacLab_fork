import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as BasePPO_RNN

from skrl.agents.torch import Agent  # per chiamare direttamente il base class in record_transition

from skrl import config  # usato per distributed reduce / scheduler
import itertools
import torch.nn.functional as F

from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# from isaaclab.app import AppLauncher
# app = AppLauncher(headless=False, experience="isaaclab.python.kit").app


# ========= WRAPPER DELL'AGENTE: PPO_RNN con record_transition "safe" =========

class PPO(BasePPO_RNN):
    """Sottoclasse di PPO_RNN che:
       - forza values a (N, 1) e NON usa inverse=True sul value preprocessor in recording
       - normalizza le forme in _update per prevenire mismatch nel GAE
    """

    @torch.no_grad()
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps) -> None:
        # chiamata al base Agent (non al PPO_RNN del parent)
        Agent.record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is None:
            return

        self._current_next_states = next_states

        # reward shaping (se configurato)
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        # compute values (critic)
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            values, _, outputs = self.value.act({"states": self._state_preprocessor(states), **rnn}, role="value")

        # --- guardrail forma: deve essere (N, 1) ---
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        elif values.shape[-1] != 1:
            values = values[:, :1].contiguous()

        # preprocessor dei values in forward (NO inverse durante il recording)
        if self._value_preprocessor:
            values = self._value_preprocessor(values)

        # time-limit bootstrapping (se abilitato)
        if self._time_limit_bootstrap:
            rewards = rewards + self._discount_factor * values * truncated

        # package RNN states (come SKRL)
        rnn_states = {}
        if self._rnn:
            rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])})
            if self.policy is not self.value:
                rnn_states.update({f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])})

        # scrittura in memoria (ora values è (N, 1))
        self.memory.add_samples(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=values,
            **rnn_states,
        )
        for memory in self.secondary_memories:
            memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                **rnn_states,
            )

        # update RNN states finali e reset per episodi conclusi (come SKRL)
        if self._rnn:
            self._rnn_final_states["value"] = (
                self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
            )
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0
            self._rnn_initial_states = self._rnn_final_states

    # =========================
    #  OVERRIDE DI _update
    # =========================
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step con normalizzazione delle forme per GAE"""

        # helper per garantire forma (T, 1)
        def _ensure_col(x: torch.Tensor | None) -> torch.Tensor | None:
            if x is None:
                return None
            if x.ndim == 1:
                return x.unsqueeze(-1)
            if x.shape[-1] != 1:
                return x[..., :1].contiguous()
            return x

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ):
            """Compute GAE con forme coerenti (T, 1)"""
            # tutte (T, 1)
            rewards = _ensure_col(rewards)
            dones = _ensure_col(dones)
            values = _ensure_col(values)
            last_values = _ensure_col(last_values)

            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            T = rewards.shape[0]

            for i in reversed(range(T)):
                nv = values[i + 1] if i < T - 1 else last_values
                print("DEBUG PPO_RNN._update.compute_gae: nv", nv)
                print("DEBUG PPO_RNN._update.compute_gae: values[i]", values[i])
                print("DEBUG PPO_RNN._update.compute_gae: rewards[i]", rewards[i])
                print("DEBUG PPO_RNN._update.compute_gae: not_dones[i]", not_dones[i])
                print("DEBUG PPO_RNN._update.compute_gae: advantage before", advantage)
                print("DEBUG PPO_RNN._update.compute_gae: discount_factor", discount_factor)
                print("DEBUG PPO_RNN._update.compute_gae: lambda_coefficient", lambda_coefficient)
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (nv + lambda_coefficient * advantage)
                advantages[i] = advantage

            returns = advantages + values
            # normalize advantages (per-feature: qui è 1 colonna)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value"
            )
            self.value.train(True)
            if self._value_preprocessor:
                last_values = self._value_preprocessor(last_values, inverse=True)
            # forza (T, 1)
            last_values = _ensure_col(last_values)

        print("DEBUG PPO_RNN._update before col: values", self.memory.get_tensor_by_name("values"))
        print("DEBUG PPO_RNN._update before col: rewards", self.memory.get_tensor_by_name("rewards"))
        print("DEBUG PPO_RNN._update before col: dones", self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"))
        print("DEBUG PPO_RNN._update before col: terminated", self.memory.get_tensor_by_name("terminated"))
        print("DEBUG PPO_RNN._update before col: truncated", self.memory.get_tensor_by_name("truncated"))

        # prendi i tensori dalla memoria e forza (T, 1)
        values = _ensure_col(self.memory.get_tensor_by_name("values"))
        rewards = _ensure_col(self.memory.get_tensor_by_name("rewards"))
        dones = _ensure_col(self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"))

        print("DEBUG PPO_RNN._update after: values", values)
        print("DEBUG PPO_RNN._update after: rewards", rewards)
        print("DEBUG PPO_RNN._update after: dones", dones)
        print("DEBUG PPO_RNN._update after: last_values", trema)

        # GAE
        returns, advantages = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        # riscrivi in memoria (coerente con skrl)
        if self._value_preprocessor:
            self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
            self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        else:
            self.memory.set_tensor_by_name("values", values)
            self.memory.set_tensor_by_name("returns", returns)
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches (identico a skrl)
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length
        )

        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for i, (sampled_states,
                    sampled_actions,
                    sampled_terminated,
                    sampled_truncated,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages) in enumerate(sampled_batches):

                # RNN packages
                rnn_policy, rnn_value = {}, {}
                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = {
                            "rnn": [s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "value" in n],
                            "terminated": sampled_terminated | sampled_truncated,
                        }

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # log-prob aggiornato (policy)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
                    )

                    # approx KL
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stop KL
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0.0

                    # policy loss (clipped)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")
                    # forza (.., 1) per sicurezza
                    predicted_values = _ensure_col(predicted_values)
                    sampled_values = _ensure_col(sampled_values)
                    sampled_returns = _ensure_col(sampled_returns)

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                 self._grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(entropy_loss) if isinstance(entropy_loss, float) else entropy_loss.item()

            # scheduler
            if self._learning_rate_scheduler:
                if hasattr(self, "scheduler") and self.scheduler is not None and self.scheduler.__class__.__name__ == "KLAdaptiveLR":
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # logging (TensorBoard)
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss",  cumulative_value_loss  / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
# ========= FINE WRAPPER =========




# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,  # era False
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=200, num_layers=2, hidden_size=128, sequence_length=300):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D * num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D * num_layers, N, Hcell)

    
    def compute(self, inputs, role):
        states = inputs["states"]                     # (B*L, obs)
        terminated = inputs.get("terminated", None)   # (B*L,) o None

        hidden_states = inputs["rnn"][0].contiguous()
        cell_states   = inputs["rnn"][1].contiguous()

        # (opzionale ma consigliato) assicurare stesso dtype dell'input alla LSTM
        # utile con autocast/mixed precision
        # NB: rnn_input verrà creato più sotto, quindi uso states.dtype direttamente
        hidden_states = hidden_states.to(states.dtype)
        cell_states   = cell_states.to(states.dtype)

        
        # hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]  # (num_layers, B, H)

        if self.training:
            B = hidden_states.shape[1]
            N = states.shape[0]
            assert N % B == 0, f"[Policy] N={N} non divisibile per B={B}"
            L = N // B

            # usa reshape al posto di view (più robusto con non-contiguous)
            rnn_input = states.reshape(B, L, states.shape[-1])
            if terminated is not None:
                terminated = terminated.reshape(B, L)

            rnn_output, (h, c) = self.lstm(rnn_input, (hidden_states, cell_states))

            # opzionale: azzera stati se la sequenza è terminata all'ultimo step
            if terminated is not None and torch.any(terminated[:, -1]):
                done_last = terminated[:, -1]
                h[:, done_last, :] = 0
                c[:, done_last, :] = 0
        else:
            rnn_input = states.reshape(-1, 1, states.shape[-1])
            rnn_output, (h, c) = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)


        #     rnn_input = states.view(B, L, states.shape[-1])  # (B, L, Hin)
        #     if terminated is not None:
        #         terminated = terminated.view(B, L)           # (B, L)

        #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        #     # (opzionale) azzera stati per sequenze terminate all'ultimo step
        #     if terminated is not None and torch.any(terminated[:, -1]):
        #         done_last = terminated[:, -1]                # (B,)
        #         h, c = rnn_states
        #         h[:, done_last, :] = 0
        #         c[:, done_last, :] = 0
        #         rnn_states = (h, c)
        # else:
        #     # rollout/eval: L = 1
        #     rnn_input = states.view(-1, 1, states.shape[-1])  # (B, 1, Hin) con B=num_envs
        #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # # (B, L, H) -> (B*L, H)
        # rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        mean = torch.tanh(self.net(rnn_output))  # (B*L, num_actions)
        log_std = self.log_std_parameter         # (num_actions,)

        # return mean, log_std, {"rnn": [rnn_states[0], rnn_states[1]]}
        return mean, log_std, {"rnn": [h.contiguous(), c.contiguous()]}

    # def compute(self, inputs, role):
    #     states = inputs["states"]                                           # tensore di osservazioni
    #     terminated = inputs.get("terminated", None)                         # mask di terminazioni (opzionale)
    #     hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]     # stato LSTM corrente

    #     # training
    #     if self.training:
    #         # reshape inputs for RNN
    #         rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
    #         hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
    #         cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
    #         # get the hidden/cell states corresponding to the initial sequence (time step 0)
    #         hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
    #         cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

    #         # reset the RNN state in the middle of a sequence
    #         if terminated is not None and torch.any(terminated):
    #             rnn_outputs = []
    #             terminated = terminated.view(-1, self.sequence_length)
    #             indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

    #             for i in range(len(indexes) - 1):
    #                 i0, i1 = indexes[i], indexes[i + 1]
    #                 rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
    #                 hidden_states[:, (terminated[:,i1-1]), :] = 0
    #                 cell_states[:, (terminated[:,i1-1]), :] = 0
    #                 rnn_outputs.append(rnn_output)

    #             rnn_states = (hidden_states, cell_states)
    #             rnn_output = torch.cat(rnn_outputs, dim=1)
    #         else:
    #             rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
    #     # rollout (eval)
    #     else:
    #         rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
    #         rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

    #     # flatten the RNN output
    #     rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)

    #     # Action_space is -1 to 1
    #     return 1 * torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,   # era False
                 num_envs=200, num_layers=2, hidden_size=128, sequence_length=300):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 1))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D * num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D * num_layers, N, Hcell)

    
    def compute(self, inputs, role):
        states = inputs["states"]                     # (B*L, obs)
        terminated = inputs.get("terminated", None)   # (B*L,) o None

        hidden_states = inputs["rnn"][0].contiguous()
        cell_states   = inputs["rnn"][1].contiguous()
        # (opzionale ma consigliato) assicurare stesso dtype dell'input alla LSTM
        # utile con autocast/mixed precision
        # NB: rnn_input verrà creato più sotto, quindi uso states.dtype direttamente
        hidden_states = hidden_states.to(states.dtype)
        cell_states   = cell_states.to(states.dtype)
        # hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]  # (num_layers, B, H)

        if self.training:
            B = hidden_states.shape[1]
            N = states.shape[0]
            assert N % B == 0, f"[Value] N={N} non divisibile per B={B}"
            L = N // B


            # usa reshape al posto di view (più robusto con non-contiguous)
            rnn_input = states.reshape(B, L, states.shape[-1])
            if terminated is not None:
                terminated = terminated.reshape(B, L)

            rnn_output, (h, c) = self.lstm(rnn_input, (hidden_states, cell_states))

            # opzionale: azzera stati se la sequenza è terminata all'ultimo step
            if terminated is not None and torch.any(terminated[:, -1]):
                done_last = terminated[:, -1]
                h[:, done_last, :] = 0
                c[:, done_last, :] = 0
        else:
            rnn_input = states.reshape(-1, 1, states.shape[-1])
            rnn_output, (h, c) = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)


        #     rnn_input = states.view(B, L, states.shape[-1])  # (B, L, Hin)
        #     if terminated is not None:
        #         terminated = terminated.view(B, L)

        #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        #     # (opzionale) azzera stati per sequenze terminate all'ultimo step
        #     if terminated is not None and torch.any(terminated[:, -1]):
        #         done_last = terminated[:, -1]
        #         h, c = rnn_states
        #         h[:, done_last, :] = 0
        #         c[:, done_last, :] = 0
        #         rnn_states = (h, c)
        # else:
        #     rnn_input = states.view(-1, 1, states.shape[-1])  # (B, 1, Hin)
        #     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # (B, L, H) -> (B*L, H)
        # rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        value = self.net(rnn_output)                           # (B*L, 1)

        # return value, {"rnn": [rnn_states[0], rnn_states[1]]}
        return value, {"rnn": [h.contiguous(), c.contiguous()]}

    # def compute(self, inputs, role):
    #     states = inputs["states"]
    #     terminated = inputs.get("terminated", None)
    #     hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

    #     # training
    #     if self.training:
    #         rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

    #         hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
    #         cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
    #         # get the hidden/cell states corresponding to the initial sequence
    #         hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
    #         cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

    #         # reset the RNN state in the middle of a sequence
    #         if terminated is not None and torch.any(terminated):
    #             rnn_outputs = []
    #             terminated = terminated.view(-1, self.sequence_length)
    #             indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

    #             for i in range(len(indexes) - 1):
    #                 i0, i1 = indexes[i], indexes[i + 1]
    #                 rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
    #                 hidden_states[:, (terminated[:,i1-1]), :] = 0
    #                 cell_states[:, (terminated[:,i1-1]), :] = 0
    #                 rnn_outputs.append(rnn_output)

    #             rnn_states = (hidden_states, cell_states)
    #             rnn_output = torch.cat(rnn_outputs, dim=1)
    #         else:
    #             rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
    #     # rollout
    #     else:
    #         rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
    #         rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

    #     # flatten the RNN output
    #     rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)
    #     # print("DEBUG[Value compute] rnn_output shape:", tuple(rnn_output.shape))
    #     # print("DEBUG[Value compute] net output shape:", tuple(self.net(rnn_output).shape))

    #     return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}


# ========== ENV & TRAINING ==========


env = load_isaaclab_env(task_name="Isaac-Velocity-PaperFlat-G1-v0",
                        num_envs=2,
                        headless=False)
env = wrap_env(env)


try:
    # Isaac Lab espone spesso un world/viewer interno
    env.unwrapped.set_headless(False)  # se presente
except Exception:
    pass

try:
    # Alcuni task hanno una config viewer (dipende dalla versione)
    if hasattr(env.unwrapped, "task") and hasattr(env.unwrapped.task, "cfg"):
        if hasattr(env.unwrapped.task.cfg, "viewer"):
            env.unwrapped.task.cfg.viewer.enable = True
except Exception:
    pass


device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators)
models = {}
models["policy"] = Policy(env.single_observation_space, env.single_action_space, device, clip_actions=True, num_envs=env.num_envs)
models["value"]  = Value(env.single_observation_space, env.single_action_space, device, num_envs=env.num_envs)

# configure and instantiate the agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 3200 
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.single_observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 200000
cfg["experiment"]["directory"] = "runs/torch/G1_flat_paper_ppo"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 150000000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
env.close()


# import torch
# import torch.nn as nn

# # import the skrl components to build the RL system
# from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
# from skrl.agents.torch.ppo import PPO_RNN as PPO
# from skrl.envs.loaders.torch import load_isaaclab_env
# from skrl.envs.wrappers.torch import wrap_env
# from skrl.memories.torch import RandomMemory
# from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
# from skrl.resources.preprocessors.torch import RunningStandardScaler
# from skrl.resources.schedulers.torch import KLAdaptiveRL
# from skrl.trainers.torch import SequentialTrainer
# from skrl.utils import set_seed


# # seed for reproducibility
# set_seed()  # e.g. `set_seed(42)` for fixed seed

# class PPO(BasePPO_RNN):
#     """Sottoclasse di PPO_RNN che forza values a (N, 1) e NON usa inverse=True
#        sul value preprocessor in fase di recording (per evitare broadcast mismatch)."""

#     @torch.no_grad()
#     def record_transition(
#         self,
#         states: torch.Tensor,
#         actions: torch.Tensor,
#         rewards: torch.Tensor,
#         next_states: torch.Tensor,
#         terminated: torch.Tensor,
#         truncated: torch.Tensor,
#         infos,
#         timestep: int,
#         timesteps: int,
#     ) -> None:
#         # chiamata al base Agent (non al PPO_RNN del parent)
#         Agent.record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

#         if self.memory is None:
#             return

#         self._current_next_states = next_states

#         # reward shaping (se configurato)
#         if self._rewards_shaper is not None:
#             rewards = self._rewards_shaper(rewards, timestep, timesteps)

#         # compute values (critic)
#         with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
#             rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
#             values, _, outputs = self.value.act({"states": self._state_preprocessor(states), **rnn}, role="value")

#         # --- guardrail forma: deve essere (N, 1) ---
#         if values.ndim == 1:
#             values = values.unsqueeze(-1)
#         elif values.shape[-1] != 1:
#             values = values[:, :1].contiguous()

#         # preprocessor dei values in forward (NO inverse durante il recording)
#         if self._value_preprocessor:
#             values = self._value_preprocessor(values)

#         # time-limit bootstrapping (se abilitato)
#         if self._time_limit_bootstrap:
#             rewards = rewards + self._discount_factor * values * truncated

#         # package RNN states (come SKRL)
#         rnn_states = {}
#         if self._rnn:
#             rnn_states.update(
#                 {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
#             )
#             if self.policy is not self.value:
#                 rnn_states.update(
#                     {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
#                 )

#         # scrittura in memoria (ora values è (N, 1))
#         self.memory.add_samples(
#             states=states,
#             actions=actions,
#             rewards=rewards,
#             next_states=next_states,
#             terminated=terminated,
#             truncated=truncated,
#             log_prob=self._current_log_prob,
#             values=values,
#             **rnn_states,
#         )
#         for memory in self.secondary_memories:
#             memory.add_samples(
#                 states=states,
#                 actions=actions,
#                 rewards=rewards,
#                 next_states=next_states,
#                 terminated=terminated,
#                 truncated=truncated,
#                 log_prob=self._current_log_prob,
#                 values=values,
#                 **rnn_states,
#             )

#         # update RNN states finali e reset per episodi conclusi (come SKRL)
#         if self._rnn:
#             self._rnn_final_states["value"] = (
#                 self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
#             )
#             finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
#             if finished_episodes.numel():
#                 for rnn_state in self._rnn_final_states["policy"]:
#                     rnn_state[:, finished_episodes[:, 0]] = 0
#                 if self.policy is not self.value:
#                     for rnn_state in self._rnn_final_states["value"]:
#                         rnn_state[:, finished_episodes[:, 0]] = 0
#             self._rnn_initial_states = self._rnn_final_states
# # ========= FINE WRAPPER =========


# # define models (stochastic and deterministic models) using mixins
# class Policy(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=True,  # era False
#                  clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
#                  num_envs=200, num_layers=2, hidden_size=128, sequence_length=300):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         self.num_envs = num_envs
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
#         self.sequence_length = sequence_length

#         self.lstm = nn.LSTM(input_size=self.num_observations,
#                             hidden_size=self.hidden_size,
#                             num_layers=self.num_layers,
#                             batch_first=True)  # batch_first -> (batch, sequence, features)

#         self.net = nn.Sequential(nn.Linear(self.hidden_size, self.num_actions))
                                 
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def get_specification(self):
#         # batch size (N) is the number of envs
#         return {"rnn": {"sequence_length": self.sequence_length,
#                         "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
#                                   (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

#     def compute(self, inputs, role):
#         states = inputs["states"]                                           # tensore di osservazioni
#         terminated = inputs.get("terminated", None)                         # mask di terminazioni (opzionale)         
#         hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]     # stato LSTM corrente

#         # training
#         if self.training:
#             # reshape inputs for RNN
#             rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
#             hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
#             cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
#             # get the hidden/cell states corresponding to the initial sequence (time step 0)
#             hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
#             cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

#             # reset the RNN state in the middle of a sequence
#             if terminated is not None and torch.any(terminated):
#                 # Se in almeno un env la sequenza contiene un fine episodio (o più), segmenti la sequenza in sottosequenze [i0:i1) 
#                 # ogni volta che c’è un “taglio” (un punto dove qualcuno ha terminated=True nel tempo precedente).
#                 # indexes conterrà i confini di questi segmenti: es. [0, tA, tB, L].
#                 rnn_outputs = []
#                 terminated = terminated.view(-1, self.sequence_length)
#                 indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

#                 # per ogni segmento, esegua il forward della LSTM e resetti lo stato alla fine del segmento
#                 for i in range(len(indexes) - 1):
#                     i0, i1 = indexes[i], indexes[i + 1]
#                     rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
#                     hidden_states[:, (terminated[:,i1-1]), :] = 0
#                     cell_states[:, (terminated[:,i1-1]), :] = 0
#                     rnn_outputs.append(rnn_output)

#                 rnn_states = (hidden_states, cell_states)
#                 rnn_output = torch.cat(rnn_outputs, dim=1)
#             # no need to reset the RNN state in the sequence bacause there are no episode terminations
#             else:
#                 rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
#         # rollout
#         # In eval/rollout l’agente produce un’azione per step. Qui L=1.
#         # Gli stati LSTM si propagano da uno step al successivo al di fuori di compute(...) (SKRL glieli ripassa ad ogni chiamata).
#         else:
#             rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
#             rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

#         # flatten the RNN output
#         rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

#         # Action_space is -1 to 1
#         return 1 * torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}



# class Value(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=True,   # era False
#                  num_envs=200, num_layers=2, hidden_size=128, sequence_length=300):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.num_envs = num_envs
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
#         self.sequence_length = sequence_length

#         self.lstm = nn.LSTM(input_size=self.num_observations,
#                             hidden_size=self.hidden_size,
#                             num_layers=self.num_layers,
#                             batch_first=True)  # batch_first -> (batch, sequence, features)

#         self.net = nn.Sequential(nn.Linear(self.hidden_size, 1))

#     def get_specification(self):
#         # batch size (N) is the number of envs
#         return {"rnn": {"sequence_length": self.sequence_length,
#                         "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
#                                   (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

#     def compute(self, inputs, role):
#         states = inputs["states"]
#         terminated = inputs.get("terminated", None)
#         hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

#         # training
#         if self.training:
#             rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

#             hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
#             cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
#             # get the hidden/cell states corresponding to the initial sequence
#             hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
#             cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

#             # reset the RNN state in the middle of a sequence
#             if terminated is not None and torch.any(terminated):
#                 rnn_outputs = []
#                 terminated = terminated.view(-1, self.sequence_length)
#                 indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

#                 for i in range(len(indexes) - 1):
#                     i0, i1 = indexes[i], indexes[i + 1]
#                     rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
#                     hidden_states[:, (terminated[:,i1-1]), :] = 0
#                     cell_states[:, (terminated[:,i1-1]), :] = 0
#                     rnn_outputs.append(rnn_output)

#                 rnn_states = (hidden_states, cell_states)
#                 rnn_output = torch.cat(rnn_outputs, dim=1)
#             # no need to reset the RNN state in the sequence
#             else:
#                 rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
#         # rollout
#         else:
#             rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
#             rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

#         # flatten the RNN output
#         rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)
#         print("DEBUG[Value compute] rnn_output shape:", tuple(rnn_output.shape))
#         print("DEBUG[Value compute] net output shape:", tuple(self.net(rnn_output).shape))

#         return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}
    

# env = load_isaaclab_env(task_name="Isaac-Velocity-PaperFlat-G1-v0",
#                         num_envs=200,
#                         headless=True)
# env = wrap_env(env)
# device = env.device

# # instantiate a memory as rollout buffer (any memory can be used for this)
# memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device)


# # instantiate the agent's models (function approximators).
# # PPO requires 2 models, visit its documentation for more details
# # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
# models = {}
# models["policy"] = Policy(env.single_observation_space, env.single_action_space, device, clip_actions=True, num_envs=env.num_envs)
# models["value"] = Value(env.single_observation_space, env.single_action_space, device, num_envs=env.num_envs)


# # configure and instantiate the agent (visit its documentation to see all the options)
# # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
# cfg = PPO_DEFAULT_CONFIG.copy()
# cfg["rollouts"] = 9600  # memory_size 300*32
# cfg["learning_epochs"] = 4
# cfg["mini_batches"] = 32
# cfg["discount_factor"] = 0.9 # try eventually 0.99
# cfg["lambda"] = 0.95
# cfg["learning_rate"] = 1e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
# cfg["grad_norm_clip"] = 0.5
# cfg["ratio_clip"] = 0.2
# cfg["value_clip"] = 0.2
# cfg["clip_predicted_values"] = False
# cfg["entropy_loss_scale"] = 0.0
# cfg["value_loss_scale"] = 0.5
# cfg["kl_threshold"] = 0
# cfg["state_preprocessor"] = RunningStandardScaler  #  preprocessore online che SKRL usa per normalizzare gli stati (osservazioni) durante il training
# cfg["state_preprocessor_kwargs"] = {"size": env.single_observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device} 
# # logging to TensorBoard and write checkpoints (in timesteps)
# cfg["experiment"]["write_interval"] = 20000
# cfg["experiment"]["checkpoint_interval"] = 200000
# cfg["experiment"]["directory"] = "runs/torch/G1_flat_paper_ppo"

# agent = PPO(models=models,
#             memory=memory,
#             cfg=cfg,
#             observation_space=env.single_observation_space,
#             action_space=env.single_action_space,
#             device=device)


# # configure and instantiate the RL trainer
# cfg_trainer = {"timesteps": 150000000, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# # start training
# trainer.train()

# env.close()


# source _isaac_sim/setup_conda_env.sh