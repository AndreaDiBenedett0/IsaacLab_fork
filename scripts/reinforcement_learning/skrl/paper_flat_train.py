import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as BasePPO_RNN
from skrl.agents.torch import Agent  # per chiamare direttamente il base class in record_transition
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# ========= WRAPPER DELL'AGENTE: PPO_RNN con record_transition "safe" =========
class PPO(BasePPO_RNN):
    """Sottoclasse di PPO_RNN che forza values a (N, 1) e NON usa inverse=True
       sul value preprocessor in fase di recording (per evitare broadcast mismatch)."""

    @torch.no_grad()
    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos,
        timestep: int,
        timesteps: int,
    ) -> None:
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
            rnn_states.update(
                {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
            )
            if self.policy is not self.value:
                rnn_states.update(
                    {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                )

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
        states = inputs["states"]                                           # tensore di osservazioni
        terminated = inputs.get("terminated", None)                         # mask di terminazioni (opzionale)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]     # stato LSTM corrente

        # training
        if self.training:
            # reshape inputs for RNN
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence (time step 0)
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout (eval)
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)

        # Action_space is -1 to 1
        return 1 * torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


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
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)
        # print("DEBUG[Value compute] rnn_output shape:", tuple(rnn_output.shape))
        # print("DEBUG[Value compute] net output shape:", tuple(self.net(rnn_output).shape))

        return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}


# ========== ENV & TRAINING ==========
env = load_isaaclab_env(task_name="Isaac-Velocity-PaperFlat-G1-v0",
                        num_envs=7,
                        headless=True)
env = wrap_env(env)
device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators)
models = {}
models["policy"] = Policy(env.single_observation_space, env.single_action_space, device, clip_actions=True, num_envs=env.num_envs)
models["value"]  = Value(env.single_observation_space, env.single_action_space, device, num_envs=env.num_envs)

# configure and instantiate the agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 9600
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
cfg["experiment"]["write_interval"] = 20000
cfg["experiment"]["checkpoint_interval"] = 200000
cfg["experiment"]["directory"] = "runs/torch/G1_flat_paper_ppo"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 150000000, "headless": True}
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