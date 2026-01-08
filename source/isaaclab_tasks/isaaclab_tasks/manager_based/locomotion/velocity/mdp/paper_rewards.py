"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


### questi import potrebbero non servire tutti, da controllare dopo aver finito di scrivere le funzioni ###

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation,RigidObject
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def heading_quat_from_vel_t(vx, vy, eps=1e-4):
#     speed = torch.sqrt(vx*vx + vy*vy)
#     # mask per deadzone
#     mask = (speed >= eps)
#     yaw = torch.atan2(vy, vx)
#     half = 0.5 * yaw
#     q_des = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)
#     # dove speed<eps, metti None/skip: qui mettiamo NaN e gestiamo a valle
#     q_des = torch.where(mask.unsqueeze(-1), q_des, torch.tensor(float('nan'), device=q_des.device, dtype=q_des.dtype))
#     return q_des, mask


def quat_normalize(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)

def yaw_to_quat(yaw):
    half = 0.5 * yaw
    c = torch.cos(half)
    s = torch.sin(half)
    return torch.stack([c, torch.zeros_like(c), torch.zeros_like(c), s], dim=-1)



# normaliztion function to the exp kernel
def normalize(type_of_quantity, quantity):
    typ = type_of_quantity
    q = quantity

    if typ == "f":                      # pass the force already as the norm as quantity
        qp = torch.clamp(q, min=0)
        return 1.0 - torch.exp(-(qp * qp) / 100.0)

    elif typ == "s":                    # pass the speed already as the norm as quantity
        qp = torch.clamp(q, min=0)
        return 1.0 - torch.exp(-2.0 * (qp * qp))

    elif typ in ("dot_x", "dot_y"):     # pass the des_value-actual_value as quantity
        return 1.0 - torch.exp(-2.0 * torch.abs(q))

    elif typ == "o":                    # pass the quat_T * quat as quantity  (1-() è l'errore tra le due orientazioni, quat identici prodotto 1, opposti, -1)
        err = 1.0 - q
        return 1.0 - torch.exp(-3.0 * (err * err))

    elif typ == "ad":                   # pass the norm of the differece between actual and previous action as quantity
        qp = torch.clamp(q, min=0)
        return 1.0 - torch.exp(-5.0 * qp)

    elif typ == "t":                    # pass the norm of the net torque applied to all joints as quantity
        qp = torch.clamp(q, min=0)
        return 1.0 - torch.exp(-0.05 * qp)

    elif typ == "pa":                   # pass the sum of pelvis rot and acc (both normed) as quantity
        qp = torch.clamp(q, min=0)
        return 1.0 - torch.exp(-0.10 * qp)

    else:                               # default: tensore pieno di -1 con stessa shape/device/dtype di q
        return torch.full_like(q, -1)



# foot reward
def foot_reward(env: ManagerBasedRLEnv, 
                asset_cfg: SceneEntityCfg,
                sensor_cfg: SceneEntityCfg,
                foot: str) -> torch.Tensor:
    
    # phi = (phi + offset) % 1.0                  # col modulo faccio in pratica un wraparound per rimanere nell'intervallo [0,1]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]        # [num_envs, history_len, num_selected_bodies, 3]. these are normal contact forces in world frame
    asset = env.scene[asset_cfg.name]
    lin_body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].norm(dim=-1)  # [num_envs, num_selected_bodies, 3]

    normalized_force = normalize("f", net_contact_forces)
    normalized_speed = normalize("s", lin_body_vel)    

    # swing = indicator_von_mises(phi, swing_start, swing_end)
    # stance = indicator_von_mises(phi, stance_start, stance_end)
    if foot == "right_foot":
        # swing = env.Von_Mises_Values_right[float(env.phi_right) ][0]    # get swing value from precomputed dict
        # stance = env.Von_Mises_Values_right[float(env.phi_right) ][1]       ### controlla che acceda correttamete (vedi se serve round per riconoscere la chiave) ###
        idx = env.idx_right
        table = env.VM_right

    else:   # left_foot
        # swing = env.Von_Mises_Values_left[float(env.phi_left) ][0]    # get swing value from precomputed dict
        # stance = env.Von_Mises_Values_left[float(env.phi_left) ][1]       ### controlla che acceda correttamete (vedi se serve round per riconoscere la chiave) ###
        idx = env.idx_left
        table = env.VM_left
    
    # Estrai swing e stance per ogni env -> (num_envs,)
    # colonna 0 = swing, colonna 1 = stance
    swing  = table[idx, 0]
    stance = table[idx, 1]

    return ((0 * stance * normalized_force) 
          + ((-1) * swing * normalized_force) 
          + ((-1) * stance * normalized_speed)
          + (0 * swing * normalized_speed))

# bipedal reward
def bipedal_reward(env: ManagerBasedRLEnv, 
                   # ratio: float, # L: float,  # L = number of discrete timesteps in the period (period(s) = L*env.step_dt)       # period: float, 
                   # right_offset: float, left_offset: float, 
                   right_foot_sensor_cfg: SceneEntityCfg, 
                   left_foot_sensor_cfg: SceneEntityCfg,
                   left_foot_cfg: SceneEntityCfg,
                   right_foot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward function for bipedal locomotion tasks."""
    # current_time_s = env.episode_length_buf.unsqueeze(1) * env.step_dt  ### prova anche time.time, o env.sim_time ###
    # phi = (current_time_s % period) / period  # phase in [0, 1]

    # phi = (env.episode_length_buf % env.cfg.L) / env.cfg.L  # è come versione col tempo in secondi ma ragionando in step di simulazione, invece che il periodo in sec ho L he è il numero di tep che formano 1 periodo
    ### fai print di phi per vedere che va da 0 a 1 ogni L step ###
    
    # swing_start, swing_end = 0.0, ratio
    # stance_start, stance_end = ratio, 1.0

    reward_left = foot_reward(env, left_foot_cfg, left_foot_sensor_cfg, "left_foot")
    reward_right = foot_reward(env, right_foot_cfg, right_foot_sensor_cfg, "right_foot")

    return reward_left + reward_right
    

# cmd reward
def cmd_reward(env: ManagerBasedRLEnv, command_name: str,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward function for tracking commanded quantities."""  # pelvis lin x and y vel and pelvis orientation (quaternion)
    # extract the used quantities 
    asset = env.scene[asset_cfg.name]
    asset_ob: RigidObject = env.scene[asset_cfg.name]

    vx = env.command_manager.get_command(command_name)[:, 0]
    vy = env.command_manager.get_command(command_name)[:, 1]

    lin_root_vel_x_error = vx - asset.data.root_lin_vel_b[:, 0]              # in local frame
    lin_root_vel_y_error = vy - asset.data.root_lin_vel_b[:, 1]

    # --- Orientazione (quaternioni [w, x, y, z]) ---
    q_actual = asset.data.root_quat_w                               
    q_actual = torch.nn.functional.normalize(q_actual, p=2, dim=-1)
    # q_des la derivo da i comandi di vel x e y nel frame del robot e poi la devo trasformare in quaternione relativo al world frame
    #identita
    # q_des = torch.tensor([1.0, 0.0, 0.0, 0.0],
    #                      device=q_actual.device, dtype=q_actual.dtype)
    #oppure calcolata da comandi vel x e y
    # yaw_quat funz gia implementata che estrae la componente di yaw da un quaternione
    

    heading = asset_ob.data.heading_w  # Yaw heading of the base frame (in radians). Shape is (num_instances,)  (estraggo yaw del robot nel world frame)
    
    phi_body = torch.atan2(vy, vx)  # direzione del comando nel frame corpo
    speed2 = vx*vx + vy*vy

    # world yaw della risultante
    eps = 1e-8
    phi_world = torch.where(speed2 >= eps, heading + phi_body, heading) # dove la vel è maggiore di eps uso heading + phi_body, altrimenti heading (per evitare NaN quando la velocità è troppo bassa)

    q_des = yaw_to_quat(phi_world)
    # q_des = quat_normalize(q)

    mask = (speed2 >= eps)
    
    # # per evitare NaN, metti identità dove la velocità è troppo bassa  ### ha sneso? ###
    # q_des = torch.where(mask.unsqueeze(-1), q_des, torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_des.device, dtype=q_des.dtype))
    
    # q_des = q_des.expand_as(q_actual)
    dot = (q_actual * q_des).sum(dim=-1)
    # dot = torch.clamp(dot, -1.0, 1.0)

    q_dot_x = normalize("dot_x", lin_root_vel_x_error)
    q_dot_y = normalize("dot_y", lin_root_vel_y_error)
    q_orientation = normalize("o", dot)
    q_orientation = q_orientation * mask.float()   # se la velocità è troppo bassa, non considerare il reward di orientamento

    return -(q_dot_x + q_dot_y + q_orientation)

# smooth reward
def smooth_reward(env: ManagerBasedRLEnv, 
                  asset_cfg: SceneEntityCfg,
                  asset_root_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward function for smooth actions."""

    asset: Articulation = env.scene[asset_cfg.name]
    asset_root: Articulation = env.scene[asset_root_cfg.name]

    action_diff = env.action_manager.action - env.action_manager.prev_action
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    pelvis_acc = asset_root.data.body_ang_vel_w.norm(dim=-1) + asset_root.data.body_lin_acc_w.norm(dim=-1)  #POTRESTI AVRE PROBLEMA DI DIMENSIONI PERCHE HA DIM [ENV, 1, 1] E POTRESTI VOLERE SOLO [ENV, 1]

    q_action_diff = normalize("ad", action_diff.norm(dim=1))
    q_torques = normalize("t", torques.norm(dim=1))
    q_pelvis_acc = normalize("pa", pelvis_acc)

    return -(q_action_diff + q_torques + q_pelvis_acc)

def bias(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant bias reward term to encourage movement."""
    return torch.ones_like(env.episode_length_buf, dtype=torch.float32)


# import torch
# import torch.nn.functional as F

# def cmd_reward(env: ManagerBasedRLEnv, command_name: str,
#                asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Reward function for tracking commanded quantities."""
#     asset = env.scene[asset_cfg.name]

#     # Error sulle velocità lineari (x, y) del root
#     lin_root_vel_x_error = env.command_manager.get_command(command_name)[:, 0] - asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0]
#     lin_root_vel_y_error = env.command_manager.get_command(command_name)[:, 1] - asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 1]

#     # --- Orientazione (quaternioni [w, x, y, z]) ---
#     q_actual = asset.data.root_quat_w  # shape tipica: [N, 4] (oppure [N, B, 4])

#     # Normalizza per sicurezza (evita che piccoli drift rompano il dot)
#     q_actual = F.normalize(q_actual, p=2, dim=-1)

#     # Quaternione desiderato: identità [1, 0, 0, 0]
#     q_des = torch.tensor([1.0, 0.0, 0.0, 0.0],
#                          device=q_actual.device, dtype=q_actual.dtype)
#     # Espandi per allineare la shape di q_actual
#     q_des = q_des.expand_as(q_actual)

#     # Dot product (q_actual^T q_des) lungo l'ultima dimensione
#     dot = (q_actual * q_des).sum(dim=-1)
#     # (opzionale) clamp per stabilità numerica
#     dot = torch.clamp(dot, -1.0, 1.0)

#     # Formula: (1 - dot)^2
#     quat_error = (1.0 - dot) ** 2

#     # Se normalize() si aspetta un tensore errore per env (no dimensione corpo),
#     # riduci eventuale dimensione extra con media o somma:
#     # Esempio: se q_actual è [N, B, 4], allora quat_error è [N, B].
#     # Scegli la riduzione coerente con il tuo design (media o somma):
#     if quat_error.dim() == 2:  # [N, B]
#         quat_error = quat_error.mean(dim=1)  # oppure .sum(dim=1)

#     # Normalizza gli errori coi tuoi pesi/scale
#     q_dot_x = normalize("dot_x", lin_root_vel_x_error)
#     q_dot_y = normalize("dot_y", lin_root_vel_y_error)
#     q_orientation = normalize("o", quat_error)

#     return -(q_dot_x + q_dot_y + q_orientation)



# import torch
# import math
# from typing import Union

# def _phi_standard_normal(z: torch.Tensor) -> torch.Tensor:
#     """CDF normale standard Φ(z) via erf, tensor-wise."""
#     return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

# def wrapped_normal_cdf(
#     theta: torch.Tensor,  # [*] angoli in radianti (tensor)
#     mu: Union[float, torch.Tensor],  # media in radianti
#     sigma: float,                    # deviazione std della gaussiana 'non avvolta'
#     K: int = 2,                      # numero di termini di wrap (±K)
# ) -> torch.Tensor:
#     """
#     CDF approssimata della Wrapped Normal: somma di Φ((theta - mu + 2πk)/σ) per k in [-K, K].
#     Funziona per tensor di qualsiasi shape; broadcasting su mu.
#     """
#     # Assicura tipi/shape coerenti
#     theta = torch.as_tensor(theta)
#     mu = torch.as_tensor(mu, device=theta.device, dtype=theta.dtype)

#     # Prepara il range di k come tensor per broadcast
#     ks = torch.arange(-K, K + 1, device=theta.device, dtype=theta.dtype)  # [2K+1]
#     # (broadcast) -> shape: [*, 2K+1]
#     z = (theta.unsqueeze(-1) - mu.unsqueeze(-1) + 2.0 * math.pi * ks) / sigma
#     return _phi_standard_normal(z).sum(dim=-1)

# def phase_indicator_von_mises_like(
#     phi: torch.Tensor,   # [num_envs] fase normalizzata in [0,1]
#     start: float,        # μ_A in [0,1] (inizio fase)
#     end: float,          # μ_B in [0,1] (fine fase)
#     kappa: float,        # "concentrazione" tipo Von Mises
#     K: int = 2,          # termini di wrap nella somma (2-3 di solito bastano)
#     sigma_rule: str = "1/sqrt(kappa)",  # regola per σ dalla κ
# ) -> torch.Tensor:
#     """
#     Approssima E[I(φ)] = P(A < φ < B) con A,B ~ Wrapped Normal
#     (μ_A=2π*start, μ_B=2π*end, σ derivato da κ).

#     Ritorna un tensore ∈ [0,1] con stessa shape di `phi`.
#     """
#     device = phi.device
#     dtype = phi.dtype

#     # Converti fase in radianti
#     theta = (2.0 * math.pi) * phi.to(dtype=dtype, device=device)
#     mu_A = (2.0 * math.pi) * float(start)
#     mu_B = (2.0 * math.pi) * float(end)

#     # Mappa κ -> σ. Scelte comuni:
#     if kappa <= 0:
#         # κ→0: distribuzione "quasi uniforme". Usa σ grande per rendere transizioni morbidissime
#         sigma = torch.tensor(math.pi, device=device, dtype=dtype)  # molto diffusa
#     else:
#         if sigma_rule == "1/sqrt(kappa)":
#             sigma = torch.tensor(1.0 / math.sqrt(kappa), device=device, dtype=dtype)
#         elif sigma_rule == "best-approx":
#             # approssimazione basata su varianza circolare ~ 1 - I1(κ)/I0(κ).
#             # σ^2 ≈ -2 ln(R), con R ≈ I1(κ)/I0(κ); qui usiamo una proxy semplice e stabile.
#             # NOTA: richiederebbe i0/i1; se vuoi una versione con torch.special posso aggiungerla.
#             R = kappa / (kappa + 1.0)  # proxy liscia in (0,1)
#             sigma = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.tensor(R, device=device, dtype=dtype)), min=1e-6))
#         else:
#             raise ValueError("sigma_rule non riconosciuta")

#     # Probabilità cumulate
#     P_A_lt_phi = wrapped_normal_cdf(theta, mu_A, sigma, K=K)   # P(A < φ)
#     P_B_lt_phi = wrapped_normal_cdf(theta, mu_B, sigma, K=K)   # P(B < φ)

#     # E[I(φ)] = P(A<φ) * (1 - P(B<φ))
#     indicator = P_A_lt_phi * (1.0 - P_B_lt_phi)
#     return torch.clamp(indicator, 0.0, 1.0)



# import torch
# from typing import Union

# def normalize(
#     type_of_quantity: str,
#     quantity: Union[torch.Tensor, float, int],
#     *,
#     device: torch.device | str | None = None,
#     dtype: torch.dtype | None = None,
# ) -> torch.Tensor:
#     """
#     Normalizza diverse quantità fisiche usando kernel esponenziali in stile
#     '1 - exp(-alpha * err^2)' (o varianti), in modo vettoriale (PyTorch).

#     Parametri
#     ---------
#     type_of_quantity : {"f", "s", "dot_x", "dot_y", "o", "ad", "t", "pa"}
#         - "f":    forza (passa già la norma della forza)
#         - "s":    velocità (passa già la norma della velocità)
#         - "dot_x": errore di velocità lungo x (passa des - actual)
#         - "dot_y": errore di velocità lungo y (passa des - actual)
#         - "o":    orientazione (passa quat_T * quat -> in [-1, 1])
#         - "ad":   action difference (passa ||a_t - a_{t-1}||)
#         - "t":    torque (passa ||tau||)
#         - "pa":   pelvis acc (passa ||rot|| + ||acc||)
#     quantity : Tensor o scalare
#         Valore/i da normalizzare. Può essere un tensore su CPU o GPU oppure
#         uno scalare; in quest’ultimo caso verrà convertito a tensore.
#     device, dtype : opzionali
#         Forzano device/dtype se `quantity` non è già un tensore.

#     Ritorna
#     -------
#     torch.Tensor
#         Tensore normalizzato in [0, 1) (tende a 1 asintoticamente).

#     Note
#     ----
#     - Per quantità che rappresentano norme o grandezze intrinsecamente non negative
#       ("f", "s", "ad", "t", "pa") effettuo un clamp a >= 0.
#     - Per "o" assumo quantity in [-1, 1]; eseguo clamp per sicurezza.
#     - L'implementazione è interamente broadcast-friendly.
#     """
#     # Converte a tensore se necessario mantenendo device/dtype sensati
#     if isinstance(quantity, torch.Tensor):
#         q = quantity
#         if device is not None and q.device != torch.device(device):
#             q = q.to(device)
#         if dtype is not None and q.dtype != dtype:
#             q = q.to(dtype)
#     else:
#         q = torch.as_tensor(quantity, device=device, dtype=dtype)

#     typ = type_of_quantity

#     if typ == "f":  # forza (norma)
#         qp = torch.clamp(q, min=0)
#         return 1.0 - torch.exp(-(qp * qp) / 100.0)

#     elif typ == "s":  # velocità (norma)
#         qp = torch.clamp(q, min=0)
#         return 1.0 - torch.exp(-2.0 * (qp * qp))

#     elif typ in ("dot_x", "dot_y"):  # errore velocità lungo x/y (des - actual)
#         return 1.0 - torch.exp(-2.0 * torch.abs(q))

#     elif typ == "o":  # orientazione: passa (quat_T * quat) ~ cos(theta) in [-1, 1]
#         qc = torch.clamp(q, min=-1.0, max=1.0)
#         err = 1.0 - qc
#         return 1.0 - torch.exp(-3.0 * (err * err))

#     elif typ == "ad":  # action difference: norma >= 0
#         qp = torch.clamp(q, min=0)
#         return 1.0 - torch.exp(-5.0 * qp)

#     elif typ == "t":  # torque: norma >= 0
#         qp = torch.clamp(q, min=0)
#         return 1.0 - torch.exp(-0.05 * qp)

#     elif typ == "pa":  # pelvis acc: somma di norme >= 0
#         qp = torch.clamp(q, min=0)
#         return 1.0 - torch.exp(-0.10 * qp)

#     else:
#         # default: tensore pieno di -1 con stessa shape/device/dtype di q
#         return torch.full_like(q, -1)
       
