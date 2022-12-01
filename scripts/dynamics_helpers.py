import numpy as np
import pybullet as p

def quad_step(state, control, m_scale, l_scale, drone_style='x', return_info = False):
    """
    We refer (repeatedly) to gym_pybullet_drones.envs.BaseAviary.py > class BaseAviary > _dynamics(). 
    This can be found at https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py#L790 .
    Also,
        States: (x, y, z, q0, q1, q2, q3, r, p, y, vx, vy, vz, wr, wp, wy, rpm1, rpm2, rpm3, rpm4) ~ 20
        Control: (tgt_x, tgt_y, tgt_z, tgt_r, tgt_p, tgt_y, tgt_rpm1, tgt_rpm2, tgt_rpm3, tgt_rpm4) ~ 10
    """

    TIMESTEP = 1/240
    M = m_scale * 0.027 # nominal_mass
    L = l_scale * 0.0397 # nominal_arm_length (arm)
    kT = 2.25
    kM = 7.94e-12
    kF = 3.16e-10
    if drone_style == 'x':
        IXX = m_scale * l_scale**2 * 1.4e-5
        IYY = m_scale * l_scale**2 * 1.4e-5
        IZZ = m_scale * l_scale**2 * 2.17e-5
    elif drone_style == '+':
        IXX = m_scale * l_scale**2 * 2.3951e-5
        IYY = m_scale * l_scale**2 * 2.3951e-5
        IZZ = m_scale * l_scale**2 * 3.2347e-5
    J = np.diag([IXX, IYY, IZZ]).reshape(3, 3)    
    J_INV = np.linalg.inv(J)

    pos = state[:3].reshape(3, 1)
    quat = state[3:7]
    rpy = state[7:10].reshape(3, 1)
    vel = state[10:13].reshape(3, 1)
    rpy_rates = state[13:16].reshape(3, 1)
    current_rpm = state[16:20]
    rpm = control[-4:] * (10000) # multiplying by 10k because I divided rpms by 10k in dataset.py

    # Equation (2) in https://arxiv.org/pdf/2103.02142.pdf
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    forces = np.array(rpm**2) * kF
    thrust = np.array([0, 0, np.sum(forces)]).reshape(3, 1)
    g = 9.81
    gravity = np.array([0, 0, g]).reshape(3, 1)
    acceleration = 1/(M) * (R @ thrust) - gravity

    # Equation (3) in https://arxiv.org/pdf/2103.02142.pdf
    z_torques = np.array(rpm**2) * kM
    z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
    if drone_style == 'x':
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (L/np.sqrt(2))
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (L/np.sqrt(2))
    elif drone_style == '+':
        x_torque = (forces[1] - forces[3]) * L
        y_torque = (-forces[0] + forces[2]) * L
    
    torques = np.array([x_torque, y_torque, z_torque]).reshape(3, 1) - np.cross(rpy_rates.T, (J @ rpy_rates).T).reshape(3, 1)
    rpy_rates_deriv = J_INV @ torques

    # Update
    old_vel = vel.copy()
    vel = vel + TIMESTEP * acceleration
    rpy_rates = rpy_rates + TIMESTEP * rpy_rates_deriv
    pos = pos + TIMESTEP * vel
    rpy = rpy + TIMESTEP * rpy_rates

    next_state = np.concatenate((pos, rpy, vel, rpy_rates), axis=0)
    if return_info:
        info = f'------------------------ {vel[2]} = {old_vel[2]} + {TIMESTEP} * {acceleration[2]}'
        return next_state.reshape((12,)), info

    return next_state.reshape((12,))

def quad_dynamics(m, l):
    return lambda tup: quad_step(tup[0], tup[1], m, l)

def bicycle_step(state, control, L, dt, x_normalizer, u_normalizer):
    state = state * x_normalizer
    control = control * u_normalizer

    x, y, theta, delta = state
    v, delta_rate = control
    next_state = state + np.array([v * np.cos(theta), v * np.sin(theta), v*np.tan(delta) / L, delta_rate])*dt

    next_state = next_state / x_normalizer

    return next_state

def bicycle_dynamics(L, dt, x_normalizer, u_normalizer):
    return lambda tup: bicycle_step(tup[0], tup[1], L=L, dt=dt, x_normalizer=x_normalizer, u_normalizer=u_normalizer)

def unicycle_step(state, control, L, dt, x_normalizer, u_normalizer):
    state = state * x_normalizer
    control = control * u_normalizer

    x, y, theta = state
    v, theta_rate = control
    next_state = state + np.array([v * np.cos(theta), v * np.sin(theta), theta_rate])*dt

    if next_state[2] > np.pi:
        next_state[2] = next_state[2]%(2*np.pi)
        next_state[2] = next_state[2] - (2*np.pi)
    if next_state[2] < -np.pi:
        next_state[2] = next_state[2]%(-2*np.pi)
        next_state[2] = next_state[2] + (2*np.pi)

    next_state = next_state / x_normalizer
    return next_state

def unicycle_dynamics(L, dt, x_normalizer, u_normalizer):
    return lambda tup: unicycle_step(tup[0], tup[1], L=L, dt=dt, x_normalizer=x_normalizer, u_normalizer=u_normalizer)

import torch
def armax_step(state, control, model, glucose_normalizer, insulin_normalizer, meal_normalizer):
    T = 10
    glucose_input = state[:T] * glucose_normalizer
    insulin_input = state[T:2*T] * insulin_normalizer
    assert insulin_input[-1] == control[0] * insulin_normalizer 
    meal_input = state[2*T:3*T] * meal_normalizer

    glucose_input = torch.from_numpy(glucose_input).float()
    insulin_input = torch.from_numpy(insulin_input).float()
    meal_input = torch.from_numpy(meal_input).float() 
    prediction = model.forward(glucose_input, insulin_input, meal_input)
    
    prediction = prediction.detach().numpy()[0] / glucose_normalizer
    return [prediction]

def armax_constraint(model, glucose_normalizer, insulin_normalizer, meal_normalizer):
    return lambda tup: armax_step(tup[0], tup[1], model, glucose_normalizer, insulin_normalizer, meal_normalizer)


def spin_model_step(state, control, model, x_normalizer, next_x_normalizer, u_normalizer):
    T = 10

    inputs = torch.from_numpy(np.concatenate((state, control))).float()
    prediction = model(inputs)
    
    prediction = prediction.detach().numpy()
    return prediction

def spin_model_constraint(model, x_normalizer, next_x_normalizer, u_normalizer):
    return lambda tup: spin_model_step(tup[0], tup[1], model, x_normalizer, next_x_normalizer, u_normalizer)


