import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from actor_critic import *
from breakout import *

from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

mode = "regular"  
num_envs = 16 

def make_env():
    def _thunk():
        env = AtariGame(mode, 1000)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape

#a2c hyperparams:
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
num_steps = 5
num_frames = 300#int(10e5)

#rmsprop hyperparams:
lr    = 7e-4
eps   = 1e-5
alpha = 0.99

#Init a2c and rmsprop
actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)
optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    
if USE_CUDA:
	actor_critic = actor_critic.cuda()


rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
rollout.cuda()

all_rewards = []
all_losses  = []


state = envs.reset()
#print("state len:",len(state))
state = torch.FloatTensor(np.float32(state))

rollout.states[0].copy_(state)

episode_rewards = torch.zeros(num_envs, 1)
final_rewards   = torch.zeros(num_envs, 1)


for i_update in range(num_frames):
	for step in range(num_steps):

		action = actor_critic.act(Variable(state))

		next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
		reward = torch.FloatTensor(reward).unsqueeze(1)

		episode_rewards += reward

		masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)

		final_rewards *= masks
		final_rewards += (1-masks) * episode_rewards
		episode_rewards *= masks

		if USE_CUDA:
			masks = masks.cuda()

		state = torch.FloatTensor(np.float32(next_state))
		rollout.insert(step, state, action.data, reward, masks)

	_, next_value = actor_critic(Variable(rollout.states[-1], volatile=True))
	next_value = next_value.data

	returns = rollout.compute_returns(next_value, gamma)

	logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
	    Variable(rollout.states[:-1]).view(-1, *state_shape),
	    Variable(rollout.actions).view(-1, 1)
	)

	values = values.view(num_steps, num_envs, 1)
	action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
	advantages = Variable(returns) - values

	value_loss = advantages.pow(2).mean()
	action_loss = -(Variable(advantages.data) * action_log_probs).mean()

	optimizer.zero_grad()
	loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
	loss.backward()
	nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
	optimizer.step()

	if i_update % 100 == 0:
		print("i_update",i_update)
    #     all_rewards.append(final_rewards.mean())
    #     all_losses.append(loss.data[0])
        
    #     plt.figure(figsize=(20,5))
    #     plt.subplot(131)
    #     plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
    #     plt.plot(all_rewards)
    #     plt.subplot(132)
    #     plt.title('loss %s' % all_losses[-1])
    #     plt.plot(all_losses)
    #     #plt.show()
        
	rollout.after_update()


torch.save(actor_critic.state_dict(), "breakout" + mode)

# import time 

# def displayImage(image, step, reward):
#     clear_output(True)
#     s = "step: " + str(step) + " reward: " + str(reward)
#     plt.figure(figsize=(10,3))
#     plt.title(s)
#     plt.imshow(image)
#     plt.show()
#     time.sleep(0.1)

# env = MiniPacman(mode, 1000)

# done = False
# state = env.reset()
# total_reward = 0
# step   = 1


# while not done:
#     current_state = torch.FloatTensor(state).unsqueeze(0)
#     if USE_CUDA:
#         current_state = current_state.cuda()
        
#     action = actor_critic.act(Variable(current_state))
    
#     next_state, reward, done, _ = env.step(action.data[0, 0])
#     total_reward += reward
#     state = next_state
    
#     image = torch.FloatTensor(state).permute(1, 2, 0).cpu().numpy()
#     displayImage(image, step, total_reward)
#     step += 1