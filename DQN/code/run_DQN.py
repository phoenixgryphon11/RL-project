import gym
import minihack
from DQN import Agent
import numpy as np
import torch
import matplotlib.pyplot as plt
from minihack import RewardManager
from nle import nethack

# Create a list of all the possible actions needed for the Quest-Hard environment.
moves = tuple(nethack.CompassDirection)
abilities = (
    # open the closed doors
    nethack.Command.OPEN,
    # pick-up item since autopickup is disabled
    nethack.Command.PICKUP,
    # actions for items to deal with lava
    # drink potions
    nethack.Command.QUAFF,
    # put on ring
    nethack.Command.PUTON,
    # wear boots
    nethack.Command.WEAR,
    # apply horn
    nethack.Command.APPLY,

    # zap wand
    nethack.Command.ZAP,
    # press f
    nethack.Command.FIRE,
    # Press g
    nethack.Command.RUSH,
    # press r
    nethack.Command.READ,
    # press y
    ord("y")
)
action_space = moves + abilities
# Custom rewards manager for rewards shaping
reward_manager = RewardManager()

# pick up reward
pick_up_msgs_0 = ["f - a"]
pick_up_msgs_1 = ["g - a"]
reward_manager.add_message_event(pick_up_msgs_0, reward=0.7, repeatable=False)
reward_manager.add_message_event(pick_up_msgs_1, reward=0.7, repeatable=False)

# levitation reward
levitate_msgs = ["You start to float in the air!", "a ring of levitation (on right hand)."]
reward_manager.add_message_event(levitate_msgs, reward=0.8, repeatable=False)

# freezing lava reward
freeze_msgs = ["The lava cools and solidifies."]
reward_manager.add_message_event(freeze_msgs, reward=0.8, repeatable=True)

# kill minotaur reward
kill_msgs = ["You kill the minotaur!"]
reward_manager.add_message_event(kill_msgs, reward=0.9, repeatable=False)
# Maze solved reward
door_msg = ["The door opens."]
reward_manager.add_message_event(door_msg, reward=0.6, terminal_sufficient=False, repeatable=False)

# Penalty for walking into walls
msgs = ["It's a wall."]
reward_manager.add_message_event(msgs, reward=-0.01, terminal_sufficient=False, repeatable=True)

env = gym.make(
    "MiniHack-Quest-Hard-v0",
    observation_keys=(["glyphs", "inv_glyphs", "pixel", "message"]),
    reward_manager=reward_manager,
    reward_win=1,
    reward_lose =0,
    penalty_mode="constant",
    penalty_step= 0,
    penalty_time = -0.01,
    actions=action_space,


)
# Create a DQN agent
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=512, n_actions=env.action_space.n, eps_end=0.01,
              input_dims=[21 * 79 + 55 + 256], lr=1e-3)
# variables for the scores, results and models
scores, avg_scores, eps_history = [], [], []
n_episodes = 1000
env_name = env.unwrapped.spec.id
model_save_path = "./agents/DQN_" + env_name + ".pt"
results_save_path = "./results/DQN_training_curve_" + env_name + ".png"
# if You wish to load a pretrained model set this to true
load_model = False


if load_model:
    agent.Q_eval.load_state_dict(torch.load(model_save_path))
    agent.Q_target.load_state_dict(torch.load(model_save_path))


# function to flatten and concatenate the environment observation so that they can by based into the Q network
def format_observation(obs):
    state = obs['glyphs'].flatten()
    state = np.concatenate((state, obs["inv_glyphs"], obs["message"]))
    return state

# number of timesteps between updating the target network to equal the policy network
target_net_update_freq = 1000
# number of episodes between successive saves of the Q network
save_frequency = 50
for i in range(n_episodes):
    # Score stores the total rewards for each episode
    score = 0
    done = False
    observation = env.reset()
    observation = format_observation(observation)
    timestep = 0
    # loop for each step of an episode
    while not done:
        # choose an action using the DQN agent
        action = agent.choose_action(observation)
        # step the enviroment
        next_observation, reward, done, info = env.step(action)
        next_observation = format_observation(next_observation)
        score += reward
        # store the transition in the replay buffer
        agent.store_transition(observation, action, reward, next_observation, done)
        # update the parameters of the DQN agents policy network
        agent.learn()
        if timestep % target_net_update_freq == 0:
            # update the DQN agents target network
            agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
        observation = next_observation
        timestep += 1

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-50:])
    avg_scores.append(avg_score)
    print('Episode number: ', i, ' Episode Score: ', score, " Average Score Per Episode: ", avg_score, " Epsilon:", agent.epsilon)
    # save the Q network
    if i % save_frequency == 0:
        torch.save(agent.Q_eval.state_dict(), model_save_path)
# Save the final Q network
torch.save(agent.Q_eval.state_dict(), model_save_path)
# plot results
np.savez("DQN_results_np", np.array(scores), np.array(avg_scores))
title = "Training Curve for DQN on " + env_name
plt.title(title)
plt.ylabel("Average Total Reward per episode")
plt.xlabel("Number of episodes")
plt.plot(np.arange(n_episodes), avg_scores)
plt.savefig(results_save_path)

