import gym
import imageio
import cv2 as cv
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
    reward_lose=0,
    penalty_mode="constant",
    penalty_step=0,
    penalty_time=-0.01,
    actions=action_space,

)
# Create a DQN agent
agent = Agent(gamma=0.99, epsilon=0.01, batch_size=512, n_actions=env.action_space.n, eps_end=0.01,
              input_dims=[21 * 79 + 55 + 256], lr=1e-3)

scores, eps_history = [], []
n_games = 5
env_name = env.unwrapped.spec.id
model_save_path = "./agents/DQN_" + env_name + ".pt"
results_save_path = "./results/DQN_testing_curve_" + env_name + ".png"

# Load the trained policy network
agent.Q_eval.load_state_dict(torch.load(model_save_path))


# function to flatten and concatenate the environment observation so that they can by based into the Q network
def format_observation(obs):
    state = obs['glyphs'].flatten()
    state = np.concatenate((state, obs["inv_glyphs"], obs["message"]))
    return state


# a function to get an image of each game state
def get_frame(obs):
    return obs["pixel"]


# a function to get the messages and inventory, message, and reward for each game state. This will be used for rendering
def get_frame_title(obs, reward):
    message = bytes(obs['message'])
    title = ""
    message = message[: message.index(b"\0")].decode("utf-8")
    title += "MESSAGE: " + message
    title += "\n \n"
    title += "Inevntory: \n"
    inv = obs["inv_strs"]

    for i in range(len(inv)):
        item = bytes(inv[i])
        string = item[: item.index(b"\0")].decode("utf-8")
        if string == "":
            break
        title += string + "\n"
    title += "REWARD: " + str(reward)
    return title


counter = 0
n_test_episodes = 1
for i in range(n_test_episodes):
    score = 0
    done = False
    observation = env.reset()
    # record frame
    plt.title(get_frame_title(observation, 0))
    plt.tight_layout
    plt.imshow(get_frame(observation))
    plt.savefig("./out_vid/frame_0.png")
    observation = format_observation(observation)
    while not done:
        counter += 1
        action = agent.choose_action(observation, test=True)
        print(action)
        next_observation, reward, done, info = env.step(action)
        # record frame
        plt.title(get_frame_title(next_observation, reward))
        plt.tight_layout()
        plt.imshow(get_frame(next_observation))
        plt.savefig("./out_vid/frame_" + str(counter) + ".png")

        next_observation = format_observation(next_observation)
        score += reward
        observation = next_observation
    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])
    print('Episode number: ', i, ' Episode Score: ', score, " Average Score Per Episode: ", avg_score, " Epsilon:",
          agent.epsilon)
# plot test results
# plot results
title = "Training Curve for DQN on " + env_name
plt.title(title)
plt.ylabel("Total Reward per episode")
plt.xlabel("Number of episodes")
plt.plot(np.arange(n_test_episodes), scores)
plt.savefig(results_save_path)

