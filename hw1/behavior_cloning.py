#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

'''
Hopper-v2의 경우로 주석처리, action = 3(thigh_joint, leg_joint, foot_joint)
'''
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        # Stack expert's observation and action, check the expert's reward, demonstratioon when render option is true
        # Not using the demonstartion data, but using the experte's weight from saved file
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:]) # action = (1,3)
                observations.append(obs) # obs = (11,)
                actions.append(action) 
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 1000 == 0: 
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        #    pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        ''' ------------------- The above sources are provided from Berkeley  ------------------- '''

        print('Extracted observation shape', expert_data['observations'].shape)
        print('Extracted action shape', expert_data['actions'].shape)

        # Start the training through behavior cloning
        our_model = model.Model(expert_data['observations'], expert_data['actions'], args.envname[:-3], "behavior_cloning")
        
        print('Model training start..!')
        our_model.train() 

        for i in range(5):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = our_model.sample(obs) # action = (3,)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                if args.render:
                    env.render()
                if steps % 1000 == 0: 
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    print('end')
                    break

            print('Behavior cloning reward: ', totalr)

if __name__ == '__main__':
    main()
