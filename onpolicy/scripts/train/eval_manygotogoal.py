#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

import ManyAgent_GoTOGoal  # noqa: F401
import gymnasium as gym
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from onpolicy.runner.separated.mujoco_runner import MujocoRunner as Runner


def normalize_scenario(scenario):
    aliases = {
        "ManyAgentGoToGoalEnv": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoalEnv-v0": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoal": "ManyAgentGoToGoal-v0",
    }
    return aliases.get(scenario, scenario)


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                scenario = normalize_scenario(all_args.scenario)
                env = gym.make(scenario, disable_env_checker=True)
            else:
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario", type=str, default="ManyAgentGoToGoal-v0")
    parser.add_argument("--num_agents", type=str, default="5")
    parser.add_argument("--agent_obsk", type=int, default=0)
    parser.add_argument("--faulty_node", type=int, default=-1)
    parser.add_argument("--eval_faulty_node", type=int, nargs="+", default=None)
    parser.add_argument("--eval_loops", type=int, default=100)
    parser.add_argument("--add_center_xy", action="store_true", default=False)
    parser.add_argument("--use_state_agent", action="store_true", default=False)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    os.makedirs(str(run_dir), exist_ok=True)

    run = None
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project="IROS_Eval_" + all_args.scenario,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.env_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    eval_envs = make_eval_env(all_args)
    num_agents = eval_envs.n_agents

    config = {
        "all_args": all_args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    faulty_nodes = all_args.eval_faulty_node if all_args.eval_faulty_node is not None else [all_args.faulty_node]
    for i in range(all_args.eval_loops):
        for node in faulty_nodes:
            runner.eval(i, int(node))

    eval_envs.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
