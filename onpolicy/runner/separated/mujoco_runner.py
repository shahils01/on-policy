import time
import csv
import os
import numpy as np
import torch
import wandb

from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """Runner for continuous-control multi-agent envs with separated policies."""

    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)
        self._agg_reached_counts_by_fault = {}
        self._agg_collision_counts_by_fault = {}

    def _flatten_info_dicts(self, obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, np.ndarray):
            out = []
            for x in obj.flat:
                out.extend(self._flatten_info_dicts(x))
            return out
        if isinstance(obj, (list, tuple)):
            out = []
            for x in obj:
                out.extend(self._flatten_info_dicts(x))
            return out
        return []

    def _update_episode_team_flags(self, eval_infos, reached_flags, collision_flags):
        for env_i, env_info in enumerate(eval_infos):
            agent_infos = self._flatten_info_dicts(env_info)
            max_agents = min(self.num_agents, len(agent_infos))
            for agent_i in range(max_agents):
                info_i = agent_infos[agent_i]
                if not isinstance(info_i, dict):
                    continue
                if "reached_goal" in info_i:
                    reached_flags[env_i, agent_i] = bool(info_i["reached_goal"])
                if "collision" in info_i:
                    collision_flags[env_i, agent_i] = collision_flags[env_i, agent_i] or bool(info_i["collision"])

    def _append_eval_team_stats(self, rows):
        if not rows:
            return

        if self.use_wandb:
            out_path = os.path.join(str(self.run_dir), "eval_team_stats.csv")
        else:
            out_path = os.path.join(str(self.run_dir), "eval_team_stats.csv")

        file_exists = os.path.exists(out_path)
        fieldnames = list(rows[0].keys())

        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _update_aggregate_eval_stats(self, faulty_node, reached_counts, collision_counts):
        if faulty_node not in self._agg_reached_counts_by_fault:
            self._agg_reached_counts_by_fault[faulty_node] = []
        if faulty_node not in self._agg_collision_counts_by_fault:
            self._agg_collision_counts_by_fault[faulty_node] = []
        self._agg_reached_counts_by_fault[faulty_node].extend(list(reached_counts))
        self._agg_collision_counts_by_fault[faulty_node].extend(list(collision_counts))

    def _log_ci_bar_chart_to_wandb(self, reached_counts, collision_counts, faulty_node, step):
        if len(reached_counts) == 0 or len(collision_counts) == 0:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        def _mean_ci95(x):
            x = np.asarray(x, dtype=np.float64)
            mean = float(np.mean(x))
            if x.size <= 1:
                return mean, 0.0
            sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
            return mean, 1.96 * sem

        reached_rates = np.asarray(reached_counts, dtype=np.float64) / max(self.num_agents, 1)
        collision_rates = np.asarray(collision_counts, dtype=np.float64) / max(self.num_agents, 1)

        reached_mean, reached_ci = _mean_ci95(reached_rates)
        collision_mean, collision_ci = _mean_ci95(collision_rates)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        labels = ["Reached Goal Rate", "Collision Rate"]
        means = [reached_mean, collision_mean]
        cis = [reached_ci, collision_ci]
        ax.bar(labels, means, yerr=cis, capsize=8, color=["#2A9D8F", "#E76F51"])
        ax.set_ylabel("Rate per Episode")
        ax.set_title(
            f"{self.algorithm_name} | Team {self.num_agents} | Faulty {faulty_node}\n"
            "Mean over eval episodes with 95% CI"
        )
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()

        wandb.log(
            {f"faulty_node_{faulty_node}/eval_counts_ci95_bar": wandb.Image(fig)},
            step=step,
        )
        plt.close(fig)

    def _log_ci_summary_table_to_wandb(self, reached_counts, collision_counts, faulty_node, step):
        if len(reached_counts) == 0 or len(collision_counts) == 0:
            return

        def _mean_ci95(x):
            x = np.asarray(x, dtype=np.float64)
            mean = float(np.mean(x))
            if x.size <= 1:
                return mean, 0.0
            sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
            return mean, 1.96 * sem

        reached_mean, reached_ci = _mean_ci95(reached_counts)
        collision_mean, collision_ci = _mean_ci95(collision_counts)

        reached_rate = np.asarray(reached_counts, dtype=np.float64) / max(self.num_agents, 1)
        collision_rate = np.asarray(collision_counts, dtype=np.float64) / max(self.num_agents, 1)
        reached_rate_mean, reached_rate_ci = _mean_ci95(reached_rate)
        collision_rate_mean, collision_rate_ci = _mean_ci95(collision_rate)

        table = wandb.Table(
            columns=[
                "algorithm_name",
                "team_size",
                "faulty_node",
                "metric",
                "mean",
                "ci95",
                "n_episodes",
                "step",
            ]
        )
        n_eps = int(len(reached_counts))
        rows = [
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "reached_goal_count", reached_mean, reached_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "collision_count", collision_mean, collision_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "reached_goal_rate", reached_rate_mean, reached_rate_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "collision_rate", collision_rate_mean, collision_rate_ci, n_eps, int(step)],
        ]
        for row in rows:
            table.add_data(*row)

        wandb.log({"comparison/eval_ci_summary": table}, step=step)

    def _apply_faulty_action(self, actions, faulty_node):
        action_fault = actions.copy()
        if faulty_node >= 0 and faulty_node < action_fault.shape[1]:
            action_fault[:, faulty_node, :] = 0.0
        return action_fault


    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            done_episodes_rewards = []

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).reshape(-1)
                train_episode_rewards += reward_env
                done_episodes_rewards.extend(train_episode_rewards[dones_env].tolist())
                train_episode_rewards[dones_env] = 0.0

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = float(np.mean(done_episodes_rewards))
                    print("some episodes done, average rewards:", aver_episode_rewards)
                    if self.use_wandb:
                        wandb.log({"train_episode_rewards/aver_rewards": aver_episode_rewards}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(
                            "train_episode_rewards",
                            {"aver_rewards": aver_episode_rewards},
                            total_num_steps,
                        )

            if episode % self.eval_interval == 0 and self.use_eval:
                faulty_nodes = getattr(self.all_args, "eval_faulty_node", None)
                if faulty_nodes is None:
                    faulty_nodes = [getattr(self.all_args, "faulty_node", -1)]
                for node in faulty_nodes:
                    self.eval(total_num_steps, int(node))

    def warmup(self):
        obs, share_obs, _ = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                share_obs[:, agent_id],
                obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                None,
                active_masks[:, agent_id],
                None,
            )

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps, faulty_node=-1):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)
        episode_reached_flags = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=bool)
        episode_collision_flags = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=bool)
        episode_reached_counts = []
        episode_collision_counts = []
        per_episode_rows = []

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_actions = self._apply_faulty_action(eval_actions, faulty_node)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(eval_actions)
            self._update_episode_team_flags(eval_infos, episode_reached_flags, episode_collision_flags)

            eval_rewards = np.mean(eval_rewards, axis=1).reshape(-1)
            one_episode_rewards += eval_rewards

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    ep_reward = float(one_episode_rewards[eval_i])
                    eval_episode_rewards.append(ep_reward)
                    reached_count = int(np.sum(episode_reached_flags[eval_i]))
                    collision_count = int(np.sum(episode_collision_flags[eval_i]))
                    episode_reached_counts.append(reached_count)
                    episode_collision_counts.append(collision_count)
                    per_episode_rows.append(
                        {
                            "algorithm_name": self.algorithm_name,
                            "team_size": int(self.num_agents),
                            "faulty_node": int(faulty_node),
                            "eval_step": int(total_num_steps),
                            "episode_index": int(eval_episode),
                            "reached_goal_count": reached_count,
                            "collision_count": collision_count,
                            "reached_goal_rate": float(reached_count / max(self.num_agents, 1)),
                            "collision_rate": float(collision_count / max(self.num_agents, 1)),
                            "episode_reward": ep_reward,
                        }
                    )
                    one_episode_rewards[eval_i] = 0.0
                    episode_reached_flags[eval_i] = False
                    episode_collision_flags[eval_i] = False

            if eval_episode >= self.all_args.eval_episodes:
                key_average = 'faulty_node_' + str(faulty_node) + '/eval_average_episode_rewards'
                key_max = 'faulty_node_' + str(faulty_node) + '/eval_max_episode_rewards'
                key_reached_count = 'faulty_node_' + str(faulty_node) + '/eval_reached_goal_count'
                key_collision_count = 'faulty_node_' + str(faulty_node) + '/eval_collision_count'
                key_reached_rate = 'faulty_node_' + str(faulty_node) + '/eval_reached_goal_rate'
                key_collision_rate = 'faulty_node_' + str(faulty_node) + '/eval_collision_rate'

                eval_env_infos = {
                    key_average: eval_episode_rewards,
                    key_max: [np.max(eval_episode_rewards) if len(eval_episode_rewards) > 0 else 0.0],
                    key_reached_count: episode_reached_counts,
                    key_collision_count: episode_collision_counts,
                    key_reached_rate: [c / max(self.num_agents, 1) for c in episode_reached_counts],
                    key_collision_rate: [c / max(self.num_agents, 1) for c in episode_collision_counts],
                }
                self.log_env(eval_env_infos, total_num_steps)
                self._append_eval_team_stats(per_episode_rows)
                print("faulty_node {} eval_average_episode_rewards is {}.".format(
                    faulty_node, np.mean(eval_episode_rewards) if len(eval_episode_rewards) > 0 else 0.0)
                )
                print(
                    "faulty_node {} avg reached_goals {:.3f}/{}, avg collisions {:.3f}/{}.".format(
                        faulty_node,
                        np.mean(episode_reached_counts) if len(episode_reached_counts) > 0 else 0.0,
                        self.num_agents,
                        np.mean(episode_collision_counts) if len(episode_collision_counts) > 0 else 0.0,
                        self.num_agents,
                    )
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "eval_average_episode_rewards": np.mean(eval_episode_rewards) if len(eval_episode_rewards) > 0 else 0.0,
                            "eval_reached_goal_count": np.mean(episode_reached_counts) if len(episode_reached_counts) > 0 else 0.0,
                            "eval_collision_count": np.mean(episode_collision_counts) if len(episode_collision_counts) > 0 else 0.0,
                            "eval_reached_goal_rate": np.mean(episode_reached_counts) / max(self.num_agents, 1) if len(episode_reached_counts) > 0 else 0.0,
                            "eval_collision_rate": np.mean(episode_collision_counts) / max(self.num_agents, 1) if len(episode_collision_counts) > 0 else 0.0,
                            "eval_team_size": int(self.num_agents),
                            "eval_faulty_node": int(faulty_node),
                        },
                        step=total_num_steps,
                    )
                    self._update_aggregate_eval_stats(
                        faulty_node,
                        episode_reached_counts,
                        episode_collision_counts,
                    )
                    if len(per_episode_rows) > 0:
                        table_columns = list(per_episode_rows[0].keys())
                        table = wandb.Table(columns=table_columns)
                        for row in per_episode_rows:
                            table.add_data(*[row[c] for c in table_columns])
                        wandb.log(
                            {f"faulty_node_{faulty_node}/eval_team_stats_table": table},
                            step=total_num_steps,
                        )
                    agg_reached_counts = self._agg_reached_counts_by_fault.get(faulty_node, episode_reached_counts)
                    agg_collision_counts = self._agg_collision_counts_by_fault.get(faulty_node, episode_collision_counts)
                    self._log_ci_bar_chart_to_wandb(
                        agg_reached_counts,
                        agg_collision_counts,
                        faulty_node,
                        total_num_steps,
                    )
                    self._log_ci_summary_table_to_wandb(
                        agg_reached_counts,
                        agg_collision_counts,
                        faulty_node,
                        total_num_steps,
                    )
                break
