from rl_games.common import algo_observer
import torch

class RewardObserver(algo_observer.AlgoObserver):

    def after_init(self, algo):
        self.algo = algo

    def after_print_stats(self, frame, epoch_num, total_time):
        mean_reward = None

        rewards = getattr(self.algo, "game_rewards", None)

        if rewards is not None:
            # Case 1: rl_games AverageMeter (most common)
            if hasattr(rewards, "get_mean"):
                mean_reward = rewards.get_mean()

            # Case 2: Torch tensor (rare, but safe)
            elif torch.is_tensor(rewards) and rewards.numel() > 0:
                mean_reward = rewards.mean().item()
        if epoch_num % 10 == 0:
            print(
                f"[observer] frame={frame} "
                f"epoch={epoch_num} "
                f"mean_episode_reward={mean_reward}"
            )
