import os
import warnings
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict, Any

from stable_baselines.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization

from agents.helpers import evaluate_lstm_policy, prepare_performance_plot, analyze_agent
from agents.environments import UMAREnv

from abc import ABC
import typing

from stable_baselines import logger

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error


class BaseCallback(ABC):
    """
    Base class for callback.
    :param verbose: (int)
    """

    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()
        # The RL model
        self.model = None  # type: Optional[BaseRLModel]
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals = None  # type: Optional[Dict[str, Any]]
        self.globals = None  # type: Optional[Dict[str, Any]]
        self.logger = None  # type: Optional[logger.Logger]
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, model: 'BaseRLModel') -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()
        self.logger = logger.Logger.CURRENT
        self._init_callback()

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Updates the local variables of the training process
        For reference to which variables are accessible,
        check each individual algorithm's documentation
        :param `locals_`: (Dict[str, Any]) current local variables
        """
        self.locals.update(locals_)

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.
    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self

    def init_callback(self, model: 'BaseRLModel') -> None:
        super(EventCallback, self).init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.
    :param callbacks: (List[BaseCallback]) A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            #self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.
    :param callback: (Callable)
    :param verbose: (int)
    """

    def __init__(self, callback, verbose=0):
        super(ConvertCallback, self).__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True



class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).
    It must be used with the `EvalCallback`.
    :param reward_threshold: (float)  Minimum expected reward per episode
        to stop training.
    :param verbose: (int)
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, ("`StopTrainingOnRewardThreshold` callback must be used "
                                         "with an `EvalCallback`")
        # Convert np.bool to bool, otherwise callback.on_step() is False won't work
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print("Stopping training because the mean reward {:.2f} "
                  " is above the threshold {}".format(self.parent.best_mean_reward, self.reward_threshold))
        return continue_training


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every `n_steps` timesteps
    :param n_steps: (int) Number of timesteps between two trigger.
    :param callback: (BaseCallback) Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: BaseCallback):
        super(EveryNTimesteps, self).__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class EvalCallback_old(EventCallback):
    """
    Custom callback working for LSTM policies - this is again  a copy of the 'EvalCallback' in
    stable-baselines, where we simply modify the evaluating function to make it work
    with recurrent policies, using the our custom evaluation function
    """

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 fixed_sequences: bool = False,
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 normalizing: bool = True,
                 all_goals: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.normalizing = normalizing
        self.all_goals = all_goals

        # Define comparing sequences
        if fixed_sequences:
            if eval_env.her: # isinstance(eval_env, UMAREnv): #
                lengths = np.array([len(seq) for seq in eval_env.umar_model.test_sequences])
            else:
                lengths = np.array([len(seq) for seq in eval_env.envs[0].umar_model.test_sequences])
            self.sequences = []
            for i in range(n_eval_episodes):
                if eval_env.her: #isinstance(eval_env, UMAREnv): #
                    if len(lengths) > 1:
                        sequence = np.random.choice(eval_env.umar_model.test_sequences, p=lengths / sum(lengths))
                    else:
                        sequence = eval_env.umar_model.test_sequences[0]
                    if len(sequence) > eval_env.threshold_length + eval_env.n_autoregression:
                        start = np.random.randint(eval_env.n_autoregression, len(sequence) - eval_env.threshold_length + 1)
                        self.sequences.append(sequence[start - eval_env.n_autoregression: start + eval_env.threshold_length])
                    else:
                        self.sequences.append(sequence)

                else:
                    if len(lengths) > 1:
                        sequence = np.random.choice(eval_env.envs[0].umar_model.test_sequences, p=lengths / sum(lengths))
                    else:
                        sequence = eval_env.envs[0].umar_model.test_sequences[0]
                    if len(sequence) > eval_env.envs[0].threshold_length + eval_env.envs[0].n_autoregression:
                        start = np.random.randint(eval_env.envs[0].n_autoregression, len(sequence) - eval_env.envs[0].threshold_length + 1)
                        self.sequences.append(sequence[start - eval_env.envs[0].n_autoregression: start + eval_env.envs[0].threshold_length])
                    else:
                        self.sequences.append(sequence)

        else:
            self.sequences = None

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv) and not eval_env.her: #not isinstance(eval_env, UMAREnv): #
            eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            print("Evaluating...")

            ###### Modification ######
            episode_rewards, episode_lengths = evaluate_lstm_policy(self.model, self.eval_env,
                                                                    n_eval_episodes=self.n_eval_episodes,
                                                                    sequences=self.sequences,
                                                                    render=self.render,
                                                                    deterministic=self.deterministic,
                                                                    normalizing=self.normalizing,
                                                                    all_goals=self.all_goals,
                                                                    return_episode_rewards=True)
            ##########################

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    From https://github.com/araffin/rl-baselines-zoo/blob/fd9d38862047d7fd4f67be8eb3f6736e093eac9f/utils/callbacks.py

    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, '{}_{}_steps.pkl'.format(self.name_prefix, self.num_timesteps))
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True


class CompareAgentCallback_old(BaseCallback):
    """
    Callback running
    """

    def __init__(self, rbagent, eval_env,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 render: bool = False,
                 normalizing: bool = True,
                 verbose: int = 1):
        super(CompareAgentCallback, self).__init__(verbose=verbose)

        self.rbagent = rbagent
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.render = render
        self.verbose = verbose
        self.normalizing = normalizing

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv) and not self.eval_env.her: #isinstance(self.eval_env, UMAREnv):#
            eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'RB')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folder if needed
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            if self.eval_env.her: # isinstance(self.eval_env, UMAREnv):#
                episode_cumul_rewards, episode_length = self.rbagent.run(self.eval_env.last_sequence,
                                                                         self.eval_env.last_goal_number,
                                                                         render=self.render)
            else:
                if self.normalizing:
                    episode_cumul_rewards, episode_length = self.rbagent.run(self.eval_env.venv.venv.envs[0].last_sequence,
                                                                             render=self.render)
                else:
                    episode_cumul_rewards, episode_length = self.rbagent.run(self.eval_env.venv.envs[0].last_sequence,
                                                                             render=self.render)

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f}".format(self.num_timesteps, episode_cumul_rewards))
                print("Episode length: {:.2f}".format(episode_length))

        return True


class EvalCallback_old_two(EventCallback):
    """
    Custom callback working for LSTM policies - this is again  a copy of the 'EvalCallback' in
    stable-baselines, where we simply modify the evaluating function to make it work
    with recurrent policies, using the our custom evaluation function
    """

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 rbagent=None,
                 bangbang=None,
                 unavoidable=None,
                 best_mean_reward=None,
                 best_mean_comfort=None,
                 best_mean_prices=None,
                 fixed_sequences: bool = False,
                 sequences: list = None,
                 evaluate_rb: bool = True,
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 normalizing: bool = True,
                 all_goals: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf if best_mean_reward is None else best_mean_reward
        self.best_mean_comfort = np.inf if best_mean_comfort is None else best_mean_comfort
        self.best_mean_prices = -1 if best_mean_prices is None else best_mean_prices
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.normalizing = normalizing
        self.all_goals = all_goals
        self.rbagent = rbagent
        self.bangbang = bangbang
        self.unavoidable = unavoidable
        self.mean_rb_rewards = 1
        self.std_rb_rewards = 0

        self.rb_rewards = []
        self.rb_comfort_violations = []
        self.rb_prices = []

        self.bangbang_rewards = []
        self.bangbang_comfort_violations = []
        self.bangbang_prices = []

        self.unavoidable_rewards = []
        self.unavoidable_comfort_violations = []
        self.unavoidable_prices = []

        # Define comparing sequences
        if fixed_sequences:
            if sequences is not None:
                self.sequences = sequences
            else:
                if isinstance(eval_env, UMAREnv):
                    lengths = np.array([len(seq) for seq in eval_env.umar_model.test_sequences])
                else:
                    lengths = np.array([len(seq) for seq in eval_env.envs[0].umar_model.test_sequences])
                self.sequences = []
                for i in range(n_eval_episodes):
                    if isinstance(eval_env, UMAREnv):
                        if len(lengths) > 1:
                            sequence = np.random.choice(eval_env.umar_model.test_sequences, p=lengths / sum(lengths))
                        else:
                            sequence = eval_env.umar_model.test_sequences[0]
                        if len(sequence) > eval_env.threshold_length + eval_env.n_autoregression:
                            start = np.random.randint(eval_env.n_autoregression,
                                                      len(sequence) - eval_env.threshold_length + 1)
                            self.sequences.append(
                                sequence[start - eval_env.n_autoregression: start + eval_env.threshold_length])
                        else:
                            self.sequences.append(sequence)

                    else:
                        if len(lengths) > 1:
                            sequence = np.random.choice(eval_env.envs[0].umar_model.test_sequences,
                                                        p=lengths / sum(lengths))
                        else:
                            sequence = eval_env.envs[0].umar_model.test_sequences[0]
                        if len(sequence) > eval_env.envs[0].threshold_length + eval_env.envs[0].n_autoregression:
                            start = np.random.randint(eval_env.envs[0].n_autoregression,
                                                      len(sequence) - eval_env.envs[0].threshold_length + 1)
                            self.sequences.append(sequence[start - eval_env.envs[0].n_autoregression: start + eval_env.envs[
                                0].threshold_length])
                        else:
                            self.sequences.append(sequence)

                if (rbagent is not None) and (evaluate_rb):
                    print("Evaluating the performance of the RB agents...")
                    for sequence in self.sequences:
                        if isinstance(eval_env, UMAREnv):
                            if all_goals:
                                for goal_number in range(len(eval_env.desired_goals)):
                                    rb_reward, _ = self.rbagent.run(sequence=sequence,
                                                                    goal_number=eval_env.last_goal_number,
                                                                    render=self.render)
                                    bangbang_reward, _ = self.bangbang.run(sequence=sequence,
                                                                           goal_number=eval_env.last_goal_number,
                                                                           render=self.render)
                            else:
                                rb_reward, _ = self.rbagent.run(sequence=sequence,
                                                                goal_number=eval_env.last_goal_number,
                                                                render=self.render)
                                bangbang_reward, _ = self.bangbang.run(sequence=sequence,
                                                                       goal_number=eval_env.last_goal_number,
                                                                       render=self.render)
                        else:
                            rb_reward, _ = self.rbagent.run(sequence=sequence,
                                                            render=self.render)
                            bangbang_reward, _ = self.bangbang.run(sequence=sequence,
                                                                   render=self.render)
                        unavoidable_reward, _ = self.unavoidable.run(sequence=sequence,
                                                                     render=self.render)

                        self.rb_rewards.append(rb_reward)
                        self.rb_comfort_violations.append(np.sum(self.rbagent.env.last_comfort_violations))
                        self.rb_prices.append(np.sum(self.rbagent.env.last_prices))

                        self.bangbang_rewards.append(bangbang_reward)
                        self.bangbang_comfort_violations.append(np.sum(self.bangbang.env.last_comfort_violations))
                        self.bangbang_prices.append(np.sum(self.bangbang.env.last_prices))

                        self.unavoidable_rewards.append(unavoidable_reward)
                        self.unavoidable_comfort_violations.append(np.sum(self.unavoidable.env.last_comfort_violations))
                        self.unavoidable_prices.append(np.sum(self.unavoidable.env.last_prices))

                    print(f"Rule-based performance on the testing set:")
                    print(f"Rewards:              {np.mean(self.rb_rewards):.2f} +/- {np.std(self.rb_rewards):.2f}.")
                    print(f"Comfort violations:   {np.mean(self.rb_comfort_violations):.2f} +/- {np.std(self.rb_comfort_violations):.2f}.")
                    print(f"Prices:               {np.mean(self.rb_prices):.2f} +/- {np.std(self.rb_prices):.2f}.\n")

                    print(f"Bang-bang performance on the testing set:")
                    print(f"Rewards:              {np.mean(self.bangbang_rewards):.2f} +/- {np.std(self.bangbang_rewards):.2f}.")
                    print(
                        f"Comfort violations:   {np.mean(self.bangbang_comfort_violations):.2f} +/- {np.std(self.bangbang_comfort_violations):.2f}.")
                    print(f"Prices:               {np.mean(self.bangbang_prices):.2f} +/- {np.std(self.bangbang_prices):.2f}.\n")

                    print(f"Unavoidable performance on the testing set:")
                    print(
                        f"Rewards:              {np.mean(self.unavoidable_rewards):.2f} +/- {np.std(self.unavoidable_rewards):.2f}.")
                    print(
                        f"Comfort violations:   {np.mean(self.unavoidable_comfort_violations):.2f} +/- {np.std(self.unavoidable_comfort_violations):.2f}.")
                    print(
                        f"Prices:               {np.mean(self.unavoidable_prices):.2f} +/- {np.std(self.unavoidable_prices):.2f}.\n")

        else:
            self.sequences = None

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv) and not isinstance(eval_env, UMAREnv):
            eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.rewards = []
        self.comfort_violations = []
        self.prices = []

        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            print("Evaluating...")

            ###### Modification ######
            episode_rewards, comfort_violations, prices, episode_lengths = \
                evaluate_lstm_policy(self.model,
                                     self.eval_env,
                                     n_eval_episodes=self.n_eval_episodes,
                                     sequences=self.sequences,
                                     render=self.render,
                                     deterministic=self.deterministic,
                                     normalizing=self.normalizing,
                                     all_goals=self.all_goals,
                                     return_episode_rewards=False)
            ##########################

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

            self.rewards.append(episode_rewards)
            self.comfort_violations.append(comfort_violations)
            self.prices.append(prices)

            if (len(self.prices) > 200) & (self.best_mean_prices < 0):
                self.best_mean_prices = np.inf

            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"\nEval num_timesteps={self.num_timesteps} ----------------- Rewards:     {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Comfort violations:   {np.mean(comfort_violations):.2f} +/- {np.std(comfort_violations):.2f}.")
                print(f"Prices:               {np.mean(prices):.2f} +/- {np.std(prices):.2f}.\n")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            if np.mean(comfort_violations) < self.best_mean_comfort:
                if self.verbose > 0:
                    print("New best mean comfort!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model_comfort'))
                self.best_mean_comfort = np.mean(comfort_violations)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            if np.mean(prices) < self.best_mean_prices:
                if self.verbose > 0:
                    print("New best mean cost!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model_price'))
                self.best_mean_prices = np.mean(prices)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


class CompareAgentCallback(BaseCallback):
    """
    Callback running
    """

    def __init__(self, eval_env, rbagent, bangbang, unavoidable,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 normalizing: bool = True,
                 deterministic: bool = True,
                 sequences: list = None,
                 verbose: int = 1):
        super(CompareAgentCallback, self).__init__(verbose=verbose)

        self.rbagent = rbagent
        self.bangbang = bangbang
        self.unavoidable = unavoidable
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.verbose = verbose
        self.normalizing = normalizing
        self.sequences = sequences
        self.deterministic = deterministic

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv) and not isinstance(self.eval_env, UMAREnv):
            eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self._eval_env = eval_env
                
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'RB')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folder if needed
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self._eval_env)
            
            if isinstance(self.eval_env, UMAREnv):
                self.eval_env = self._eval_env
            else:
                if self.normalizing:
                    self.eval_env = self._eval_env.venv.venv.envs[0]
                else:
                    self.eval_env = self._eval_env.venv.envs[0]
            
            evaluate_lstm_policy(self.model,
                                 self.eval_env,
                                 n_eval_episodes=self.n_eval_episodes,
                                 sequences=self.sequences,
                                        render=False,
                                        deterministic=self.deterministic,
                                        normalizing=self.normalizing,
                                        all_goals=False,
                                        return_episode_rewards=True)

            self.unavoidable.run(sequence=self.eval_env.last_sequence,
                                 init_temp=self.eval_env.last_init_temp,
                                 render=False)
                
            print(f"__________________________\nUnavoidable:")
            print(f"\nReward:             {np.sum(np.array(self.unavoidable.env.last_rewards)):.2f}")
            print(f"Comfort violations:   {np.sum(np.array(self.unavoidable.env.last_comfort_violations)):.2f}")
            print(f"Total benefits/costs: {np.sum(np.array(self.unavoidable.env.last_prices)):.2f}\n")

            self.bangbang.run(sequence=self.eval_env.last_sequence,
                              goal_number=self.eval_env.last_goal_number,
                              init_temp=self.eval_env.last_init_temp,
                              render=False)

            axes, data = prepare_performance_plot(env=self.bangbang.env,
                                               sequence=self.bangbang.env.last_sequence,
                                               data=self.bangbang.env.last_data,
                                               rewards=self.bangbang.env.last_rewards,
                                               electricity_imports=self.bangbang.env.last_electricity_imports,
                                               lower_bounds=self.bangbang.env.last_lower_bounds,
                                               upper_bounds=self.bangbang.env.last_upper_bounds,
                                               prices=self.bangbang.env.last_prices,
                                               comfort_violations=self.bangbang.env.last_comfort_violations,
                                               battery_soc=self.bangbang.env.last_battery_soc,
                                               battery_powers=self.bangbang.env.last_battery_powers,
                                               label="Bang-bang",
                                               color="red",
                                               elec_price=False,
                                               print_=True,
                                               show_=False)

            analyze_agent(name="Bang bang",
                          env=self.bangbang.env,
                          data=data,
                          rewards=self.bangbang.env.last_rewards,
                          comfort_violations=self.bangbang.env.last_comfort_violations,
                          prices=self.bangbang.env.last_prices,
                          electricity_imports=self.bangbang.env.last_electricity_imports,
                          lower_bounds=self.bangbang.env.last_lower_bounds,
                          upper_bounds=self.bangbang.env.last_upper_bounds,
                          battery_soc=self.bangbang.env.last_battery_soc,
                          battery_powers=self.bangbang.env.last_battery_powers)

            self.rbagent.run(self.eval_env.last_sequence,
                             self.eval_env.last_goal_number,
                             init_temp=self.eval_env.last_init_temp,
                             render=False)

            axes, data = prepare_performance_plot(env=self.rbagent.env,
                                               sequence=self.rbagent.env.last_sequence,
                                               data=self.rbagent.env.last_data,
                                               rewards=self.rbagent.env.last_rewards,
                                               electricity_imports=self.rbagent.env.last_electricity_imports,
                                               lower_bounds=self.rbagent.env.last_lower_bounds,
                                               upper_bounds=self.rbagent.env.last_upper_bounds,
                                               prices=self.rbagent.env.last_prices,
                                               comfort_violations=self.rbagent.env.last_comfort_violations,
                                               battery_soc=self.rbagent.env.last_battery_soc,
                                               battery_powers=self.rbagent.env.last_battery_powers,
                                               label="Rule-based",
                                               color="orange",
                                               elec_price=False,
                                               print_=False,
                                               show_=False,
                                               axes=axes)

            analyze_agent(name="Rule based",
                          env=self.rbagent.env,
                          data=data,
                          rewards=self.rbagent.env.last_rewards,
                          comfort_violations=self.rbagent.env.last_comfort_violations,
                          prices=self.rbagent.env.last_prices,
                          electricity_imports=self.rbagent.env.last_electricity_imports,
                          lower_bounds=self.rbagent.env.last_lower_bounds,
                          upper_bounds=self.rbagent.env.last_upper_bounds,
                          battery_soc=self.rbagent.env.last_battery_soc,
                          battery_powers=self.rbagent.env.last_battery_powers)

            _, data = prepare_performance_plot(env=self.eval_env,
                                                  sequence=self.eval_env.last_sequence,
                                                  data=self.eval_env.last_data,
                                                  rewards=self.eval_env.last_rewards,
                                                  electricity_imports=self.eval_env.last_electricity_imports,
                                                  lower_bounds=self.eval_env.last_lower_bounds,
                                                  upper_bounds=self.eval_env.last_upper_bounds,
                                                  prices=self.eval_env.last_prices,
                                                  comfort_violations=self.eval_env.last_comfort_violations,
                                                  battery_soc=self.eval_env.last_battery_soc,
                                                  battery_powers=self.eval_env.last_battery_powers,
                                                  label="RL Agent",
                                                  color="blue",
                                                  elec_price=True,
                                                  print_=False,
                                                  show_=False,
                                                axes=axes)
            analyze_agent(name="RL Agent",
                          env=self.eval_env,
                          data=data,
                          rewards=self.eval_env.last_rewards,
                          comfort_violations=self.eval_env.last_comfort_violations,
                          prices=self.eval_env.last_prices,
                          electricity_imports=self.eval_env.last_electricity_imports,
                          lower_bounds=self.eval_env.last_lower_bounds,
                          upper_bounds=self.eval_env.last_upper_bounds,
                          battery_soc=self.eval_env.last_battery_soc,
                          battery_powers=self.eval_env.last_battery_powers)


            plt.tight_layout()
            plt.show()
            plt.close()

        return True


class EvalCallback(EventCallback):
    """
    Custom callback working for LSTM policies - this is again  a copy of the 'EvalCallback' in
    stable-baselines, where we simply modify the evaluating function to make it work
    with recurrent policies, using the our custom evaluation function
    """

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 rbagent=None,
                 bangbang=None,
                 unavoidable=None,
                 best_mean_reward=None,
                 best_mean_comfort=None,
                 best_mean_prices=None,
                 fixed_sequences: bool = False,
                 sequences: list = None,
                 init_temps: list = None,
                 evaluate_rb: bool = True,
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 normalizing: bool = True,
                 all_goals: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf if best_mean_reward is None else best_mean_reward
        self.best_mean_comfort = np.inf if best_mean_comfort is None else best_mean_comfort
        self.best_mean_prices = -1 if best_mean_prices is None else best_mean_prices
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.normalizing = normalizing
        self.all_goals = all_goals
        self.rbagent = rbagent
        self.bangbang = bangbang
        self.unavoidable = unavoidable
        self.mean_rb_rewards = 1
        self.std_rb_rewards = 0

        self.rb_rewards = []
        self.rb_comfort_violations = []
        self.rb_prices = []

        self.bangbang_rewards = []
        self.bangbang_comfort_violations = []
        self.bangbang_prices = []

        self.unavoidable_rewards = []
        self.unavoidable_comfort_violations = []
        self.unavoidable_prices = []

        self.init_temps = init_temps

        # Define comparing sequences
        if fixed_sequences:
            if isinstance(eval_env, UMAREnv):
                env = eval_env
            else:
                env = eval_env.envs[0]

            if sequences is not None:
                self.sequences = sequences
            else:
                lengths = np.array([len(seq) for seq in env.umar_model.test_sequences])
                self.sequences = []
                for i in range(n_eval_episodes):
                    if len(lengths) > 1:
                        sequence = np.random.choice(env.umar_model.test_sequences, p=lengths / sum(lengths))
                    else:
                        sequence = env.umar_model.test_sequences[0]
                    if len(sequence) > env.threshold_length + env.n_autoregression:
                        start = np.random.randint(env.n_autoregression, len(sequence) - env.threshold_length + 1)
                        self.sequences.append(sequence[start - env.n_autoregression: start + env.threshold_length])
                    else:
                        self.sequences.append(sequence)

            if (rbagent is not None) and (evaluate_rb):
                print("Evaluating the performance of the RB agents...")
                for num, sequence in enumerate(self.sequences):
                    if env.her and all_goals:
                        for goal_number in range(len(eval_env.desired_goals)):
                            rb_reward, _ = self.rbagent.run(sequence=sequence,
                                                            goal_number=goal_number,
                                                            init_temp=self.init_temps[num] if self.init_temps is not None else None,
                                                            render=self.render)
                            bangbang_reward, _ = self.bangbang.run(sequence=sequence,
                                                                   goal_number=goal_number,
                                                                   init_temp=self.init_temps[num] if self.init_temps is not None else None,
                                                                   render=self.render)
                    else:
                        rb_reward, _ = self.rbagent.run(sequence=sequence,
                                                        goal_number=env.last_goal_number,
                                                        init_temp=self.init_temps[num] if self.init_temps is not None else None,
                                                        render=self.render)
                        bangbang_reward, _ = self.bangbang.run(sequence=sequence,
                                                               goal_number=env.last_goal_number,
                                                               init_temp=self.init_temps[num] if self.init_temps is not None else None,
                                                               render=self.render)
                    unavoidable_reward, _ = self.unavoidable.run(sequence=sequence,
                                                                 init_temp=self.init_temps[num] if self.init_temps is not None else None,
                                                                 render=self.render)

                    self.rb_rewards.append(rb_reward)
                    self.rb_comfort_violations.append(np.sum(self.rbagent.env.last_comfort_violations))
                    self.rb_prices.append(np.sum(self.rbagent.env.last_prices))

                    self.bangbang_rewards.append(bangbang_reward)
                    self.bangbang_comfort_violations.append(np.sum(self.bangbang.env.last_comfort_violations))
                    self.bangbang_prices.append(np.sum(self.bangbang.env.last_prices))

                    self.unavoidable_rewards.append(unavoidable_reward)
                    self.unavoidable_comfort_violations.append(np.sum(self.unavoidable.env.last_comfort_violations))
                    self.unavoidable_prices.append(np.sum(self.unavoidable.env.last_prices))

                print(f"Rule-based performance on the testing set:")
                print(f"Rewards:              {np.mean(self.rb_rewards):.2f} +/- {np.std(self.rb_rewards):.2f}.")
                print(f"Comfort violations:   {np.mean(self.rb_comfort_violations):.2f} +/- {np.std(self.rb_comfort_violations):.2f}.")
                print(f"Prices:               {np.mean(self.rb_prices):.2f} +/- {np.std(self.rb_prices):.2f}.\n")

                print(f"Bang-bang performance on the testing set:")
                print(f"Rewards:              {np.mean(self.bangbang_rewards):.2f} +/- {np.std(self.bangbang_rewards):.2f}.")
                print(
                    f"Comfort violations:   {np.mean(self.bangbang_comfort_violations):.2f} +/- {np.std(self.bangbang_comfort_violations):.2f}.")
                print(f"Prices:               {np.mean(self.bangbang_prices):.2f} +/- {np.std(self.bangbang_prices):.2f}.\n")

                print(f"Unavoidable performance on the testing set:")
                print(
                    f"Rewards:              {np.mean(self.unavoidable_rewards):.2f} +/- {np.std(self.unavoidable_rewards):.2f}.")
                print(
                    f"Comfort violations:   {np.mean(self.unavoidable_comfort_violations):.2f} +/- {np.std(self.unavoidable_comfort_violations):.2f}.")
                print(
                    f"Prices:               {np.mean(self.unavoidable_prices):.2f} +/- {np.std(self.unavoidable_prices):.2f}.\n")

        else:
            self.sequences = None

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv) and not isinstance(eval_env, UMAREnv):
            eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.rewards = []
        self.comfort_violations = []
        self.prices = []

        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            print("Evaluating...")

            ###### Modification ######
            episode_rewards, comfort_violations, prices, episode_lengths = \
                evaluate_lstm_policy(self.model,
                                     self.eval_env,
                                     n_eval_episodes=self.n_eval_episodes,
                                     sequences=self.sequences,
                                     init_temps=self.init_temps,
                                     render=self.render,
                                     deterministic=self.deterministic,
                                     normalizing=self.normalizing,
                                     all_goals=self.all_goals,
                                     return_episode_rewards=False)
            ##########################

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

            self.rewards.append(episode_rewards)
            self.comfort_violations.append(comfort_violations)
            self.prices.append(prices)

            if (len(self.prices) > 200) & (self.best_mean_prices < 0):
                self.best_mean_prices = np.inf

            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"\nEval num_timesteps={self.num_timesteps} ----------------- Rewards:     {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Comfort violations:   {np.mean(comfort_violations):.2f} +/- {np.std(comfort_violations):.2f}.")
                print(f"Prices:               {np.mean(prices):.2f} +/- {np.std(prices):.2f}.\n")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            if np.mean(comfort_violations) < self.best_mean_comfort:
                if self.verbose > 0:
                    print("New best mean comfort!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model_comfort'))
                self.best_mean_comfort = np.mean(comfort_violations)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            if np.mean(prices) < self.best_mean_prices:
                if self.verbose > 0:
                    print("New best mean cost!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model_price'))
                self.best_mean_prices = np.mean(prices)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
