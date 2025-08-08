import tensorflow as tf

from stable_baselines.common.policies import LstmPolicy
from stable_baselines.td3.policies import FeedForwardPolicy

from parameters import agent_kwargs

class CustomLSTMPolicy(LstmPolicy):
    """
    Define a new policy
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps=96, n_batch=96, reuse=False, **_kwargs):
        n_lstm = _kwargs["n_lstm"]
        extraction_size = _kwargs["extraction_size"]
        vf = _kwargs["vf_layers"]
        pi = _kwargs["pi_layers"]
        del _kwargs["n_lstm"]
        del _kwargs["extraction_size"]
        del _kwargs["vf_layers"]
        del _kwargs["pi_layers"]
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm, reuse=reuse,
                         net_arch=[extraction_size, 'lstm', dict(vf=vf, pi=pi)],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


class CustomMlpPolicy(FeedForwardPolicy):
    """
    Define a new policy
    """

    def __init__(self, sess, ob_space, ac_space, layers=agent_kwargs["vf_layers"], **_kwargs):
        super().__init__(sess, ob_space, ac_space,
                         layers=layers, cnn_extractor=None, feature_extraction="mlp", layer_norm=True,
                         act_fun=tf.nn.relu, **_kwargs)
