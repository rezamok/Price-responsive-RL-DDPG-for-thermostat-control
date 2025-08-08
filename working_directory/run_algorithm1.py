
import os
os.chdir('C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL')

import sys
sys.path.insert(0,'..')
# Warningsa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

# imports
from kosta.environment import get_env
from kosta.one_for_many import train_both_networks
import kosta.hyperparams as hp

# Change working directory
if sys.platform == "win32":
    os.chdir(os.path.abspath(os.path.join(os.path.dirname("__file__"),'..')))
else:
    sys.path.append("../")


if __name__ == '__main__':
    env, _, _, _= get_env(env_model=hp.env_model,set_room=hp.main_room)
    train_both_networks(env,network2=None)  