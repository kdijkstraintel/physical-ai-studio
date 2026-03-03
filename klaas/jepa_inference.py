from physicalai.gyms import PushTGym
from physicalai.policies import JEPA

policy = JEPA(hf_weights="jepa_wm_pusht")
policy.eval()

push_t = PushTGym(observation_height=policy.config.img_size,
                  observation_width=policy.config.img_size,
                  #with_velocity=policy.config.with_velocity
                  )
push_t.shape = 'T'

obs, state = push_t.reset()
policy(obs)

