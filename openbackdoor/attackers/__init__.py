from .attacker import Attacker
from .ep_attacker import EPAttacker
from .sos_attacker import SOSAttacker
from .neuba_attacker import NeuBAAttacker
from .por_attacker import PORAttacker
from .lwp_attacker import LWPAttacker
from .lws_attacker import LWSAttacker
from .ripples_attacker import RIPPLESAttacker


class OrderBkdAttacker(Attacker):
    def __init__(self, metrics=["accuracy"], **kwargs):
        super().__init__(
            poisoner={"name": "ordbkd"},
            train={"name": "base"},
            metrics=metrics,
            sample_metrics=[],
            **kwargs
        )


ATTACKERS = {
    "base": Attacker,
    "ep": EPAttacker,
    "sos": SOSAttacker,
    "neuba": NeuBAAttacker,
    "por": PORAttacker,
    "lwp": LWPAttacker,
    'lws': LWSAttacker,
    'ripples': RIPPLESAttacker,
    'ordbkd': OrderBkdAttacker
}




def load_attacker(config):
    return ATTACKERS[config["name"].lower()](**config)
