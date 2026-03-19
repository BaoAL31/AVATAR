import sys
sys.path.insert(0, '/home/jembo/AVATAR/models/usr')

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST