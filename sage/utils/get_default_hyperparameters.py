from pathlib import Path
from transformers import AutoConfig
from transformers import MBartForConditionalGeneration


def config_decoder(config):
    decoder_type = config.get('model', {}).get('decoder_type', 'LD')

    if decoder_type == 'LD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes=True, config=AutoConfig.from_pretrained(str(Path(config['model']['visual_encoder']) / 'config.json')))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes=True, config=AutoConfig.from_pretrained(str(Path(config['model']['transformer']) / 'LLMD_config.json')))


def get_freeze_backbone():
    return False
