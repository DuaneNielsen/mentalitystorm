from unittest import TestCase
from mentalitystorm import config

class TestConfig(TestCase):
    def test_config_loading(self):
        print(config.config['last_run_id'])

