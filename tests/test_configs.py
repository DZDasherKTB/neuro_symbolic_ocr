from src.pipeline.config_loader import ConfigLoader

def test_config_values():
  cfg = ConfigLoader()
  assert cfg.pipeline.device in ["cuda", "cpu"]
  assert cfg.models.recognition.beam_size == 5
  print("Config Loader: PASS")

if __name__ == "__main__":
  test_config_values()