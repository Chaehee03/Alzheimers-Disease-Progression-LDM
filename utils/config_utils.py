def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value

def validate_class_config(condition_config):
    assert 'class_condition_config' in condition_config, \
        "Class conditioning desired but class condition config missing."
    assert 'num_classes' in condition_config['class_condition_config'], \
    "num_class missing in class condition config"

def validate_context_config(condition_config):
    assert 'context_condition_config' in condition_config, \
        "Context conditioning desired but context condition config missing"
    assert 'context_embed_dim' in condition_config['context_condition_config'], \
        "context_embed_dim missing in context condition config"

