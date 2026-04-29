from .deit import DeiTRKT

def load_model(model_name: str, num_classes: int = 1000, pretrained: bool = True, checkpoint_path: str = "", token_manager=None, can_tokens_revive: bool = False):
    if 'deit' in model_name:
        _, model_size = model_name.split('_')
        model = DeiTRKT(model_size=model_size, num_classes=num_classes, pretrained=pretrained, checkpoint_path=checkpoint_path, token_manager=token_manager, can_tokens_revive=can_tokens_revive)
    
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    return model