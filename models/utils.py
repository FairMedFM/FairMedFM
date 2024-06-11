import torch
import models
import json


def get_model(args):
    model_setting = args.model_setting

    model_name = getattr(models, args.model)

    model = model_name()

    if model_setting is not None and "pretrained_path" in model_setting.keys():
        model.from_pretrained(model_setting["pretrained_path"])

    return model
