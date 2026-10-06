import torch
from fairmedfm.models.utils import get_model
from fairmedfm.wrappers.utils import get_warpped_model

def get_trainer(args, model, logger, test_dataloader=None):
    if args.task == "cls":
        if args.method == "erm":
            from fairmedfm.trainers import CLSTrainer
            return CLSTrainer(args, model, test_dataloader, logger)
        elif args.method == "sa":
            from fairmedfm.trainers import SATrainer
            return SATrainer(args, model, test_dataloader, logger)
        else:
            raise NotImplementedError
    elif args.task == "seg":
        if args.method == "erm":
            from fairmedfm.trainers import SegTrainer
            return SegTrainer(args, model, logger)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
