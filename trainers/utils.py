def get_trainer(args, model, logger):
    if args.task == "cls":
        if args.method == "erm":
            from trainers import CLSTrainer

            return CLSTrainer(args, model, logger)
    elif args.task == "seg":
        if args.method == "erm":
            from trainers import SegTrainer

            return SegTrainer(args, model, logger)
    else:
        raise NotImplementedError
