from pathlib import Path
from ignite.engine.events import State

import torch
try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers

from spring_amr import ROOT
from spring_amr.optim import RAdam
from spring_amr.evaluation import write_predictions, compute_smatch, predict_amrs
from spring_amr.utils import instantiate_model_and_tokenizer, instantiate_loader

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
import numpy as np
import random
from ignite.handlers import ModelCheckpoint, global_step_from_engine

def do_train(checkpoint=None, direction='amr', split_both_decoder=False, fp16=False, seed=1):


    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    # print(model)
    # print(model.config)

    if checkpoint is not None:
        print(f'Checkpoint restored ({checkpoint})!')

    optimizer = RAdam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    if 'bart-large' in config['model']:
        if config['warm_start']:
            model_name = 'bart-large'
        else:
            model_name = 'transformer-large'
    elif 'bart-base' in config['model']:
        if config['warm_start']:
            model_name = 'bart-base'
        else:
            model_name = 'transformer-base'
    if 'amr_annotation_3.0' in config['train']:
        version = '3.0'
        where_checkpoints="AMR3.0"
    else:
        where_checkpoints="AMR2.0"
        version = '2.0'

    dev_gold_path = ROOT / f'data/tmp/AMR{version}dev-gold.txt'
    dev_pred_path = ROOT / f'data/tmp/{where_checkpoints}.txt'
    print(f'dev pred path: {dev_pred_path}')
    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=True, out=dev_gold_path,
        use_recategorization=config['use_recategorization'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    def train_step(engine, batch):
        model.train()
        x, y, extra = batch
        model.amr_mode = True
        with autocast(enabled=fp16):
            loss, *_ = model(**x, **y)
        scaler.scale((loss / config['accum_steps'])).backward()
        return loss.item()

    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()
        x, y, extra = batch
        model.amr_mode = True
        loss, *_ = model(**x, **y)
        return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['trn_text_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def smatch_eval(engine):
        device = next(model.parameters()).device
        dev_loader.device = device
        graphs = predict_amrs(dev_loader, model, tokenizer, restore_name_ops=config['collapse_name_ops'])
        write_predictions(dev_pred_path, tokenizer, graphs)
        try:
            smatch = compute_smatch(dev_gold_path, dev_pred_path)
        except:
            smatch = 0.
        engine.state.metrics['dev_smatch'] = smatch

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['dev_text_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | bleu: {engine.state.metrics['dev_bleu']:.3f}"
        print(log_msg)

 
    RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_amr_loss')
    RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_amr_loss')

    if config['log_wandb']:
        from ignite.contrib.handlers.wandb_logger import WandBLogger
        wandb_logger = WandBLogger(init=False)

   
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="iterations/trn_amr_loss",
            output_transform=lambda loss: loss
        )
        
        metric_names_trn = ['trn_amr_loss']
        metric_names_dev = ['dev_amr_loss']
        metric_names_dev.append('dev_smatch')

        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_trn,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_dev,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def wandb_log_lr(engine):
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=engine.state.iteration)

    prefix = 'best-smatch'
    score_function = lambda x: evaluator.state.metrics['dev_smatch']
    to_save = {'model': model, 'optimizer': optimizer}
    root = ROOT/'runs'
    where_checkpoints = str(root) + '/' + where_checkpoints
    handler = ModelCheckpoint(
        where_checkpoints,
        prefix,
        n_saved=1,
        create_dir=True,
        score_function=score_function,
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    model.cuda()
    device = next(model.parameters()).device
    SC_train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        training_form='SC',
        SC_steps=config['SC_steps']
    )
    SC_train_loader.device = device
    print(where_checkpoints)
    print(f"[SC Training]")
    trainer.run(SC_train_loader, max_epochs=1)
    print('[SC Training Finished]')
    trainer.state = State()
    IC_train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        training_form='IC',
        IC_steps=config['IC_steps']
    )
    IC_train_loader.device = device
    print(where_checkpoints)
    print(f"[IC Training]")
    trainer.run(IC_train_loader, max_epochs=1)
    print('[IC Training Finished]')
    trainer.state = State()
    train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        training_form='origin'
    )
    print(where_checkpoints)
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml

    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--direction', type=str, default='amr', choices=['amr', 'text', 'both'],
        help='Train a uni- (amr, text) or bidirectional (both).')
    parser.add_argument('--split-both-decoder', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--IC_steps', type=int, default=500)
    parser.add_argument('--SC_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_epochs', type=int, default=30)


    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    set_seed(args.seed)

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    config['SC_steps'] = args.SC_steps
    config['IC_steps'] = args.IC_steps

    if args.lr:
        config['learning_rate'] = args.lr
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs

    if config['log_wandb']:
        wandb.init(
            entity="SOME-RUNS",
            project="SOME-PROJECT",
            config=config,
            dir=str(ROOT / 'runs/'))
        config = wandb.config

    print(config)

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    do_train(
        checkpoint=checkpoint,
        direction=args.direction,
        split_both_decoder=args.split_both_decoder,
        fp16=args.fp16,
        seed=args.seed
    )