import argparse
import os
import logging
import torch
from mydataset import BaseDataset, ParamDesDataset, MultiEncoderASTDataset,CodeSliceDataset, ASTDataset
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import logging
import pandas as pd
from metrics.metric import calculate_metric
from models import BaseCodeBert, MultiEncoderCodeBert

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, 
                        help="Path to training data")
    parser.add_argument("--valid_data_path", type=str, 
                        help="Path to validation data")
    parser.add_argument("--test_data_path", type=str,
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory")
    parser.add_argument("--checkpoint", type=str,
                        help="different model checkpoint")
    parser.add_argument("--dataset_type", type=str,
                        help="Select different dataset")
    parser.add_argument("--model_type", type=str,
                        help="Select different model")
    parser.add_argument("--test", type=int, default=0,
                        help="If test the model.")
    parser.add_argument("--device", type=int, default=0,
                        help="The device to train.")
    parser.add_argument("--max_input_len", type=int, default=512,
                        help="The max length of code.")
    parser.add_argument("--max_output_len", type=int, default=128,
                        help="The max length of target.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="The gradient accumulation steps.")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="The batch size of train.")
    parser.add_argument("--valid_batch_size", type=int, default=16,
                        help="The batch size of valid.")
    parser.add_argument("--num_train_epochs", type=int, default=50,
                        help="The number of train epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The learning rate of train.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="The adam epsilon.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="The warmup proportion.")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="The beam size.")

    return parser.parse_args()

def train(model, train_data, valid_data, args):
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data,
                                    sampler=train_sampler,
                                    batch_size=args.train_batch_size)

    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("total steps:{}".format(total_steps))
    

    model.to(torch.device("cuda", args.device))
    

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    torch.cuda.empty_cache()

    tr_loss = 0.0
    last_eval_loss = float("inf")
    counter = 0
    model_to_save = model.module if hasattr(model, "module") else model
    save_path = os.path.join(args.output_dir, args.dataset_type + "_" + args.checkpoint + "_checkpoint")


    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        model.train()
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False,  position=0)
        for step, batch in enumerate(iter_bar):
            source_ids = batch["source_ids"].to(torch.device("cuda", args.device))
            source_mask = batch["source_mask"].to(torch.device("cuda", args.device))
            target_ids = batch["target_ids"].to(torch.device("cuda", args.device))
            target_mask = batch["target_mask"].to(torch.device("cuda", args.device))
      
            loss = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            tr_loss += loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)" % (loss.item()))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                logger.info("optimize step:{}".format(step))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        eval_loss = valid(model, valid_data, args)
        if eval_loss < last_eval_loss:
            last_eval_loss = eval_loss
            torch.save(model_to_save, save_path)
            counter = 0
            model_to_save = model.module if hasattr(model, "module") else model
        else:
            counter += 1
            if counter >= 3:
                return model_to_save
            
        test(model, args.device, tokenizer, args)
        
        torch.cuda.empty_cache()

def valid(model, valid_data, args):
    valid_sampler = SequentialSampler(valid_data)
    valid_data_loader = DataLoader(valid_data, sampler=valid_sampler,
                                  batch_size=args.valid_batch_size)
    iter_bar = tqdm(valid_data_loader, desc="iter", disable=False, position=0)
    total_loss, total = 0.0, 0.0
    
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(iter_bar):
            source_ids = batch["source_ids"].to(torch.device("cuda", args.device))
            source_mask = batch["source_mask"].to(torch.device("cuda", args.device))
            target_ids = batch["target_ids"].to(torch.device("cuda", args.device))
            target_mask = batch["target_mask"].to(torch.device("cuda", args.device))
            loss = model.forward(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            total_loss += loss.item()
            total += len(batch["source_ids"])
        valid_loss = total_loss / total
        return valid_loss

def test(model, device, tokenizer, args):
    model = model.to(device)
    model.eval()
    results = []
    test_dataset = eval(args.dataset_type)(file_name=args.test_data_path, tokenizer=tokenizer,
                                                        max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    test_sampler = SequentialSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset, sampler=test_sampler,
                                batch_size=1)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_data_loader)):
            source_ids = batch["source_ids"].to(device)
            source_mask = batch["source_mask"].to(device)
            labels = batch["target_ids"].to(device)
            hypothesis = model.generate(source_ids=source_ids, source_mask=source_mask)
            reference = tokenizer.decode(labels.squeeze(0), skip_special_tokens=True)
            results.append({"hypothesis": hypothesis, "reference": reference})
    with open(os.path.join(args.output_dir, "test_results.jsonl"), "w") as f:
        for result in results:
            f.write(str(result) + "\n")
    
    calculate_metric(results, key_hyp="hypothesis", key_ref="reference")
  
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    args = set_parser()
    
    tokenizer = AutoTokenizer.from_pretrained(r"")
    logger.info("load " + args.dataset_type + " dataset...")
    train_dataset = eval(args.dataset_type)(file_name=args.train_data_path, tokenizer=tokenizer,
                                                        max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    valid_dataset = eval(args.dataset_type)(file_name=args.valid_data_path, tokenizer=tokenizer,
                                                        max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    
    logger.info("load model...")
    model=eval(args.model_type)(path=r"", args=args)
    
    if args.test == 0:
        logger.info("start training...")
        model = train(model, train_dataset, valid_dataset, args)
    else:
        # model = torch.load(os.path.join(args.output_dir, args.dataset_type + "_" + args.checkpoint + "_checkpoint"))
        test(model, args.device, tokenizer, args)