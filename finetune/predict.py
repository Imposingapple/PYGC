import argparse
import os
from pytorch_lightning import Trainer
import re
import json
# debug专用
### 开始
import sys
sys.path.append("/home/haoping/Projects/chinese_spell_checking/PYGC")
from finetune.train import CSCTask


def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--ckpt_path", required=True, type=str, help="ckpt file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default='/home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv',
         type=str, help="label file")
    # 以下为apple新加的
    parser.add_argument("--vocab_path", default="./datas/", help="path of vocab.txt & vocab_pinyin.txt")
    parser.add_argument(
        "--pretrained_hanzi_lm", 
        default='hfl/chinese-roberta-wwm-ext', 
        type=str
    )
    parser.add_argument(
        "--pretrained_pinyin_lm_file", 
        default='./checkpoints/pinyin_bert-finetuned-wikizh_electra/checkpoint-233130', 
        type=str
    )
    parser.add_argument("--multitask", action="store_true", default=False, help="det and cor")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument("--save_topk", default=5, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default='train', type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CSCTask(args)

    trainer = Trainer.from_argparse_args(args)
    if '15'in args.label_file:
        output=trainer.predict(model=model,dataloaders=model.test15_dataloader(),ckpt_path=args.ckpt_path)
    else: # cscd-ime dataset
        output=trainer.predict(model=model,dataloaders=model.test_cscd_dataloader(),ckpt_path=args.ckpt_path)

    from metrics.metric import Metric
    metric = Metric(vocab_path=args.vocab_path)
    with open(os.path.join(args.save_path, "score.txt"), 'w') as f:
        # 第一遍
        pred_txt_path = os.path.join(args.save_path, "preds_1.txt")
        pred_lbl_path = os.path.join(args.save_path, "labels_1.txt")
        results = metric.metric(
                batches=output,
                pred_txt_path=pred_txt_path,
                pred_lbl_path=pred_lbl_path,
                label_path=args.label_file,
                should_remove_de=False
            )
        print(results)
        f.write(json.dumps(results, indent=4)+'\n')
        # 第二遍
        pred_txt_path = os.path.join(args.save_path, "preds_2.txt")
        pred_lbl_path = os.path.join(args.save_path, "labels_2.txt")
        for ex in output:
            ex['pred_idx'] = ex['post_pred_idx']
        results = metric.metric(
                batches=output,
                pred_txt_path=pred_txt_path,
                pred_lbl_path=pred_lbl_path,
                label_path=args.label_file,
                should_remove_de=False
            )
        print(results)
        f.write(json.dumps(results, indent=4)+'\n')


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()