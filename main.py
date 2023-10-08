"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pcr.engines.defaults import default_argument_parser, default_config_parser, default_setup, Trainer
from pcr.engines.launch import launch
import os


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()

    args.config_file = "./configs/s3dis/semseg-ptv2m2-0-base.py"
    args.num_gpus = 2
    args.options = {'save_path': 'exp/s3dis/semseg-ptv2m2-0-proto-8'}

    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    config_file = ''
    main()
