import argparse

parser = argparse.ArgumentParser()

yml_path = {
    "mit-states": './config/cluspro/mit-states.yml',
    "ut-zappos": './config/cluspro/ut-zappos.yml',
    "cgqa": './config/cluspro/cgqa.yml'
}
# model config
parser.add_argument("--model_name", help="model name", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--optimizer", help="optimizer", type=str, default='Adam')
parser.add_argument("--scheduler", help="optimizer", type=str, default='StepLR')
parser.add_argument("--step_size", help="optimizer", type=int, default=5)
parser.add_argument("--gamma", help="optimizer", type=float, default=0.5)
parser.add_argument("--val_metric", help="optimizer", type=str, default="AUC")
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=32, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=64, type=int)
parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--clip_arch", help="clip path", type=str, default="ViT-L/14")
parser.add_argument("--prompt_template", help="prompt_template", type=list, default=["a photo of x x", "a photo of x", "a photo of x"])
parser.add_argument("--ctx_init", help="ctx_init", type=list, default=["a photo of ", "a photo of ", "a photo of "])
parser.add_argument("--dataset_path", help="dataset path", type=str)
parser.add_argument("--save_path", help="save path", type=str,default="data/model/ut-zappos/zanshi")
parser.add_argument("--save_every_n", default=20, type=int, help="saves the model every n epochs")
parser.add_argument("--save_final_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
parser.add_argument("--same_prim_sample", help="if sample same prim samples", action="store_true")

parser.add_argument("--open_world", help="evaluate on open world setup", default=False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=5128, type=int)
parser.add_argument('--threshold', type=float, default=None, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")

parser.add_argument("--adapter_dim", help="middle dimension of Adapter", type=int, default=64)
parser.add_argument("--adapter_dropout", help="adapter_dropout", type=float, default=0.1)
parser.add_argument("--pair_loss_weight", help="pair_loss_weight", type=float, default=1.0)
parser.add_argument("--pair_inference_weight", help="adapter_dropout", type=float, default=1.0)
parser.add_argument("--attr_loss_weight", help="adapter_dropout", type=float, default=1.0)
parser.add_argument("--attr_inference_weight", help="adapter_dropout", type=float, default=1.0)
parser.add_argument("--obj_loss_weight", help="adapter_dropout", type=float, default=1.0)
parser.add_argument("--obj_inference_weight", help="adapter_dropout", type=float, default=1.0)
parser.add_argument("--init_lamda", help="lamda initialization value", type=float, default=0.1)
parser.add_argument("--cmt_layers", help="Number of layers in cross-attention", type=int, default=2)
parser.add_argument("--cluster_num", help="prototype number", type=int, default=5)
