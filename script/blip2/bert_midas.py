import argparse
from datetime import datetime
import os
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification

from evaluation.attattr import attention_attribute, scaled_input
from evaluation.midas import midas_evaluator, midas_scorer
from model_gradient.modeling_bert import BertForSequenceClassification as B2
from utils import activate_attn, get_module_logger, read_jsonls_w_source


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--caption_dir",
        help="Directory for captions",
    )
    parser.add_argument(
        "-m",
        "--memes_dir",
        help="Directory for hateful memes",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory for output bert results",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=100,
        type=int,
        help="# epoch for training",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--resume_exp",
        action="store_true",
    )
    parser.add_argument(
        "-eval",
        "--eval_only",
        action="store_true",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        help="Checkpoint path from output_dir. Works if resume_exp is True",
    )
    parser.add_argument(
        "-ts",
        "--train_set",
        default="train,dev_seen",
        help="HM subsets for training",
    )
    parser.add_argument(
        "-es",
        "--eval_set",
        default="test_seen",
        help="HM subsets for training",
    )
    parser.add_argument(
        "-en",
        "--exp_name",
        default="dev",
        help="Experiment name",
    )
    parser.add_argument(
        "-cf",
        "--conf_path",
        default="path/to/conf",
        help="Path for confounder.parquet",
    )
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "-rs",
        "--random_seed",
        default=1987,
        type=int,
        help="Random seed for torch",
    )
    args = parser.parse_args()
    return args


def load_inputs(
    args,
    export_cols_org=["id", "label", "split"],
    caption_col="text_captioned",
    text_col="text",
    id_col="id",
):
    """
    Load inputs
    """
    export_cols = export_cols_org + [
        "caption",
        text_col,
    ]
    df_caption = pd.read_parquet(
        f"{args.caption_dir}/hm_captions.parquet", engine="pyarrow"
    )[["image_id", caption_col]].rename(
        {"image_id": id_col, caption_col: "caption"}, axis=1
    )
    df_caption[id_col] = df_caption[id_col].astype(str).str.zfill(5)
    df_memes = read_jsonls_w_source(args.memes_dir)[
        export_cols_org
        + [
            text_col,
        ]
    ]
    df_memes[id_col] = df_memes[id_col].astype(str).str.zfill(5)
    df_out = pd.merge(df_caption, df_memes, left_on=id_col, right_on="id")
    return df_out[export_cols]


def load_model(args, model_name="bert-base-uncased"):
    """
    Load model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    )
    model_gradient = B2.from_pretrained(model_name, output_attentions=True)
    if args.resume_exp:
        model_dict = torch.load(
            f"{args.output_dir}/{args.exp_name}/{args.checkpoint_path}/bert_gradient.model",
            map_location=torch.device("cpu"),
        )
        model_gradient.load_state_dict(model_dict)
    else:
        pass
    return tokenizer, model, model_gradient


def remove_2w(text):
    # remove double whitespaces
    return " ".join(str(text).split())


def tokenize_2sentences(tokenizer, text, caption):
    """
    Tokenize text and caption
    """
    text_tokenized = tokenizer(
        remove_2w(text),
        remove_2w(caption),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
    )
    return text_tokenized


def tokenize_2sentence_batch(tokenizer, df):
    """
    Tokenize texts and captions
    """
    texts_tokenized = df.apply(
        lambda x: tokenize_2sentences(tokenizer, x["text"], x["caption"]),
        axis=1,
    )
    return texts_tokenized


class HMDataset(Dataset):
    def __init__(self, encodings, labels, image_ids):
        self.encodings = encodings
        self.labels = labels
        self.image_ids = image_ids

    def __getitem__(self, idx, id_col="id"):
        item = {key: val for key, val in self.encodings[idx].items()}
        # item = {key: val[idx] for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx])
        item[id_col] = self.image_ids[idx]
        return item

    def __len__(self):
        return len(self.image_ids)


def activate_layer(model, layer_num=-1):
    for param in model.bert.encoder.layer[layer_num].parameters():
        param.requires_grad = True
    return model


def initialize_layer(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def set_model_activation(model, layer_nums=[-1, -2, -3, -4]):
    model = initialize_layer(model)
    for layer_num in layer_nums:
        model = activate_layer(model, layer_num)
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def set_model_optim(model, layer_nums=[-1, -2, -3, -4]):
    init_params = [{"params": model.classifier.parameters(), "lr": 1e-4}]
    optimizer = optim.Adam(
        init_params
        + [
            {"params": model.bert.encoder.layer[layer].parameters(), "lr": 5e-5}
            for layer in layer_nums
        ]
    )
    return optimizer


def set_model(model, layer_nums=[-1, -2, -3, -4]):
    model = set_model_activation(model, layer_nums=layer_nums)
    optimizer = set_model_optim(model, layer_nums=layer_nums)
    return model, optimizer


def train_loop(
    model,
    train_loader,
    optimizer,
    device,
    logger,
    args,
    return_dict=True,
):
    epoch = 1 if args.debug else args.epoch
    model = model.to(device)
    model.train()
    for i in range(epoch):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            input_ids = input_ids.reshape(
                (
                    input_ids.shape[0],
                    input_ids.shape[-1],
                )
            ).to(device)
            attention_mask = batch["attention_mask"]
            attention_mask = attention_mask.reshape(
                (attention_mask.shape[0], attention_mask.shape[-1])
            ).to(device)
            token_type_ids = batch["token_type_ids"]
            token_type_ids = token_type_ids.reshape(
                (token_type_ids.shape[0], token_type_ids.shape[-1])
            ).to(device)
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=return_dict,
            )
            loss = outputs.loss
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            if args.debug:
                break
        logger.info(f"Loss for epoch {i}: {loss}")
    if not args.debug:
        torch.save(
            model_gradient.cpu().state_dict(),
            f"{args.output_dir}/{args.exp_name}/{now}/bert_gradient.model",
        )
    return model


def _eval_loop(
    model,
    dev_loader,
    device,
    logger,
    args,
    return_dict=True,
    attention_weights=None,
    softmax=nn.Softmax(dim=1),
    id_col="id",
):
    gts = []
    predictions = []
    attentions = []
    probabilities = []
    iids = []
    for i, batch in enumerate(dev_loader):
        image_ids = batch[id_col]
        input_ids = batch["input_ids"]
        input_ids = input_ids.reshape(
            (
                input_ids.shape[0],
                input_ids.shape[-1],
            )
        ).to(device)
        attention_mask = batch["attention_mask"]
        attention_mask = attention_mask.reshape(
            (attention_mask.shape[0], attention_mask.shape[-1])
        ).to(device)
        token_type_ids = batch["token_type_ids"]
        token_type_ids = token_type_ids.reshape(
            (token_type_ids.shape[0], token_type_ids.shape[-1])
        ).to(device)
        labels = batch["label"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=return_dict,
            attention_weights=attention_weights,
        )

        score, atts = outputs.logits, outputs.attentions
        _, pred = torch.max(score, 1)

        proba = softmax(score)

        gts += list(labels.cpu().numpy())
        predictions += list(pred.cpu().numpy())
        attentions += [att.cpu() for att in atts]
        probabilities += list(proba[:, 1].cpu().numpy())
        iids += list(image_ids)
        if args.debug:
            break
    return predictions, gts, attentions, probabilities


def _eval_loop_gradient(
    model,
    dev_loader,
    device,
    logger,
    args,
    attention_weights,
    return_dict=True,
    softmax=nn.Softmax(dim=1),
    num_steps=10,
    tokenizer=None,
    predictions=None,
    probabilities=None,
    id_col="id",
    label_col="label",
):
    assert (
        attention_weights is not None
    ), "attntion_weights is required for gradient calculation"
    model.train()
    out = {
        id_col: [],
        "pred": probabilities,
        label_col: predictions,
    }
    for i, batch in enumerate(dev_loader):
        attention_weight = attention_weights[i]
        image_ids = batch[id_col]
        num_batch = len(image_ids)
        input_ids = batch["input_ids"]
        input_ids = input_ids.reshape(
            (
                input_ids.shape[0],
                input_ids.shape[-1],
            )
        ).to(device)
        attention_mask = batch["attention_mask"]
        attention_mask = attention_mask.reshape(
            (attention_mask.shape[0], attention_mask.shape[-1])
        ).to(device)
        token_type_ids = batch["token_type_ids"]
        token_type_ids = token_type_ids.reshape(
            (token_type_ids.shape[0], token_type_ids.shape[-1])
        ).to(device)
        labels = batch["label"].to(device)
        scaled_att = scaled_input(attention_weight, num_steps)
        for step in range(num_steps):
            att_step = scaled_att[step]
            att_step = activate_attn(att_step, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=return_dict,
                attention_weights=att_step,
            )

            score = outputs.logits
            proba = softmax(score)

            for sample_num in range(proba.shape[0]):
                gradient = torch.autograd.grad(
                    torch.unbind(proba[sample_num, :]),
                    att_step,
                    retain_graph=True,
                )[0]
                if sample_num == 0:
                    gradient_step = gradient
                else:
                    gradient_step += gradient
            if i == 0:
                gradient_batch = gradient_step
            else:
                gradient_batch += gradient_step
        attattr = attention_attribute(
            attention_weight,
            gradient_batch.cpu(),
            num_steps,
        )
        scores = {
            "attention": attention_weight,
            "attattr": attattr,
            "gradient": gradient_batch.cpu(),
        }
        midas = midas_scorer(input_ids, scores, tokenizer, device)
        out[id_col] += list(image_ids)
        for k in midas.keys():
            for k2 in midas[k].keys():
                if i == 0:
                    out[k + "_" + k2] = midas[k][k2][:num_batch].tolist()
                else:
                    out[k + "_" + k2] += midas[k][k2][:num_batch].tolist()
        if args.debug:
            break
    return out


def eval_loop(
    model,
    dev_loader,
    device,
    logger,
    args,
    target_names=["benign", "hateful"],
    return_dict=True,
    attention_weights=None,
    softmax=nn.Softmax(dim=0),
    tokenizer=None,
    predictions=None,
    pred_labels=None,
    probabilities=None,
):
    model = model.to(device)
    if attention_weights is None:
        with torch.no_grad():
            predictions, gts, attentions, probabilities = _eval_loop(
                model,
                dev_loader,
                device,
                logger,
                args,
                return_dict=return_dict,
                attention_weights=attention_weights,
                softmax=softmax,
            )
            intermediates = [predictions, attentions, probabilities]
    else:
        intermediates = _eval_loop_gradient(
            model,
            dev_loader,
            device,
            logger,
            args,
            return_dict=return_dict,
            attention_weights=attention_weights,
            softmax=softmax,
            tokenizer=tokenizer,
            predictions=predictions,
            probabilities=probabilities,
        )

    if not attention_weights:
        result = classification_report(
            predictions,
            gts,
            target_names=target_names,
        )
        result_auc = roc_auc_score(gts, probabilities)
        result_acc = accuracy_score(gts, predictions)
        logger.info(
            f"""
            Eval complete w/ classification report: \n{result}
            \n{args.eval_set} set AUC: {result_auc}
            \n{args.eval_set} set Accuracy: {result_acc}
            """
        )
    else:
        logger.info("Gradient extracted")

    return model, intermediates


def preprocess(df, subset, tokenizer, logger, mode_str="Train", batch_size=16):
    set_df = df[df["split"].isin(subset)].reset_index(drop=True)
    logger.info(f"{mode_str} set length: {len(set_df)}")
    tokenized = tokenize_2sentence_batch(tokenizer, set_df)
    logger.info(f"Tokenized inputs w/ keys: {tokenized[0].keys()}")
    logger.info(f"Input_ids: {tokenized[0]['input_ids'][0].shape}")
    logger.info(f"Attention_mask: {tokenized[0]['attention_mask'][0].shape}")
    logger.info(f"Token_type_ids: {tokenized[0]['token_type_ids'][0].shape}")
    ds = HMDataset(tokenized, set_df["label"].values, set_df["id"].values)
    logger.info(f"Created dataset w/ length: {len(ds)}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logger.info(f"{mode_str} dataloader length: {len(loader)}")
    return loader


if __name__ == "__main__":
    t0 = time.time()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    args = arg_parser()
    assert not args.resume_exp or args.checkpoint_path is not None
    if not args.resume_exp:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)
        os.makedirs(
            f"{args.output_dir}/{args.exp_name}/{now}",
            exist_ok=True,
        )
    logger = get_module_logger(__name__)
    torch.manual_seed(args.random_seed)

    df = load_inputs(args)
    logger.info(f"Loaded inputs with length {len(df)}")

    tokenizer, model, model_gradient = load_model(args)
    logger.info("Loaded model")

    model_gradient, optimizer = set_model(model_gradient)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.eval_only:
        train_loader = preprocess(
            df, args.train_set.split(","), tokenizer, logger, mode_str="Train"
        )
        model_gradient = train_loop(
            model_gradient,
            train_loader,
            optimizer,
            device,
            logger,
            args,
        )
        logger.info("Fine-tuning finished")
    else:
        pass
    eval_loader = preprocess(
        df, args.eval_set.split(","), tokenizer, logger, mode_str="Eval"
    )

    model.eval()
    model_gradient, intermediates = eval_loop(
        model_gradient,
        eval_loader,
        device,
        logger,
        args,
    )
    predictions, attentions, probabilities = intermediates
    model_gradient, midas_dict = eval_loop(
        model_gradient,
        eval_loader,
        device,
        logger,
        args,
        tokenizer=tokenizer,
        attention_weights=attentions,
        predictions=predictions,
        probabilities=probabilities,
    )
    logger.info("Eval complete")
    _ = midas_evaluator(args, logger, pd.DataFrame(midas_dict), conf_modality="img", pick_correct_labels=False)
    _ = midas_evaluator(args, logger, pd.DataFrame(midas_dict), conf_modality="img", pick_correct_labels=True)
    _ = midas_evaluator(args, logger, pd.DataFrame(midas_dict), conf_modality="txt", pick_correct_labels=False)
    _ = midas_evaluator(args, logger, pd.DataFrame(midas_dict), conf_modality="txt", pick_correct_labels=True)

    logger.info(f"All process complete w/ {time.time()-t0} seconds")
