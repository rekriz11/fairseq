#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints negative_constraints mask_constraints forced_slot_constraints disjoint_slot_constraints slot_delimiters constraint_type")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)
    #print("cfg.generation.constraints: {}".format(cfg.generation.constraints))

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints and negative constraints, if present, from input lines,
        # "&&"-delimited two types of constraint,
        # store them in batch_constraints and batch_negative_constraints
        batch_constraints = [list() for _ in lines]
        batch_negative_constraints = [list() for _ in lines]
        batch_mask_constraints = [list() for _ in lines]
        batch_forced_slot_constraints = [list() for _ in lines]
        batch_disjoint_slot_constraints = [list() for _ in lines]
        batch_slot_delimiters = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                if "@@" in line:
                    ## Line is of the form:
                    ## <input>\t<AA>&&<BB>@@<CC>&&<DD>&&<EE>
                    ## AA = tab delimited positive constraints
                    ## BB = tab-delimited negative constraints
                    ## CC = tab-delimited forced slot constraints
                    ## DD = tab-delimited disjoint slot constraints
                    ## EE = tab-delimited delimiters used to start/reset disjoint constraints
                    other_line_info, disjoint_constraint_info = line.split("@@")
                    forced_constraint, disjoint_constraint, delimiter = disjoint_constraint_info.split("&&")
                    ## Splitting everything by tab
                    if "&&" in other_line_info:
                        line_constraint, negative_constraint = other_line_info.split("&&")
                        lines[i], *batch_constraints[i] = line_constraint.split("\t")
                        *batch_negative_constraints[i], = negative_constraint.split("\t")
                    else:
                        lines[i], *batch_constraints[i] = other_line_info.split("\t")
                    *batch_forced_slot_constraints[i], = forced_constraint.split("\t")
                    *batch_disjoint_slot_constraints[i], = disjoint_constraint.split("\t")
                    *batch_slot_delimiters[i], = delimiter.split("\t")
                elif "&&" in line:
                    ## Line is of the form <input>\t<AA>&&<BB>
                    line_constraint, negative_constraint = line.split("&&")
                    lines[i], *batch_constraints[i] = line_constraint.split("\t")
                    *batch_negative_constraints[i], = negative_constraint.split("\t")
                else:
                    # By default, only use positive constraints
                    lines[i], *batch_constraints[i] = line.split("\t")
        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list if constraint
            ]
        #print("batch_constraints: {}".format([c.shape for c in batch_constraints[0]]))
        for i, negative_constraint_list in enumerate(batch_negative_constraints):
            batch_negative_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(negative_constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for negative_constraint in negative_constraint_list if negative_constraint
            ]
        #print("batch_forced_slot_constraints: {}".format(batch_forced_slot_constraints))
        for i, forced_constraint_list in enumerate(batch_forced_slot_constraints):
            batch_forced_slot_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(forced_constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for forced_constraint in forced_constraint_list if forced_constraint
            ]
        #print("batch_forced_slot_constraints after encoding: {}".format(batch_forced_slot_constraints))
        #print("batch_disjoint_slot_constraints: {}".format(batch_disjoint_slot_constraints))
        for i, disjoint_constraint_list in enumerate(batch_disjoint_slot_constraints):
            batch_disjoint_slot_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(disjoint_constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for disjoint_constraint in disjoint_constraint_list if disjoint_constraint
            ]
        #print("batch_disjoint_slot_constraints after encoding: {}".format(batch_disjoint_slot_constraints))
        #print("batch_slot_delimiters: {}".format(batch_slot_delimiters))
        for i, delimiter_list in enumerate(batch_slot_delimiters):
            batch_slot_delimiters[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(delimiter),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for delimiter in delimiter_list if delimiter
            ]
        #print("batch_slot_delimiters after encoding: {}".format(batch_slot_delimiters))
        ## Option to mask invalid subwords
        if cfg.generation.constraints in ['ordered_mask', 'unordered_mask', 'mask']:
            null_encoded = task.target_dictionary.encode_line(
                encode_fn_target('null'),
                append_eos=False,
                add_if_not_exist=False,
            )
            #print("null_encoded: {}".format(null_encoded.shape))
            for i, constraint_list in enumerate(batch_mask_constraints):
                line_encoded = task.target_dictionary.encode_line(
                    encode_fn_target(lines[i]),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                #print("line_encoded: {}, {}".format(line_encoded.shape, lines[i]))
                batch_mask_constraints[i] = torch.cat([line_encoded, null_encoded], dim=0)
                for constraint in batch_constraints[i]:
                    batch_mask_constraints[i] = torch.cat([batch_mask_constraints[i], constraint], dim=0)
            #print("batch_mask_constraints: {}".format(batch_mask_constraints))
        else:
            batch_mask_constraints = torch.tensor(batch_mask_constraints)

        constraints_tensor = pack_constraints(batch_constraints)
        negative_constraints_tensor = pack_constraints(batch_negative_constraints)
        constraints = {"positive": constraints_tensor, "negative": negative_constraints_tensor, "mask": batch_mask_constraints, \
        'forced': batch_forced_slot_constraints, 'disjoint': batch_disjoint_slot_constraints, 'delimiters': batch_slot_delimiters, 'constraint_type': cfg.generation.constraints}
    else:
        constraints = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)
        negative_constraints = batch.get("negative_constraints", None)
        mask_constraints = batch.get("mask_constraints", None)
        forced_slot_constraints = batch.get("forced_slot_constraints", None)
        disjoint_slot_constraints = batch.get("disjoint_slot_constraints", None)
        slot_delimiters = batch.get("slot_delimiters", None)
        constraint_type = batch.get("constraint_type", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
            negative_constraints=negative_constraints,
            mask_constraints=mask_constraints,
            forced_slot_constraints=forced_slot_constraints,
            disjoint_slot_constraints=disjoint_slot_constraints,
            slot_delimiters=slot_delimiters,
            constraint_type=constraint_type
        )


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    #logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    #logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    #if cfg.generation.constraints:
    #    logger.warning(
    #        "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
    #    )

    #if cfg.interactive.buffer_size > 1:
    #    logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    #logger.info("NOTE: hypothesis and token scores are output in base 2")
    #logger.info("Type the input sentence and press return:")
    start_id = 0
    for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            negative_constraints = batch.negative_constraints
            mask_constraints = batch.mask_constraints
            forced_slot_constraints = batch.forced_slot_constraints
            disjoint_slot_constraints = batch.disjoint_slot_constraints
            slot_delimiters = batch.slot_delimiters
            constraint_type = batch.constraint_type
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()
                if negative_constraints is not None:
                    negative_constraints = negative_constraints.cuda()
            if constraints is not None and negative_constraints is not None:
                constraints_dict = dict()
                constraints_dict["positive"] = constraints
                constraints_dict["negative"] = negative_constraints
                constraints_dict['mask'] = mask_constraints
                constraints_dict['forced'] = forced_slot_constraints
                constraints_dict['disjoint'] = disjoint_slot_constraints
                constraints_dict['delimiters'] = slot_delimiters
                constraints_dict['constraint_type'] = constraint_type

            else:
                constraints_dict = None
            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translate_start_time = time.time()
            translations = task.inference_step(
                generator, models, sample, constraints=constraints_dict
            )
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            list_negative_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
                list_negative_constraints = [unpack_constraints(c, cfg.generation.constraints) for c in negative_constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                negative_constraints = list_negative_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "negative_constraints": negative_constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ""
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                print("S-{}\t{}".format(id_, src_str))
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            id_,
                            tgt_dict.string(constraint, cfg.common_eval.post_process),
                        )
                    )
                #for negative_constraint in info["negative_constraints"]:
                #    print("N-{}\t{}".format(
                #            id_, tgt_dict.string(negative_constraint, cfg.common_eval.post_process)
                #        )
                #    )

            # Process top predictions
            for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(id_, score, hypo_str))
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
                print(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )
                if cfg.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    print("A-{}\t{}".format(id_, alignment_str))

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
