# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils, search
from fairseq.data import LanguagePairDataset

from . import register_task
from .translation import TranslationTask, load_langpair_dataset


@register_task("translation_from_pretrained_bart")
class TranslationFromPretrainedBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs',  type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self.langs = args.langs.split(",")
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, "max_source_positions", 1024),
            max_target_positions=getattr(self.args, "max_target_positions", 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, "prepend_bos", False),
            append_source_id=True,
        )

    def build_generator(self, models, args, prefix_allowed_tokens_fn=None, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator

            # Choose search strategy. Defaults to Beam Search.
            sampling = getattr(args, "sampling", False)
            sampling_topk = getattr(args, "sampling_topk", -1)
            sampling_topp = getattr(args, "sampling_topp", -1.0)
            diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
            diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
            match_source_len = getattr(args, "match_source_len", False)
            diversity_rate = getattr(args, "diversity_rate", -1)
            constrained = getattr(args, "constraints", False)
            if prefix_allowed_tokens_fn is None:
                prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
            if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
            ):
                raise ValueError("Provided Search parameters are mutually exclusive.")
            assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
            assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

            if sampling:
                search_strategy = search.Sampling(
                    self.target_dictionary, sampling_topk, sampling_topp
                )
            elif diverse_beam_groups > 0:
                search_strategy = search.DiverseBeamSearch(
                    self.target_dictionary, diverse_beam_groups, diverse_beam_strength
                )
            elif match_source_len:
                # this is useful for tagging applications where the output
                # length should match the input length, so we hardcode the
                # length constraints for simplicity
                search_strategy = search.LengthConstrainedBeamSearch(
                    self.target_dictionary,
                    min_len_a=1,
                    min_len_b=0,
                    max_len_a=1,
                    max_len_b=0,
                )
            elif diversity_rate > -1:
                search_strategy = search.DiverseSiblingsSearch(
                    self.target_dictionary, diversity_rate
                )
            elif constrained:
                search_strategy = search.LexicallyConstrainedBeamSearch(
                    self.target_dictionary, args.constraints
                )
            elif prefix_allowed_tokens_fn:
                search_strategy = search.PrefixConstrainedBeamSearch(
                    self.target_dictionary, prefix_allowed_tokens_fn
                )
            else:
                search_strategy = search.BeamSearch(self.target_dictionary)

            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index("[{}]".format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
        return dataset
