# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ChemProt"""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
ADE dataset.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

class ChemprotConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ChemprotConfig, self).__init__(**kwargs)


class Chemprot(datasets.GeneratorBasedBuilder):
    """Ade Version 1.0."""

    BUILDER_CONFIGS = [
        ChemprotConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://www.sciencedirect.com/science/article/pii/S1532046412000615?via%3Dihub/",
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train_abs"],
                "dev": self.config.data_files["dev_abs"],
                "test": self.config.data_files["test_abs"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)

        f = list()
        with open(filepath, 'r') as file:
            for line in file:
                json_line = json.loads(line.strip())
                f.append(json_line)
        for id_, row in enumerate(f):
            triplets = ''
            prev_head = None
            text = row['title'] + ' ' + row['abstract']
            relations = row['relations']
            for relation in relations:
                head = relation['head']
                tail = relation['tail']
                label = relation['label']
                if prev_head == head:
                    triplets += f' <subj> ' + tail + f' <obj> ' + label
                elif prev_head == None:
                    triplets += '<triplet> ' + head + f' <subj> ' + tail + f' <obj> ' + label
                    prev_head = head
                else:
                    triplets += ' <triplet> ' + head + f' <subj> ' + tail + f' <obj> ' + label
                    prev_head = head
            yield str(row["doc_key"]), {
                "title": str(row["doc_key"]),
                "context": text,
                "id": str(row["doc_key"]),
                "triplets": triplets,
            }