# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Paths for named models in the zoo """

GRAPH_BACKBONE_MODELS = {
    "public": {
        "s3_uri": "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_backbone_v1.0.pt",
        "data": "Generate Structure ETL: July 25 2022",
        "task": "BLNL backbone model training with EMA, trained July 2023",
    },
}

GRAPH_CLASSIFIER_MODELS = {
    "public": {
        "s3_uri": "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_proclass_v1.0.pt",
        "data": "Generate Structure ETL: June 2022",
        "task": "Backbone classification model training with cross-entropy loss",
    },
}

GRAPH_DESIGN_MODELS = {
    "public": {
        "s3_uri": "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_design_v1.0.pt",
        "data": "Generate Structure ETL: July 25 2022",
        "task": "Autoregressive joint prediction of sequence and chi angles, two-stage",
    },
}

PROCAP_MODELS = {
    "public": {
        "s3_uri": "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_procap_v1.0.pt",
        "data": "Generate Structure ETL: June 2022",
        "task": "Backbone caption model training with cross-entropy loss, using M5 ProClass GNN embeddings",
    },
}

NAMED_MODELS = {
    "GraphBackbone": GRAPH_BACKBONE_MODELS,
    "GraphDesign": GRAPH_DESIGN_MODELS,
    "GraphClassifier": GRAPH_CLASSIFIER_MODELS,
    "ProteinCaption": PROCAP_MODELS,
}
