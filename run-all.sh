#!/bin/bash
python main.py --type pretrained --prefix bert-base-enhanced-18 --config resources/config/bert-base-enhanced.yaml > resources/log/bert-base-enhanced-18.txt
python main.py --type pretrained --prefix roberta-base-enhanced-18 --config resources/config/roberta-base-enhanced.yaml > resources/log/roberta-base-enhanced-18.txt
python main.py --type pretrained --prefix xlnet-base-enhanced-18 --config resources/config/xlnet-base-enhanced.yaml > resources/log/xlnet-base-enhanced-18.txt
python main.py --type pretrained --prefix bert-large-enhanced-18 --config resources/config/bert-large-enhanced.yaml > resources/log/bert-large-enhanced-18.txt
python main.py --type pretrained --prefix roberta-large-enhanced-18 --config resources/config/roberta-large-enhanced.yaml > resources/log/roberta-large-enhanced-18.txt
python main.py --type pretrained --prefix xlnet-large-enhanced-18 --config resources/config/xlnet-large-enhanced.yaml > resources/log/xlnet-large-enhanced-18.txt