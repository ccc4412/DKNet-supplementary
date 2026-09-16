# -*- coding: utf-8 -*-
# 让 `python hashet_zh/run.py` 与包内相对导入两种方式都可运行
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
