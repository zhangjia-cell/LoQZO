# -*- coding: utf-8 -*-
"""
run_mezo.py 轻量包装版。

建议：
- 以后优先使用统一入口 run.py + quzo.sh。
- 如果你仍希望保留 run_mezo.py 这个文件名，可以直接把它替换成这个包装版。

这个包装版会直接复用同目录下 run.py 的 main()。
你只需要在 shell 脚本里把 --trainer 设成 mezo（或 zo）即可。
"""
from run import main

if __name__ == "__main__":
    main()
