#!/usr/bin/env python

import argparse
import os

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.logger import Logger

from root import Root


Window.size = (1280, 720)

_DEFAULT_CARS_COUNT = 1
_MAX_CARS_COUNT = 6
_SCALE_FACTOR = 300


class Main(App):
    def __init__(self):
        App.__init__(self)

        self._root = Root()

    def build(self):
        def _bool(value):
            if value == "1":
                return True
            else:
                return False

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-c", "--cars_count",
            help="display a square of a given number",
            type=int,
            default=_DEFAULT_CARS_COUNT,
        )
        parser.add_argument(
            "-s", "--scale_factor",
            help="softmax scale factor",
            type=int,
            default=_SCALE_FACTOR,
        )
        parser.add_argument(
            "-up", "--use_pytorch",
            help="Use pytorch framework for NN",
            type=_bool,
            default=True,
        )
        parser.add_argument(
            "-ws", "--write_status_file",
            help="Write simple visualization in text file",
            type=_bool,
            default=False,
        )

        args = parser.parse_args()

        if args.cars_count > _MAX_CARS_COUNT:
            Logger.warning("zOrg app: maximum cars number exceeded, falling back to 6")
            args.cars_count = _MAX_CARS_COUNT

        self._root.build(args)

        Clock.schedule_interval(self._root.update, 1.0 / 30.0)

        return self._root


if __name__ == '__main__':
    Main().run()
