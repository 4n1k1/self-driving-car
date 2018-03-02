#!/usr/bin/python

import argparse
import os

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.logger import Logger

from bl import Root

Window.size = (1280, 720)

_DEFAULT_CARS_COUNT = 1
_MAX_CARS_COUNT = 6

class Main(App):
	def __init__(self):
		App.__init__(self)

		self.__map = Root()

	def build(self):
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"-c", "--cars_count",
			help="display a square of a given number",
			type=int,
			default=_DEFAULT_CARS_COUNT
		)
		args = parser.parse_args()

		if args.cars_count > 6:
			Logger.warning("zOrg app: maximum cars number exceeded, falling back to 6")
			args.cars_count = 6

		self.__map.build(args.cars_count)

		Clock.schedule_interval(self.__map.update, 1.0/60.0)

		return self.__map


if __name__ == '__main__':
    Main().run()
