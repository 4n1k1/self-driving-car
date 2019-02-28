#!/usr/bin/python

import os

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window

from bl import Root
from absl import app, flags


Window.size = (1280, 720)

_DEFAULT_CARS_COUNT = 1
_MAX_CARS_COUNT = 6


class Main(App):
	def __init__(self):
		App.__init__(self)

		self.__map = Root()

	def build(self):
		self.__map.build(args.cars_count)

		Clock.schedule_interval(self.__map.update, 1.0/30.0)

		return self.__map


def main(_):
	Main().run()


if __name__ == '__main__':
	flags.DEFINE_integer("cars_count", _DEFAULT_CARS_COUNT, "Number of cars to train", upper_bound=6)
	flags.DEFINE_boolean("use_pytorch", False, "Use pytorch AI implementation")
	flags.DEFINE_float("learning_rate", 10.0, "Defines gradient descent step modifier value.")
	flags.DEFINE_float("discount_factor", 0.9, "")
	app.run(main)
