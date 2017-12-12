#!/usr/bin/python

import numpy

from kivy.app import App
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Color
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate, Translate
from kivy.clock import Clock

_CARS_COUNT = 2


class Car(RelativeLayout):
	def __init__(self, car_idx):
		self.pos = Vector((car_idx + 1) * 250, (car_idx + 1) * 250)
		self.velocity = 5 + car_idx * 2
		self.orientation = Vector(-1, 0)
		RelativeLayout.__init__(self)

		self.body = Body(Vector(-5, -5), (1, 1, 1, 1))
		self.middle_sensor = Sensor(Vector(-20, -5), (1, 0, 0, 1))
		self.right_sensor = Sensor(Vector(-12, 10), (0, 0, 1, 1))
		self.left_sensor = Sensor(Vector(-12, -20), (0, 1, 0, 1))
		self.center_mark = Center()

		self.add_widget(self.body)
		self.add_widget(self.middle_sensor)
		self.add_widget(self.right_sensor)
		self.add_widget(self.left_sensor)
		self.add_widget(self.center_mark)

	def move(self):
		with self.canvas.before:
			PushMatrix()
			r = Rotate()
			r.angle = 5

		with self.canvas.after:
			PopMatrix()

		self.orientation = self.orientation.rotate(5)
		self.pos = self.orientation * self.velocity + self.pos
		"""
		self.left_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.left_sensor.abs_pos.x) - 10 : int(self.left_sensor.abs_pos.x) + 10,
					int(self.left_sensor.abs_pos.y) - 10 : int(self.left_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.
		self.middle_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.middle_sensor.abs_pos.x) - 10 : int(self.middle_sensor.abs_pos.x) + 10,
					int(self.middle_sensor.abs_pos.y) - 10 : int(self.middle_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.
		self.right_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.right_sensor.abs_pos.x) - 10 : int(self.right_sensor.abs_pos.x) + 10,
					int(self.right_sensor.abs_pos.y) - 10 : int(self.right_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.
		"""

class Center(Widget):
	pass


class Body(Widget):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos

		Widget.__init__(self)



class Sensor(Widget):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos

		Widget.__init__(self)

	@property
	def abs_pos(self):
		return self.parent.pos + self.pos

class Map(Widget):

	def __init__(self):
		self.width = 20
		self.height = 20

		Widget.__init__(self)

	def build(self, cars_count):
		for i in range(0, cars_count):
			self.add_widget(Car(i))

	def update(self, dt):
		for car in self.children:
			car.move()


class CarApp(App):
	def __init__(self):
		App.__init__(self)

		self.__map = Map()

	def build(self):
		self.__map.build(_CARS_COUNT)

		Clock.schedule_interval(self.__map.update, 1.0/60.0)

		return self.__map


if __name__ == '__main__':
    CarApp().run()
