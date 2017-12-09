#!/usr/bin/python

from kivy.app import App
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.clock import Clock

_CARS_COUNT = 2


class Car(Widget):
	def __init__(self, car_idx):
		self.color = (1, 1, 1, 1)
		self.pos = Vector((car_idx + 1) * 5, (car_idx + 1) * 5)

		Widget.__init__(self)

		self.add_widget(Sensor(Vector(-7, -15) + self.pos, (1, 0, 0, 1)))
		self.add_widget(Sensor(Vector(-15, 0) + self.pos, (0, 1, 0, 1)))
		self.add_widget(Sensor(Vector(-7, 15) + self.pos, (0, 0, 1, 1)))

		self.velocity = Vector(car_idx + 1, 0)

	def move(self):
		self.pos = self.velocity + self.pos

		for widget in self.children:
			widget.move()


class Sensor(Widget):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos

		Widget.__init__(self)

	def move(self):
		self.pos = self.parent.velocity + self.pos


class Map(Widget):

	def __init__(self):
		self.width = 20
		self.height = 20

		Widget.__init__(self)

	def build(self, cars_count):
		for i in range(0, cars_count):
			self.add_widget(Car(i))

	def update(self, dt):
		for widget in self.children:
			widget.move()


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
