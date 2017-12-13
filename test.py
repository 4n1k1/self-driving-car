#!/usr/bin/python

import numpy

from kivy.app import App
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate, Translate
from kivy.clock import Clock
from kivy.core.window import Window

_CARS_COUNT = 2
_CHECK_POINT_OFFEST = Vector(50, 50)
_ROTATIONS = (0, 20, -20)


class _RGBAColor:
	RED = (1, 0, 0, 1)
	GREEN = (0, 1, 0, 1)
	BLUE = (0, 0, 1, 1)
	WHITE = (1, 1, 1, 1)
	GREY = (0.5, 0.5, 0.5, 1)


class Car(RelativeLayout):
	def __init__(self, car_idx):
		self.pos = Vector((car_idx + 1) * 250, (car_idx + 1) * 250)
		self.velocity = 5 + car_idx * 2
		self.direction = Vector(-1, 0)
		self.destination = Vector(0, 0)
		self.action = 0

		self.reward = 0.0
		self.state = (0.0, 0.0, 0.0, 0.0, 0.0)
		self.scores = []

		RelativeLayout.__init__(self)

		self.body = Body(Vector(-5, -5), _RGBAColor.WHITE)
		self.middle_sensor = Sensor(Vector(-20, -5), _RGBAColor.RED)
		self.right_sensor = Sensor(Vector(-12, 10), _RGBAColor.GREEN)
		self.left_sensor = Sensor(Vector(-12, -20), _RGBAColor.BLUE)
		self.center_mark = Center()

		self.add_widget(self.body)
		self.add_widget(self.middle_sensor)
		self.add_widget(self.right_sensor)
		self.add_widget(self.left_sensor)
		self.add_widget(self.center_mark)

	def move(self):
		angle_of_rotation = _ROTATIONS[self.action]

		with self.canvas.before:
			PushMatrix()
			r = Rotate()
			r.angle = _ROTATIONS[angle_of_rotation]

		with self.canvas.after:
			PopMatrix()

		self.direction = self.direction.rotate(angle_of_rotation)
		self.pos = self.direction * self.velocity + self.pos

		self.brain.update(
			self._get_reward(),
			self._get_state(),
		)

		self.action = self.brain.pick_action()

		self.scores.append(self.brain.score())

	def _get_state(self):
		self.orientation = self.direction.angle(self.destination)/180.0
		self.left_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.left_sensor.abs_pos.x) - 10 : int(self.left_sensor.abs_pos.x) + 10,
					int(self.left_sensor.abs_pos.y) - 10 : int(self.left_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.0
		self.middle_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.middle_sensor.abs_pos.x) - 10 : int(self.middle_sensor.abs_pos.x) + 10,
					int(self.middle_sensor.abs_pos.y) - 10 : int(self.middle_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.0
		self.right_sensor_signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.right_sensor.abs_pos.x) - 10 : int(self.right_sensor.abs_pos.x) + 10,
					int(self.right_sensor.abs_pos.y) - 10 : int(self.right_sensor.abs_pos.y) + 10,
				]
			)
		) / 400.0

		return (
			self.left_sensor_signal,
			self.middle_sensor_signal,
			self.right_sensor_signal,
			self.orientation,
			-self.orientation,
		)

	def _get_reward(self):
		current_position = Vector(*self.pos[:2])

		distance = numpy.sqrt((self.pos.x - goal_x)**2 + (self.car.y - goal_y)**2)

		self.ball1.pos = self.car.sensor1
		self.ball2.pos = self.car.sensor2
		self.ball3.pos = self.car.sensor3

		if sand[int(self.car.x),int(self.car.y)] > 0:
			self.car.velocity = Vector(1, 0).rotate(self.car.angle)
			last_reward = -0.5
		else:
			self.car.velocity = Vector(6, 0).rotate(self.car.angle)
			last_reward = -0.2
			if distance < last_distance:
				last_reward = 0.1

		if self.car.x < 10:
			self.car.x = 10
			last_reward = -1
		if self.car.x > self.width - 10:
			self.car.x = self.width - 10
			last_reward = -1
		if self.car.y < 10:
			self.car.y = 10
			last_reward = -1
		if self.car.y > self.height - 10:
			self.car.y = self.height - 10
			last_reward = -1

		if distance < 100:
			goal_x = self.width-goal_x
			goal_y = self.height-goal_y

		last_distance = distance

		return last_reward


class Center(Widget):
	pass


class DeparturePoint(Widget):
	def __init__(self):
		Widget.__init__(self)

		self.pos = Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		with self.canvas:
			Color(rgba=_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)


class Destination(Widget):
	def __init__(self):
		Widget.__init__(self)

		self.pos = Vector(*Window.size) - Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		with self.canvas:
			Color(rgba=_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)


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
		# self.parent.pos + self.pos = list of 4 (x, y, z, w I guess)
		return Vector(*(self.parent.pos + self.pos)[:2])


class Painter(Widget):

	def __init__(self):
		Widget.__init__(self)

		self.__last_x = 0
		self.__last_y = 0
		self.__length = 0
		self.__n_points = 0

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0.8,0.7,0)
			touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)

			self.__last_x = int(touch.x)
			self.__last_y = int(touch.y)
			self.__n_points = 0
			self.__length = 0

			self.parent.sand[int(touch.x),int(touch.y)] = 1

	def on_touch_move(self, touch):
		if touch.button == 'left':
			touch.ud['line'].points += [touch.x, touch.y]
			self.__length += numpy.sqrt(max((touch.x - self.__last_x)**2 + (touch.y - self.__last_y)**2, 2))
			self.__n_points += 1.
			density = self.__n_points/(self.__length)
			touch.ud['line'].width = int(20 * density + 1)
			self.parent.sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
			self.__last_x = touch.x
			self.__last_y = touch.y


class Map(Widget):

	def __init__(self):
		self.cars = []
		self.size = Window.size
		self.pos = (0, 0)

		Widget.__init__(self)

		self.sand = numpy.zeros(
			(self.width, self.height,)
		)

	def build(self, cars_count):
		self.add_widget(DeparturePoint())
		self.add_widget(Destination())
		self.add_widget(Painter())

		for i in range(0, cars_count):
			car = Car(i)

			self.cars.append(car)
			self.add_widget(car)

	def update(self, dt):
		for car in self.cars:
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
